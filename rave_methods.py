import numpy as np
from sys import stdout
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import pdbfixer
    openmm_install=True
except:
    openmm_install=False

def RegSpaceClustering(z, min_dist, max_centers=200, batch_size=100):
    '''Regular space clustering.
    Args:
        data: ndarray containing (n,d)-shaped float data
        max_centers: the maximum number of cluster centers to be determined, integer greater than 0 required
        min_dist: the minimal distances between cluster centers
    '''
    num_observations, d = z.shape
    p = np.hstack((0,np.random.permutation(num_observations-1)+1))
    data = z[p]
    center_list = data[0, :].copy().reshape(d,1)
    centerids=[p[0]+1]
    i = 1
    while i < num_observations:
        x_active = data[i:i+batch_size, :]
        distances = np.sqrt((np.square(np.expand_dims(center_list.T,0) - np.expand_dims(x_active,1))).sum(axis=-1))
        indice = tuple(np.nonzero(np.all(distances > min_dist, axis=-1))[0])
        if len(indice) > 0:
            # the first element will be used
            #print(center_list.shape,x_active.shape,x_active[indice[0]].shape)
            center_list = np.hstack((center_list, x_active[indice[0]].reshape(d,1)))
            centerids.append(p[i+indice[0]]+1)
            i += indice[0]
        else:
            i += batch_size
        if len(centerids) >= max_centers:
            print("%i centers: Exceeded the maximum number of cluster centers!\n"%len(centerids))
            print("Please increase dmin!\n")
            raise ValueError
    return center_list,centerids

def fix_pdb(index):
    """
    fixes the raw pdb from colabfold using pdbfixer.
    This needs to be performed to cleanup the pdb and to start simulation 

    Fixes performed: missing residues, missing atoms and missing Terminals
    """
    raw_pdb=f'pred_{index}.pdb';

    # fixer instance
    fixer = pdbfixer.PDBFixer(raw_pdb)

    #finding and adding missing residues including terminals
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    out_handle = open(f'fixed_{index}.pdb','w')
    PDBFile.writeFile(fixer.topology, fixer.positions, out_handle,keepIds=True)
    
def add_hydrogen(index,forcefield):
    """
    Runs an unbiased simulation on the cluster center using openMM.
    The MD engine also uses plumed for on the fly calculations
    input : raw pdb from colabfold
    forcefields : amber03 and tip3p
    output : fixed_{index}.pdb, unb_{index}.pdb, COLVAR_unb
    """
    print(f'We are at {os.getcwd()}')

    #fixing PDBs to avoid missing residue or terminal issues
    fix_pdb(index);
    pdb_fixed=f'fixed_{index}.pdb'

    #Get the structure and assign force field
    pdb = PDBFile(pdb_fixed) 
    forcefield = ForceField('amber03.xml', 'tip3p.xml')

    # Placing in a box and adding hydrogens, ions and water
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    PDBFile.writeFile(modeller.topology, modeller.positions, open(f'fixed_Hydrogen_{index}.pdb', 'w'))
    

def plot_exp_var(pca,n_comp=10):

  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(5,3));
  plt.plot([-2,n_comp+2],[0.9,0.9],'--k',alpha=0.4)
  plt.plot([-2,n_comp+2],[1,1],'--k',alpha=0.4)
  plt.bar(np.arange(n_comp),pca.explained_variance_ratio_,alpha=0.2,color='g')
  plt.plot(np.arange(n_comp),pca.explained_variance_ratio_.cumsum(),'--or')
  plt.ylabel('Explained Variance',fontsize=17)
  plt.xlabel('Components',fontsize=17)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.ylim([0,1.02])
  plt.xlim([-1,n_comp+1])
  plt.show()

def plot_proj_labels(projections,proj_labels):

  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(4,4));
  sca = plt.scatter(projections[::1,0],projections[::1,1],c=proj_labels[::1],marker='.',s=2);
  cb = fig.colorbar(sca, location='top')
  cb.set_label('Labels',fontsize=15)
  plt.xlabel(f'Projection 1',fontsize=18) ;   
  plt.ylabel(f'Projection 2',fontsize=18);
  plt.xticks(fontsize=15);
  plt.yticks(fontsize=15);


def get_clusters(projections,colvar_list,num_init_labels,random_seed=10):

  from sklearn.cluster import KMeans

  proj_kmeans = KMeans(n_clusters=num_init_labels,random_state=random_seed).fit(projections)
  proj_labels = proj_kmeans.labels_
  unique_labels = np.unique(proj_labels)
  plot_proj_labels(projections[::10,:2],proj_labels[::10])

  start_ind = 0
  labels = []
  one_hot_encoding_labels = np.eye(num_init_labels,dtype=np.int8)[proj_labels]

  for ii,tr in enumerate(colvar_list):
    stop_ind = start_ind + tr.shape[0]
    one_hot_per_traj = one_hot_encoding_labels[start_ind:stop_ind,:]
    labels.append(one_hot_per_traj)
    start_ind = stop_ind

  return labels


def get_tica_labels(colvar_list,num_init_labels,dimensions,time_lag):

  import tqdm
  from deeptime.decomposition import TICA

  tica = TICA(lagtime=time_lag, dim=dimensions)

  for f in tqdm.tqdm(range(len(colvar_list))):
    shape = colvar_list[f].shape[0]

    if shape < time_lag + 1:
        print(f"Skipping {f} with {shape} frames.")
        continue
    try:
        tica.partial_fit((colvar_list[f][:-time_lag,:], colvar_list[f][time_lag:,:]))

    except ValueError:

        print(f"Skipping {f}.")

  tica_output = [tica.transform(colvar_list[k]) for k in range(len(colvar_list))]

  labels = get_clusters(np.concatenate(tica_output),colvar_list,num_init_labels)

  return labels, tica_output

def get_pca_labels(colvar_list,num_init_labels,dimensions):

  from sklearn.decomposition import PCA

  colvars_all = np.concatenate(colvar_list)

  n_comp=10
  pca = PCA(n_components=n_comp).fit(colvars_all)  # fit the 2-dimensional data
  plot_exp_var(pca,n_comp=10)

  pca_projection = pca.transform(colvars_all)

  labels = get_clusters(pca_projection[:,:dimensions],colvar_list,num_init_labels)

  pca_output = [pca.transform(colvar_list[k]) for k in range(len(colvar_list))]

  return labels, pca_output

def get_time_labels(colvar_list):

  num_initial_states = len(colvar_list)*2
  labels = []

  for ii in range(len(colvar_list)):

    lentraj = len(colvar_list[ii])
    zeroone = np.hstack([np.zeros(int(lentraj/2),dtype=np.int8),np.ones(lentraj-int(lentraj/2),dtype=np.int8)])
    initlabels = np.eye(num_initial_states)[zeroone+int(ii*2)]
    labels.append(initlabels)

  print(f"Total number of initial labels: {num_initial_states}\n")

  return labels


def create_labels(colvar_file_list,label='tica',num_init_labels=100,time_lag=50,dimensions=2):

  colvar_list = [np.load(ff) for ff in colvar_file_list]
  
  if label.upper() not in ['TICA','PCA','TIME']:
    
    raise ValueError('not part of ("TICA","PCA","TIME")')
    return None

  elif label.upper() == 'TICA':

    labels, tica_outputs = get_tica_labels(colvar_list,num_init_labels,dimensions,time_lag)
    return labels, tica_outputs
  
  elif label.upper() == 'PCA':

    labels, pca_outputs = get_pca_labels(colvar_list,num_init_labels,dimensions)
    return labels, pca_outputs
  
  elif label.upper() == 'TIME':

    labels = get_time_labels(colvar_list)
    return labels

def generate_config(traj_data,initial_labels,dt = 0 ,d = 1,beta =1e-3,learning_rate = 1e-3,encoder_type = "Linear",batch_size = 512,neuron_num1 = 16,neuron_num2 = 16):
    
    f=open('config.ini','w')

    f.write("\n[Model Parameters]\n")
    f.write(f"\ndt = [{dt}]\n")
    f.write(f"\nd = [{d}]\n")
    f.write(f"\nencoder_type = {encoder_type}\n")
    f.write(f"\nneuron_num1 = [{neuron_num1}]\n")
    f.write(f"\nneuron_num2 = [{neuron_num2}]\n")

    f.write("\n[Training Parameters]\n")
    
    f.write(f'\nbatch_size ={batch_size}\n')
    f.write(f'\nbeta =[{beta}]\n')
    f.write(f'\nlearning_rate =[{learning_rate}]\n')
    f.writelines(["\nthreshold = 0.01\n", \
    "\npatience = 2\n", \
    "\nrefinements = 8\n", \
    "\nlr_scheduler_step_size = 5\n", \
    "\nlr_scheduler_gamma = 0.9\n", \
    "\nlog_interval = 10000\n"])
    
    f.write("\n[Data]\n")
    f.write("\ntraj_data = [%s]\n"%",".join(traj_data))
    f.write("\ninitial_labels = [%s]\n"%",".join(initial_labels))
    f.write("\ntraj_weights \n")

    f.writelines(["\n[Other Controls]\n", \
    "\nseed = [0]\n", \
    "\nUpdateLabel = True\n", \
    "\nSaveTrajResults = True\n"])
    
    f.close()
