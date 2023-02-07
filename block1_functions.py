import numpy as np
from sys import stdout
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import pdbfixer

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

