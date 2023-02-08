import numpy as np
import mdtraj as md

def get_value_plumed(file,arg):
    f=open(file,'r')
    lines=f.readlines()
    reqline=None
    for l in lines:
        wrds=l.split(' ')
        if wrds[0]=='PRINT' and len(wrds[0].strip())>0:
            for wrd in wrds:
                cmnd=wrd.split('=')
                if cmnd[0].strip()==arg:
                    reqline=cmnd[1].strip()
    return reqline

def get_chi(tr,top,key,value):
    chi_atms={}
    if len(key)>3:
        angle=int(key[-1]);
        if angle == 1:
            all_chis=md.compute_chi1(tr)[0]+1
        if angle == 2:
            all_chis=md.compute_chi2(tr)[0]+1
        if angle == 3:
            all_chis=md.compute_chi3(tr)[0]+1
        if angle == 4:
            all_chis=md.compute_chi4(tr)[0]+1
        if angle == 5:        
            all_chis=md.compute_chi5(tr)[0]+1
        chi_atms[key]={}
        for j,chis in enumerate(all_chis):
            resid=int(str(top.atom(chis[0]-1).residue)[3:])
            chi_atms[key][resid]=chis
        if value==[0]:
            return chi_atms
        else:
            req_chi_atms={key:{k :chi_atms[key][k] for k in value if k in list(chi_atms[key].keys())}}
            return req_chi_atms
    else:
        all_chis=[md.compute_chi1(tr)[0]+1,md.compute_chi2(tr)[0]+1,md.compute_chi3(tr)[0]+1,md.compute_chi4(tr)[0]+1,md.compute_chi5(tr)[0]+1]

        for j,chis in enumerate(all_chis):
            chi_atms[f'chi{j+1}']={}
            for i,chiatoms in enumerate(chis):
                resid=int(str(top.atom(chiatoms[0]-1).residue)[3:])
                chi_atms[f'chi{j+1}'][resid]=chiatoms
        return chi_atms
    
def get_phi(tr,top,key,value):
    all_phis=md.compute_phi(tr)[0]+1
    phi_atms={}
    for i,phiatoms in enumerate(all_phis):
        resid=int(str(top.atom(phiatoms[0]-1).residue)[3:])
        phi_atms[resid]=phiatoms
    if value==[0]:
        return phi_atms
    else:
        req_phi_atms={k :phi_atms[k] for k in value if k in list(phi_atms.keys())}
        return req_phi_atms
        
def get_psi(tr,top,key,value):
    all_psis=md.compute_psi(tr)[0]+1
    psi_atms={}
    for i,psiatoms in enumerate(all_psis):
        resid=int(str(top.atom(psiatoms[0]-1).residue)[3:])
        psi_atms[resid]=psiatoms
    if value==[0]:
        return psi_atms
    else:
        req_psi_atms={k :psi_atms[k] for k in value if k in list(psi_atms.keys())}
        return req_psi_atms

def get_dist(tr,top,key,value):
    CA_atoms=[]
    for (a,b) in value:
        CA_a=top.select(f'resid {a-1} and name == CA')[0]+1
        CA_b=top.select(f'resid {b-1} and name == CA')[0]+1
        CA_atoms.append((CA_a,CA_b))
    return CA_atoms

def write_chi(fid,chi_atoms,output):
    for i,chi_angle in enumerate(chi_atoms):
        for res in chi_atoms[chi_angle]:
            chi=chi_atoms[chi_angle][res]
            line1=f'{chi_angle}_{int(res)}: TORSION ATOMS={chi[0]},{chi[1]},{chi[2]},{chi[3]}'
            line2=f'sch{i+1}_r{int(res)}: CUSTOM ARG={chi_angle}_{int(res)} VAR={chi_angle}_{int(res)} FUNC=sin({chi_angle}_{int(res)}) PERIODIC=NO'
            line3=f'cch{i+1}_r{int(res)}: CUSTOM ARG={chi_angle}_{int(res)} VAR={chi_angle}_{int(res)} FUNC=cos({chi_angle}_{int(res)}) PERIODIC=NO'
            output+=f'sch{i+1}_r{int(res)},cch{i+1}_r{int(res)},'
            fid.write(line1+"\n")
            fid.write(line2+"\n")
            fid.write(line3+"\n")
    return output

def write_phi(fid,phi_atoms,output):
    for res in phi_atoms:
        phi=phi_atoms[res]
        line1=f'phi_{int(res)}: TORSION ATOMS={phi[0]},{phi[1]},{phi[2]},{phi[3]}'
        line2=f'sph_r{int(res)}: CUSTOM ARG=phi_{int(res)} VAR=phi_{int(res)} FUNC=sin(phi_{int(res)}) PERIODIC=NO'
        line3=f'cph_r{int(res)}: CUSTOM ARG=phi_{int(res)} VAR=phi_{int(res)} FUNC=cos(phi_{int(res)}) PERIODIC=NO'
        output+=f'sph_r{int(res)},cph_r{int(res)},'
        fid.write(line1+"\n")
        fid.write(line2+"\n")
        fid.write(line3+"\n")
    return output

def write_psi(fid,psi_atoms,output):
    for res in psi_atoms:
        psi=psi_atoms[res]
        line1=f'psi_{int(res)}: TORSION ATOMS={psi[0]},{psi[1]},{psi[2]},{psi[3]}'
        line2=f'sps_r{int(res)}: CUSTOM ARG=psi_{int(res)} VAR=psi_{int(res)} FUNC=sin(psi_{int(res)}) PERIODIC=NO'
        line3=f'cps_r{int(res)}: CUSTOM ARG=psi_{int(res)} VAR=psi_{int(res)} FUNC=cos(psi_{int(res)}) PERIODIC=NO'
        output+=f'sps_r{int(res)},cps_r{int(res)},'
        fid.write(line1+"\n")
        fid.write(line2+"\n")
        fid.write(line3+"\n")
    return output
    
def write_dist(fid,CAs,output):
    for i,(a,b) in enumerate(CAs):
        line1=f'd{i+1}: DISTANCE ATOMS={a},{b}'
        fid.write(line1+"\n")
        output+=f'd{i+1},'
    return output

def plumed_cases(tr,top,fid,key,value,output):
    
    fid.write("\n")
    
    if key[0:3]=='chi':
        chi_atoms=get_chi(tr,top,key,value)
        output=write_chi(fid,chi_atoms,output)
        
    if key[0:3]=='phi':
        phi_atoms=get_phi(tr,top,key,value)
        output=write_phi(fid,phi_atoms,output)
        
    if key[0:3]=='psi':
        psi_atoms=get_psi(tr,top,key,value)
        output=write_psi(fid,psi_atoms,output)
        
    if key[0:3]=='dis':
        CAs=get_dist(tr,top,key,value)
        output=write_dist(fid,CAs,output)
    
    return output
        
def generate_plumed(pdbfile,CVs_interest,plumedfile,stride,outputfile):
    tr=md.load(pdbfile)
    top=tr.topology
    fid_plmd=open(plumedfile,'w')
    output='PRINT ARG='
    for key in list(CVs_interest.keys()):
        output=plumed_cases(tr,top,fid_plmd,key,CVs_interest[key],output)
    output=output[:-1]
    output+=f' STRIDE={stride} FILE={outputfile}'
    fid_plmd.write(output+"\n")
    fid_plmd.close()
