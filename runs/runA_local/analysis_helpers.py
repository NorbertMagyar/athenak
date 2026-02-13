import sys, glob, numpy as np
sys.path.append("vis/python")
import athena_read

f = sorted(glob.glob("runs/runA_local/bin/local_128.w_bcc_full.*.athdf"))

d = list(range(len(f)))
for i,file in enumerate(f):
   print(i)
   d[i] = athena_read.athdf(file, quantities=["dens","velx","vely","velz","bcc1","bcc2","bcc3","eint"])

def vel(file):
    return np.sqrt(file['velx']**2+file['vely']**2+file['velz']**2)

def mag(file): 
    return np.sqrt(file['bcc1']**2+file['bcc2']**2+file['bcc3']**2)

def zmin(file):
    return np.sqrt((file['velx']-file['bcc1']/np.sqrt(file['dens']))**2 +
                   (file['vely']-file['bcc2']/np.sqrt(file['dens']))**2 +
                   (file['velz']-(file['bcc3']-1)/np.sqrt(file['dens']))**2)

def zpos(file):
    return np.sqrt((file['velx']+file['bcc1']/np.sqrt(file['dens']))**2 +
                   (file['vely']+file['bcc2']/np.sqrt(file['dens']))**2 +
                   (file['velz']+(file['bcc3']-1)/np.sqrt(file['dens']))**2)

def mach(file):
    return vel(file)/mag(file)/np.sqrt(file['dens'])

