import sys, glob, numpy as np
sys.path.append("vis/python")
import athena_read
from matplotlib import pyplot as plt

f = sorted(glob.glob("/home/norbertm/Uniturb/384_d1-3_dv.w_bcc_x3slice.*.athdf"))

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

def perpspectra_vel(A):
    spec = np.zeros(A['velx'].shape[1]//2)
    fA = np.abs(np.fft.fftshift(np.fft.fftn(A['velx'][0,:,:])))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['vely'][0,:,:])))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['velz'][0,:,:])))**2
    for i in range(A['velx'].shape[1]):
        for j in range(A['velx'].shape[2]):
            k = int(np.sqrt((i-A['velx'].shape[1]/2)**2+(j-A['velx'].shape[1]/2)**2))
            if k < A['velx'].shape[1]/2:
                spec[k] += fA[i,j]
    return spec/A['velx'].shape[1]**2/2

def perpspectra_mag(A):
    spec = np.zeros(A['velx'].shape[1]//2)
    fA = np.abs(np.fft.fftshift(np.fft.fftn(A['bcc1'][0,:,:])))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['bcc2'][0,:,:])))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['bcc3'][0,:,:])))**2
    for i in range(A['velx'].shape[1]):
        for j in range(A['velx'].shape[2]):
            k = int(np.sqrt((i-A['velx'].shape[1]/2)**2+(j-A['velx'].shape[1]/2)**2))
            if k < A['velx'].shape[1]/2:
                spec[k] += fA[i,j]
    return spec/A['velx'].shape[1]**2/2

def perpspectra_zmin(A):
    spec = np.zeros(A['velx'].shape[1]//2)
    fA = np.abs(np.fft.fftshift(np.fft.fftn(A['velx'][0,:,:] - A['bcc1'][0,:,:]/A['dens'][0,:,:]**0.5)))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['vely'][0,:,:] - A['bcc2'][0,:,:]/A['dens'][0,:,:]**0.5)))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['velz'][0,:,:] - A['bcc3'][0,:,:]/A['dens'][0,:,:]**0.5)))**2
    for i in range(A['velx'].shape[1]):
        for j in range(A['velx'].shape[2]):
            k = int(np.sqrt((i-A['velx'].shape[1]/2)**2+(j-A['velx'].shape[1]/2)**2))
            if k < A['velx'].shape[1]/2:
                spec[k] += fA[i,j]
    return spec/A['velx'].shape[1]**2/2

def perpspectra_zpos(A):
    spec = np.zeros(A['velx'].shape[1]//2)
    fA = np.abs(np.fft.fftshift(np.fft.fftn(A['velx'][0,:,:] + A['bcc1'][0,:,:]/A['dens'][0,:,:]**0.5)))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['vely'][0,:,:] + A['bcc2'][0,:,:]/A['dens'][0,:,:]**0.5)))**2 + \
         np.abs(np.fft.fftshift(np.fft.fftn(A['velz'][0,:,:] + A['bcc3'][0,:,:]/A['dens'][0,:,:]**0.5)))**2
    for i in range(A['velx'].shape[1]):
        for j in range(A['velx'].shape[2]):
            k = int(np.sqrt((i-A['velx'].shape[1]/2)**2+(j-A['velx'].shape[1]/2)**2))
            if k < A['velx'].shape[1]/2:
                spec[k] += fA[i,j]
    return spec/A['velx'].shape[1]**2/2


#Show some graphs

# plt.plot(np.arange(len(f)+1),[np.mean(mach(file)) for file in d]); plt.title("Aveage Mach number"); plt.show()
# plt.plot(np.arange(len(f)+1),[np.mean(0.5*file['dens']*vel(file)**2) for file in d]); plt.title("Aveage Kinetic Energy"); plt.show()
# plt.plot(np.arange(len(f)+1),[np.mean(mach(file)) for file in d]); plt.title("Aveage Mach number"); plt.show()
plt.loglog(np.fft.rfftfreq(d[0]['velx'].shape[1]-1),perpspectra_mag(d[125]),label='Magnetic field')
plt.loglog(np.fft.rfftfreq(d[0]['velx'].shape[1]-1),perpspectra_vel(d[125]),label='Velocity')
plt.loglog(np.fft.rfftfreq(d[0]['velx'].shape[1]-1),perpspectra_zmin(d[125]),label='zmin')
plt.loglog(np.fft.rfftfreq(d[0]['velx'].shape[1]-1),perpspectra_zpos(d[125]),label='zpos')
plt.loglog(np.fft.rfftfreq(256-1),np.fft.rfftfreq(256-1)**(-5/3)/10**1,label='Kolmogorov')
plt.legend(); plt.show()
