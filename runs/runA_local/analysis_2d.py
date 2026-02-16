import sys, glob, numpy as np
sys.path.append("vis/python")
import athena_read
from matplotlib import pyplot as plt

f = sorted(glob.glob("/home/norbertm/Uniturb/bin/local_128.w_bcc_x3slice.*.athdf"))
# f = sorted(glob.glob("/home/norbertm/Uniturb/512_d1-3_dp.w_bcc_x3slice.*.athdf"))

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


def snapshot_time(A, idx):
    if "Time" in A:
        return float(A["Time"])
    return float(idx)


def plot_spectra_evolution(datasets, spec_fn, label, ax, times=None, cmap_name="coolwarm", stride=10):
    if len(datasets) == 0:
        return
    if times is None:
        times = np.array([snapshot_time(ds, i) for i, ds in enumerate(datasets)], dtype=float)
    else:
        times = np.asarray(times, dtype=float)

    cmap = plt.get_cmap(cmap_name)
    tmin, tmax = float(np.min(times)), float(np.max(times))
    denom = (tmax - tmin) if (tmax > tmin) else 1.0
    plot_idx = list(range(0, len(datasets), max(1, int(stride))))
    if plot_idx[-1] != len(datasets) - 1:
        plot_idx.append(len(datasets) - 1)

    all_specs = []
    for i in plot_idx:
        ds = datasets[i]
        spec = spec_fn(ds)
        all_specs.append(spec)
        k = np.arange(spec.size, dtype=float)
        color = cmap((times[i] - tmin) / denom)
        m = k > 0
        ax.loglog(k[m], spec[m], color=color, alpha=0.75, lw=1.0)

    mean_spec = np.mean(np.array(all_specs), axis=0)
    k = np.arange(mean_spec.size, dtype=float)
    m = k > 0
    ax.loglog(k[m], mean_spec[m], color="k", lw=2.0, label="time average")
    ax.set_title(label)
    ax.set_xlabel(r"$k_\perp$")
    ax.set_ylabel("power")
    ax.legend(loc="best")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=tmin, vmax=tmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="time")


#Show some graphs

# plt.plot(np.arange(len(f)+1),[np.mean(mach(file)) for file in d]); plt.title("Aveage Mach number"); plt.show()
# plt.plot(np.arange(len(f)+1),[np.mean(0.5*file['dens']*vel(file)**2) for file in d]); plt.title("Aveage Kinetic Energy"); plt.show()
# plt.plot(np.arange(len(f)+1),[np.mean(mach(file)) for file in d]); plt.title("Aveage Mach number"); plt.show()
plt.loglog(np.fft.rfftfreq(d[-1]['velx'].shape[1]-1),perpspectra_mag(d[-1]),label='Magnetic field')
plt.loglog(np.fft.rfftfreq(d[-1]['velx'].shape[1]-1),perpspectra_vel(d[-1]),label='Velocity')
plt.loglog(np.fft.rfftfreq(d[-1]['velx'].shape[1]-1),perpspectra_zmin(d[-1]),label='zmin')
plt.loglog(np.fft.rfftfreq(d[-1]['velx'].shape[1]-1),perpspectra_zpos(d[-1]),label='zpos')
plt.loglog(np.fft.rfftfreq(d[-1]['velx'].shape[1]-1),np.fft.rfftfreq(d[-1]['velx'].shape[1]-1)**(-5/3)/10**1,label='Kolmogorov')
plt.loglog(np.fft.rfftfreq(d[-1]['velx'].shape[1]-1),np.fft.rfftfreq(d[-1]['velx'].shape[1]-1)**(-3/2)/10**1,label='IK')
plt.loglog(np.fft.rfftfreq(d[-1]['velx'].shape[1]-1),np.fft.rfftfreq(d[-1]['velx'].shape[1]-1)**(-2)/10**1,label='-2')
plt.legend(); plt.show()


# Evolution plots across all loaded snapshots (color-coded by time)
if len(d) > 0:
    times = np.array([snapshot_time(ds, i) for i, ds in enumerate(d)], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plot_spectra_evolution(d, perpspectra_vel, "Velocity spectrum evolution", axes[0, 0], times=times, stride=10)
    plot_spectra_evolution(d, perpspectra_mag, "Magnetic spectrum evolution", axes[0, 1], times=times, stride=10)
    plot_spectra_evolution(d, perpspectra_zmin, "z- spectrum evolution", axes[1, 0], times=times, stride=10)
    plot_spectra_evolution(d, perpspectra_zpos, "z+ spectrum evolution", axes[1, 1], times=times, stride=10)
    plt.show()
