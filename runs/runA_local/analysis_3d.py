import glob
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "vis" / "python"))
import athena_read  # noqa: E402


DEFAULT_QUANTITIES = ["dens", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3", "eint"]


def load_cube(file_path, quantities=None):
    if quantities is None:
        quantities = DEFAULT_QUANTITIES
    cube = athena_read.athdf(file_path, quantities=quantities)
    time = float(cube.get("Time", 0.0))
    return cube, time


def load_cubes(pattern, quantities=None):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    if quantities is None:
        quantities = DEFAULT_QUANTITIES
    cubes = [athena_read.athdf(f, quantities=quantities) for f in files]
    times = np.array([float(c.get("Time", i)) for i, c in enumerate(cubes)], dtype=float)
    return files, cubes, times


def _as_3d(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [x3,x2,x1], got shape {arr.shape}")
    return arr


def _vector_components(cube, field="v", rho_mode="mean"):
    if field == "v":
        return _as_3d(cube["velx"]), _as_3d(cube["vely"]), _as_3d(cube["velz"])
    if field == "b":
        return _as_3d(cube["bcc1"]), _as_3d(cube["bcc2"]), _as_3d(cube["bcc3"])
    if field in ("zplus", "zminus"):
        if rho_mode == "mean":
            inv_sqrt_rho = 1.0 / np.sqrt(float(np.mean(cube["dens"])))
        elif rho_mode == "local":
            inv_sqrt_rho = 1.0 / np.sqrt(_as_3d(cube["dens"]))
        else:
            raise ValueError("rho_mode must be 'mean' or 'local'")
        sgn = 1.0 if field == "zplus" else -1.0
        return (
            _as_3d(cube["velx"]) + sgn * _as_3d(cube["bcc1"]) * inv_sqrt_rho,
            _as_3d(cube["vely"]) + sgn * _as_3d(cube["bcc2"]) * inv_sqrt_rho,
            _as_3d(cube["velz"]) + sgn * _as_3d(cube["bcc3"]) * inv_sqrt_rho,
        )
    raise ValueError(f"Unknown field '{field}'. Use v, b, zplus, zminus.")


def fit_power_law(k, ek, kmin=2, kmax=20):
    mask = (k >= kmin) & (k <= kmax) & (k > 0) & (ek > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.nan
    return np.polyfit(np.log10(k[mask]), np.log10(ek[mask]), 1)[0]


def _rfft_hermitian_weights(nx):
    nxh = nx // 2 + 1
    w = np.ones(nxh, dtype=np.float64)
    if nx % 2 == 0:
        if nxh > 2:
            w[1:-1] = 2.0
    else:
        if nxh > 1:
            w[1:] = 2.0
    return w


def spectra_3d(cube, field="v", subtract_mean=True, work_dtype=np.float64):
    """3D isotropic and reduced spectra from one cube.

    AthenaK array ordering is [x3, x2, x1] = [z, y, x].
    For B0 || z: k_parallel = kz, k_perp = sqrt(kx^2 + ky^2).
    This implementation is memory-leaner than full 3D k-grid construction.
    """
    f1, f2, f3 = _vector_components(cube, field=field)
    nz, ny, nx = f1.shape
    ntot = float(nx * ny * nz)

    kx = np.fft.rfftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz

    ky2, kx2 = np.meshgrid(ky, kx, indexing="ij")
    kperp2 = kx2 * kx2 + ky2 * ky2
    ib_perp_2d = np.rint(np.sqrt(kperp2)).astype(np.int32)
    kperp_max = int(ib_perp_2d.max())

    kz_abs = np.abs(np.rint(kz)).astype(np.int32)
    kpar_max = int(kz_abs.max())
    kmag_max = int(np.ceil(np.sqrt((nx // 2) ** 2 + (ny // 2) ** 2 + (nz // 2) ** 2)))

    # Precompute 2D isotropic-bin map for each unique |kz| to avoid 3D k-grids.
    kmag_bins_by_kz = {}
    for kza in np.unique(kz_abs):
        kmag_bins_by_kz[int(kza)] = np.rint(np.sqrt(kperp2 + float(kza) ** 2)).astype(np.int32).ravel()

    ek = np.zeros(kmag_max + 1, dtype=np.float64)
    ek_perp = np.zeros(kperp_max + 1, dtype=np.float64)
    ek_par = np.zeros(kpar_max + 1, dtype=np.float64)

    hx = _rfft_hermitian_weights(nx)[None, None, :]
    ib_perp_flat = ib_perp_2d.ravel()

    for comp in (f1, f2, f3):
        arr = np.asarray(comp, dtype=work_dtype)
        if subtract_mean:
            arr = arr - np.mean(arr, dtype=np.float64)

        fh = np.fft.rfftn(arr)
        p = (fh.real * fh.real + fh.imag * fh.imag) / (ntot * ntot)
        p *= hx  # compensate half-spectrum along x from rfftn

        # Reduced perp spectrum: sum over z then radial-bin in (kx,ky).
        p_xy = p.sum(axis=0)
        ek_perp += np.bincount(
            ib_perp_flat,
            weights=p_xy.ravel(),
            minlength=kperp_max + 1,
        )

        # Parallel + isotropic spectra: loop over kz planes.
        for iz in range(nz):
            kzbin = int(kz_abs[iz])
            plane = p[iz]
            plane_sum = float(plane.sum())
            ek_par[kzbin] += plane_sum

            ib_mag_flat = kmag_bins_by_kz[kzbin]
            ek += np.bincount(
                ib_mag_flat,
                weights=plane.ravel(),
                minlength=kmag_max + 1,
            )

    return {
        "k": np.arange(ek.size),
        "E_k": ek,
        "k_perp": np.arange(ek_perp.size),
        "E_kperp": ek_perp,
        "k_par": np.arange(ek_par.size),
        "E_kpar": ek_par,
    }


def sf2_global(cube, field="v", r_values=None, n_pairs=200000, seed=1234):
    """Second-order SF in global frame (parallel/perpendicular to global B0 || z)."""
    if r_values is None:
        r_values = np.arange(1, 33, dtype=int)
    r_values = np.asarray(r_values, dtype=int)
    rng = np.random.default_rng(seed)

    f1, f2, f3 = _vector_components(cube, field=field)
    nz, ny, nx = f1.shape
    sf_par = np.zeros_like(r_values, dtype=float)
    sf_perp = np.zeros_like(r_values, dtype=float)

    for ir, r in enumerate(r_values):
        iz0 = rng.integers(0, nz, n_pairs)
        iy0 = rng.integers(0, ny, n_pairs)
        ix0 = rng.integers(0, nx, n_pairs)

        # Parallel increments: along z
        sgn = rng.choice([-1, 1], n_pairs)
        iz1 = (iz0 + sgn * r) % nz
        iy1 = iy0
        ix1 = ix0
        d2 = (
            (f1[iz1, iy1, ix1] - f1[iz0, iy0, ix0]) ** 2
            + (f2[iz1, iy1, ix1] - f2[iz0, iy0, ix0]) ** 2
            + (f3[iz1, iy1, ix1] - f3[iz0, iy0, ix0]) ** 2
        )
        sf_par[ir] = np.mean(d2)

        # Perpendicular increments: in x-y plane
        th = rng.uniform(0.0, 2.0 * np.pi, n_pairs)
        dx = np.rint(r * np.cos(th)).astype(int)
        dy = np.rint(r * np.sin(th)).astype(int)
        bad = (dx == 0) & (dy == 0)
        dx[bad] = 1

        iz1 = iz0
        iy1 = (iy0 + dy) % ny
        ix1 = (ix0 + dx) % nx
        d2 = (
            (f1[iz1, iy1, ix1] - f1[iz0, iy0, ix0]) ** 2
            + (f2[iz1, iy1, ix1] - f2[iz0, iy0, ix0]) ** 2
            + (f3[iz1, iy1, ix1] - f3[iz0, iy0, ix0]) ** 2
        )
        sf_perp[ir] = np.mean(d2)

    return {"r": r_values, "SF2_par": sf_par, "SF2_perp": sf_perp}


def sf2_local(
    cube,
    field="v",
    r_values=None,
    n_pairs=300000,
    mu_par_min=0.85,
    mu_perp_max=0.15,
    seed=1234,
):
    """Second-order SF conditioned on angle to local magnetic field."""
    if r_values is None:
        r_values = np.arange(1, 33, dtype=int)
    r_values = np.asarray(r_values, dtype=int)
    rng = np.random.default_rng(seed)
    eps = 1.0e-30

    f1, f2, f3 = _vector_components(cube, field=field)
    bx, by, bz = _vector_components(cube, field="b")
    nz, ny, nx = f1.shape

    sf_par = np.full_like(r_values, np.nan, dtype=float)
    sf_perp = np.full_like(r_values, np.nan, dtype=float)
    n_par = np.zeros_like(r_values, dtype=int)
    n_perp = np.zeros_like(r_values, dtype=int)

    for ir, r in enumerate(r_values):
        iz0 = rng.integers(0, nz, n_pairs)
        iy0 = rng.integers(0, ny, n_pairs)
        ix0 = rng.integers(0, nx, n_pairs)

        rr = rng.normal(size=(n_pairs, 3))
        rr /= np.linalg.norm(rr, axis=1, keepdims=True) + eps
        dx = np.rint(r * rr[:, 0]).astype(int)
        dy = np.rint(r * rr[:, 1]).astype(int)
        dz = np.rint(r * rr[:, 2]).astype(int)
        bad = (dx == 0) & (dy == 0) & (dz == 0)
        dx[bad] = 1

        ix1 = (ix0 + dx) % nx
        iy1 = (iy0 + dy) % ny
        iz1 = (iz0 + dz) % nz

        d2 = (
            (f1[iz1, iy1, ix1] - f1[iz0, iy0, ix0]) ** 2
            + (f2[iz1, iy1, ix1] - f2[iz0, iy0, ix0]) ** 2
            + (f3[iz1, iy1, ix1] - f3[iz0, iy0, ix0]) ** 2
        )

        bmx = 0.5 * (bx[iz0, iy0, ix0] + bx[iz1, iy1, ix1])
        bmy = 0.5 * (by[iz0, iy0, ix0] + by[iz1, iy1, ix1])
        bmz = 0.5 * (bz[iz0, iy0, ix0] + bz[iz1, iy1, ix1])
        rnorm = np.sqrt(dx * dx + dy * dy + dz * dz)
        bnorm = np.sqrt(bmx * bmx + bmy * bmy + bmz * bmz)
        mu = np.abs((dx * bmx + dy * bmy + dz * bmz) / (rnorm * bnorm + eps))

        mpar = mu >= mu_par_min
        mperp = mu <= mu_perp_max
        n_par[ir] = np.count_nonzero(mpar)
        n_perp[ir] = np.count_nonzero(mperp)
        if n_par[ir] > 0:
            sf_par[ir] = np.mean(d2[mpar])
        if n_perp[ir] > 0:
            sf_perp[ir] = np.mean(d2[mperp])

    return {
        "r": r_values,
        "SF2_par_local": sf_par,
        "SF2_perp_local": sf_perp,
        "N_par": n_par,
        "N_perp": n_perp,
    }


def slope_timeseries(cubes, field="v", kmin=2, kmax=20, which="k_perp"):
    k_out = []
    slope_out = []
    for cube in cubes:
        spec = spectra_3d(cube, field=field)
        if which == "k":
            kvals, evals = spec["k"], spec["E_k"]
        elif which == "k_par":
            kvals, evals = spec["k_par"], spec["E_kpar"]
        else:
            kvals, evals = spec["k_perp"], spec["E_kperp"]
        k_out = kvals
        slope_out.append(fit_power_law(kvals, evals, kmin=kmin, kmax=kmax))
    return np.array(k_out), np.array(slope_out)


def plot_reduced_spectrum(
    spectrum,
    which="k_perp",
    label=None,
    kmin=2,
    kmax=20,
    ref_slope=-5.0 / 3.0,
    ax=None,
):
    """Plot one reduced spectrum and optional fitted slope."""
    if which == "k":
        k, e = spectrum["k"], spectrum["E_k"]
        title = "Isotropic spectrum E(k)"
        xlab = "k"
    elif which == "k_par":
        k, e = spectrum["k_par"], spectrum["E_kpar"]
        title = "Parallel spectrum E(k_parallel)"
        xlab = "k_parallel"
    else:
        k, e = spectrum["k_perp"], spectrum["E_kperp"]
        title = "Perpendicular spectrum E(k_perp)"
        xlab = "k_perp"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    else:
        fig = ax.figure

    mask = (k > 0) & (e > 0)
    ax.loglog(k[mask], e[mask], lw=2.0, label=label or "spectrum")

    slope = fit_power_law(k, e, kmin=kmin, kmax=kmax)
    fit_mask = (k >= kmin) & (k <= kmax) & (k > 0) & (e > 0)
    if np.count_nonzero(fit_mask) >= 2:
        k_fit = k[fit_mask]
        c = np.exp(np.mean(np.log(e[fit_mask]) - slope * np.log(k_fit)))
        ax.loglog(k_fit, c * k_fit**slope, "--", lw=1.5, label=f"fit slope={slope:.2f}")

        # Reference slope, matched in amplitude at kmin of the fit interval.
        i0 = np.where(k_fit == k_fit.min())[0][0]
        c_ref = e[fit_mask][i0] / (k_fit[i0] ** ref_slope)
        ax.loglog(k_fit, c_ref * k_fit**ref_slope, ":", lw=1.5, label=f"ref {ref_slope:.2f}")

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel("power")
    ax.legend(loc="best")
    return fig, ax


def plot_reduced_spectrum_evolution(
    cubes,
    times,
    field="v",
    which="k_perp",
    n_curves=12,
    kmin=2,
    kmax=20,
):
    """Plot time evolution of reduced spectra, color-coded from early to late."""
    n = len(cubes)
    if n < 1:
        raise ValueError("Need at least one cube")
    idx = np.linspace(0, n - 1, min(n_curves, n), dtype=int)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=float(times[idx].min()), vmax=float(times[idx].max()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in idx:
        sp = spectra_3d(cubes[i], field=field)
        if which == "k":
            k, e = sp["k"], sp["E_k"]
            xlab = "k"
        elif which == "k_par":
            k, e = sp["k_par"], sp["E_kpar"]
            xlab = "k_parallel"
        else:
            k, e = sp["k_perp"], sp["E_kperp"]
            xlab = "k_perp"
        m = (k > 0) & (e > 0)
        ax.loglog(k[m], e[m], color=cmap(norm(times[i])), alpha=0.85, lw=1.2)

    # Time-averaged spectrum
    stacked = []
    for cube in cubes:
        sp = spectra_3d(cube, field=field)
        if which == "k":
            stacked.append(sp["E_k"])
            k = sp["k"]
        elif which == "k_par":
            stacked.append(sp["E_kpar"])
            k = sp["k_par"]
        else:
            stacked.append(sp["E_kperp"])
            k = sp["k_perp"]
    eavg = np.mean(np.vstack(stacked), axis=0)
    m = (k > 0) & (eavg > 0)
    ax.loglog(k[m], eavg[m], color="k", lw=2.2, label="time-average")

    slope = fit_power_law(k, eavg, kmin=kmin, kmax=kmax)
    fm = (k >= kmin) & (k <= kmax) & (k > 0) & (eavg > 0)
    if np.count_nonzero(fm) >= 2:
        kf = k[fm]
        c = np.exp(np.mean(np.log(eavg[fm]) - slope * np.log(kf)))
        ax.loglog(kf, c * kf**slope, "--", lw=1.5, color="k", label=f"avg fit={slope:.2f}")

    ax.set_title(f"{field} spectrum evolution ({which})")
    ax.set_xlabel(xlab)
    ax.set_ylabel("power")
    ax.legend(loc="best")
    fig.colorbar(sm, ax=ax, label="time")
    return fig, ax


def plot_sf2(sf_global, sf_local=None):
    """Plot second-order structure functions and anisotropy ratio."""
    r = sf_global["r"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].loglog(r, sf_global["SF2_perp"], label="SF2_perp (global)")
    axes[0].loglog(r, sf_global["SF2_par"], label="SF2_par (global)")
    if sf_local is not None:
        axes[0].loglog(r, sf_local["SF2_perp_local"], "--", label="SF2_perp (local)")
        axes[0].loglog(r, sf_local["SF2_par_local"], "--", label="SF2_par (local)")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("SF2")
    axes[0].set_title("Second-order structure functions")
    axes[0].legend(loc="best")

    ratio_g = sf_global["SF2_perp"] / np.maximum(sf_global["SF2_par"], 1.0e-30)
    axes[1].semilogx(r, ratio_g, label="global perp/par")
    if sf_local is not None:
        ratio_l = sf_local["SF2_perp_local"] / np.maximum(sf_local["SF2_par_local"], 1.0e-30)
        axes[1].semilogx(r, ratio_l, label="local perp/par")
    axes[1].axhline(1.0, color="k", ls=":", lw=1.0)
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("anisotropy ratio")
    axes[1].set_title("SF2 anisotropy ratio")
    axes[1].legend(loc="best")
    return fig, axes


def plot_slope_timeseries(times, slopes, ylabel="slope dlogE/dlogk", label=None):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(times, slopes, lw=2.0, label=label or "slope")
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title("Spectral slope vs time")
    ax.legend(loc="best")
    return fig, ax

# For H200
# mkdir /workspace/turb
# mkdir build-h200
# cd build-h200
# cmake .. \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DKokkos_ENABLE_CUDA=ON \
#   -DKokkos_ARCH_HOPPER90=ON \
#   -DCMAKE_CXX_COMPILER=/workspace/athenak/kokkos/bin/nvcc_wrapper \
#   -DPROBLEM=mhd_forced_box
# cmake --build . -j 24
# CUDA_VISIBLE_DEVICES=0 ./build-h200/src/athena -i inputs/mhd/forced_box_runA_local.athinput -d /workspace/turb
# nohup bash -lc 'cd /workspace/athenak && CUDA_VISIBLE_DEVICES=0 ./build-h200/src/athena -i inputs/mhd/forced_box_runA_local.athinput -d /workspace/turb' \
#   > /workspace/turb/stdout.log 2>&1 &
# tail -f /workspace/turb/stdout.log
