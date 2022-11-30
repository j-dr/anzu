import numpy as np
import sys

sys.path.insert(0, "/global/cfs/projectdirs/cosmosim/slac/jderose/anzu/")
from fields.common_functions import _get_resampler, tracer_power
from fields.field_level_bias import measure_field_level_bias
from nbodykit.io import CSVFile
import h5py as h5


def field_level_bias(
    pos,
    field_dict,
    field_D,
    nmesh,
    kmax,
    Lbox,
    pm,
    resampler,
    M=None,
    comm=None,
    interlaced=True,
):
    tracerfield, pk_tt_dict, pm = tracer_power(
        pos, resampler, Lbox, nmesh, pm=pm, rsd=False, interlaced=interlaced, comm=comm
    )

    if "1m" in field_dict:
        dm = field_dict.pop("1m")
        d = field_D[0]
        field_D = field_D[1:]
    else:
        dm = None
    M, _, bv, _ = measure_field_level_bias(
        comm, pm, tracerfield, field_dict, field_D, nmesh, kmax, Lbox, M=M
    )

    if dm is not None:
        field_dict["1m"] = dm
        temp = np.copy(field_D)
        field_D = np.zeros(len(field_D) + 1)
        field_D[0] = d
        field_D[1:] = temp

    bv = np.array(bv)

    return bv, M, pk_tt_dict


def load_halos(bgcfile, outfile):

    with open(bgcfile, "r") as fp:
        firstline = fp.readline()

    bgcnames = firstline.strip().replace("#", "").split(" ")

    with open(outfile, "r") as fp:
        firstline = fp.readline()
    outnames = firstline.strip().replace("#", "").split(" ")

    bgc_halos = CSVFile(
        bgcfile,
        bgcnames,
        usecols=["ID", "M200", "R200", "Np", "X", "Y", "Z", "Parent_ID"],
        skiprows=1,
    )
    out_halos = CSVFile(outfile, outnames, usecols=["ID", "M200b", "Rs"], skiprows=17)

    sorter = out_halos["ID"][:].argsort()
    idx = out_halos["ID"][:].searchsorted(bgc_halos["ID"][:], sorter=sorter)
    rs = out_halos["Rs"][:][sorter][idx]
    idx = (bgc_halos["Parent_ID"][:] == -1) & (bgc_halos["Np"][:] >= 200)
    c = bgc_halos["R200"][:][idx] / rs[idx]
    m = bgc_halos["M200"][:][idx]

    nhalos = np.sum(idx)
    pos = np.zeros((nhalos, 3))
    pos[:, 0] = bgc_halos["X"][:][idx]
    pos[:, 1] = bgc_halos["Y"][:][idx]
    pos[:, 2] = bgc_halos["Z"][:][idx]

    return m, c, pos


def measure_hmf(m, c, nbins=20):

    logmmin = np.log10(np.min(m))
    logmmax = np.log10(np.max(m))
    dm = (logmmax - logmmin) / nbins

    mbins = np.logspace(logmmin, logmmax, nbins)
    mbin = np.digitize(m, mbins)
    nm = np.bincount(mbin)
    msum = np.bincount(mbin, weights=m)
    csum = np.bincount(mbin, weights=c)
    cssum = np.bincount(mbin, weights=c**2)

    mmean = msum / nm
    cmean = csum / nm
    cstd = np.sqrt((cssum - csum**2 / nm) / nm)

    return mbins, mmean, nm, cmean, cstd, mbin


def measure_hmf_and_bias(
    config, bgcfile, outfile, field_dict, field_D, pm, nbins=20, comm=None
):

    Lbox = config["lbox"]
    nmesh = config["nmesh_out"]
    kmax = config["field_level_kmax"]
    interlaced = config["interlaced"]
    nkmax = len(kmax)
    resampler_type = "cic"
    resampler = _get_resampler(resampler_type)

    m, c, pos = load_halos(bgcfile, outfile)
    mbins, mmean, nm, cmean, cstd, mbin = measure_hmf(m, c, nbins=nbins)

    M = None
    k_edges = np.linspace(0, nmesh * np.pi / Lbox, int(nmesh // 2))
    nkbins = len(k_edges) - 1
    bias_vec = np.zeros((nbins, 3, nkmax, 4))
    bias_cov = np.zeros((nbins, 3, nkmax, 4, 4))
    pk_vec = np.zeros((nbins, 3, nkbins))

    for i in range(nbins):
        idx = mbin == i

        bv, M, pk_tt = field_level_bias(
            pos[idx],
            field_dict,
            field_D,
            nmesh,
            kmax,
            Lbox,
            pm,
            resampler,
            M=M,
            interlaced=interlaced,
            comm=comm,
        )
        k = pk_tt["k"]        
        pk_tt = pk_tt["power_wedges"]

        if i == 0:
            Minv = [np.linalg.inv(M[..., j]) for j in range(nkmax)]
        bvcov = np.array([Minv[j] / nm[i] for j in range(nkmax)])
        bias_vec[i, 0, ...] = bv
        bias_cov[i, 0, ...] = bvcov
        pk_vec[i, 0, :] = pk_tt.reshape(nkbins)

        idx = (mbin == i) & (c <= cmean[i])

        bv, M, pk_tt = field_level_bias(
            pos[idx], field_dict, field_D, nmesh, kmax, Lbox, pm, resampler, M=M, interlaced=interlaced, comm=comm
        )
        pk_tt = pk_tt["power_wedges"]

        bvcov = np.array([Minv[j] / np.sum(idx) for j in range(nkmax)])
        bias_vec[i, 1, ...] = bv
        bias_cov[i, 1, ...] = bvcov
        pk_vec[i, 1, :] = pk_tt.reshape(nkbins)

        idx = (mbin == i) & (c > cmean[i])

        bv, M, pk_tt = field_level_bias(
            pos[idx],
            field_dict,
            field_D,
            nmesh,
            kmax,
            Lbox,
            pm,
            resampler,
            M=M,
            interlaced=interlaced,
            comm=comm,
        )
        pk_tt = pk_tt["power_wedges"]
        
        bvcov = np.array([Minv[j] / np.sum(idx) for j in range(nkmax)])
        bias_vec[i, 2, ...] = bv
        bias_cov[i, 2, ...] = bvcov
        pk_vec[i, 2, :] = pk_tt.reshape(nkbins)

    halofilenum = int(bgcfile.split('.list')[0].split('_')[-1])
    outdir = "/".join(bgcfile.split('/')[:-1])

    if comm.rank == 0:
    
        with h5.File("{}/hmf_bias_data.h5".format(outdir), "a") as halodata:
            kmax = np.array(kmax)

            halodata.create_dataset(
                "bias_data/bias_params_{}".format(halofilenum), (nbins, 3, nkmax, 4), dtype=bias_vec.dtype
            )
            halodata.create_dataset(
                "bias_data/bias_cov_{}".format(halofilenum), (nbins, 3, nkmax, 4, 4), dtype=bias_vec.dtype
            )
            halodata.create_dataset(
                "bias_data/pk_auto_data_{}".format(halofilenum), (nbins, 3, nkbins), dtype=bias_vec.dtype
            )
            halodata.create_dataset("bias_data/k_{}".format(halofilenum), (nkbins), dtype=k.dtype)
            halodata.create_dataset("bias_data/Mij_{}".format(halofilenum), M.shape, dtype=M.dtype)
            halodata.create_dataset("bias_data/kmax_{}".format(halofilenum), (nkmax), dtype=kmax.dtype)

            halodata["bias_data/bias_params_{}".format(halofilenum)][:] = bias_vec
            halodata["bias_data/bias_cov_{}".format(halofilenum)][:] = bias_cov
            halodata["bias_data/pk_auto_data_{}".format(halofilenum)][:] = pk_vec
            halodata["bias_data/Mij_{}".format(halofilenum)][:] = M
            halodata["bias_data/k_{}".format(halofilenum)][:] = k
            halodata["bias_data/kmax_{}".format(halofilenum)][:] = kmax

            halodata.create_dataset("count_data/mbin_edges_{}".format(halofilenum), mbins.shape, dtype=mbins.dtype)
            halodata.create_dataset("count_data/mmean_{}".format(halofilenum), mmean.shape, dtype=mmean.dtype)            
            halodata.create_dataset("count_data/n_per_bin_{}".format(halofilenum), nm.shape, dtype=nm.dtype)
            halodata.create_dataset(
                "count_data/cmean_per_bin_{}".format(halofilenum), cmean.shape, dtype=cmean.dtype
            )
            halodata.create_dataset("count_data/cstd_per_bin_{}".format(halofilenum), cstd.shape, dtype=cstd.dtype)
            
            halodata["count_data/mbin_edges_{}".format(halofilenum)][:] = mbins
            halodata["count_data/mmean_{}".format(halofilenum)][:] = mmean
            halodata["count_data/n_per_bin_{}".format(halofilenum)][:] = nm
            halodata["count_data/cmean_per_bin_{}".format(halofilenum)][:] = cmean
            halodata["count_data/cstd_per_bin_{}".format(halofilenum)][:] = cstd
