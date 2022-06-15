import numpy as np
import sys, os

# sys.path.insert(0, '/pscratch/sd/j/jderose/anzu/')
# os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/opt/test'
from fields.make_lagfields import make_lagfields
from fields.measure_basis import measure_basis_spectra
from .common_functions import readGadgetSnapshot
from mpi4py import MPI
from copy import copy
from glob import glob
from yaml import Loader
import sys, yaml
import h5py


def get_snap_z(basedir, sim_type):
    """Count up the number of particles that will be read in by this rank.

    Args:
        snapfiles list: List of blocks assigned to this rank
        sim_type str: Type of simulation (format)

    Returns:
        npart int: Number of particles assigned to this rank.
    """

    if sim_type == "Gadget_hdf5":
        snapfiles = glob(basedir + "*hdf5")
        f = snapfiles[0]
        with h5py.File(f, "r") as block:
            z_this = block["Header"].attrs["Redshift"]

    elif sim_type == "Gadget":
        snapfiles = glob(basedir + "*")
        f = snapfiles[0]
        header = readGadgetSnapshot(f, read_id=False, read_pos=False)
        z_this = header["redshift"]

    return z_this


if __name__ == "__main__":

    config = sys.argv[1]

    with open(config, "r") as fp:
        config = yaml.load(fp, Loader=Loader)

    globsnaps = config["globsnaps"]
    snaplist = config.pop("snaplist", None)
    skip_lagfields = config.pop("skip_lagfields", False)  # if they've already been run

    lattice_type = config.pop("lattice_type", 0)  # 0 for sc, 1 for bcc, 2 for fcc
    config["lattice_type"] = lattice_type

    start_snapnum = int(config.pop("start_snapnum", 0))

    scale_dependent_growth = bool(config.pop("scale_dependent_growth", False))
    config["scale_dependent_growth"] = scale_dependent_growth

    if globsnaps:

        snapdir = "/".join(config["particledir"].split("/")[:-1])
        snapglob = "_".join(snapdir.split("_")[:-1]) + "*"
        snapdirs = np.array(glob(snapglob))
        snapstrs = np.array([d.split("_")[-1] for d in snapdirs])
        snapnums = np.array([int(d) for d in snapstrs])
        idx = np.argsort(snapnums)
        snapdirs = snapdirs[idx]
        snapstrs = snapstrs[idx]
        nsnaps = len(snapstrs)
        pdirs = [
            snapdirs[i] + "/snapshot_{}".format(snapstrs[i]) for i in range(nsnaps)
        ]

    elif snaplist:

        this_snap = config["particledir"].split("snapdir_")[-1].split("/")[0]
        pdirs = [config["particledir"].replace(this_snap, s) for s in snaplist]
        nsnaps = len(pdirs)

    else:
        nsnaps = 1
        pdirs = [config["particledir"]]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    do_rsd = config.pop("rsd", False)
    do_rs = config.pop("do_real_space", True)

    if config["compute_cv_surrogate"]:
        config_surr = copy(config)
        config["compute_cv_surrogate"] = False

        # scale independent growth good enough for surrogates(?)
        # Just make sure you use the right linear power in analytic prediction
        config_surr["scale_dependent_growth"] = False
        do_surrogates = True
    else:
        do_surrogates = False

    if rank == 0:
        print("Constructing z_ic lagrangian fields", flush=True)

    if not skip_lagfields:
        make_lagfields(config)

        if do_surrogates:
            make_lagfields(config_surr)

    if rank == 0:
        print("Processing basis spectra for {} snapshots".format(nsnaps), flush=True)

    if do_rs:

        for i in range(nsnaps):
            if i < start_snapnum:
                continue
            if rank == 0:
                print("Processing snapshot {}".format(i), flush=True)

            config["particledir"] = pdirs[i]

            if scale_dependent_growth:
                z = get_snap_z(pdirs[i], config["sim_type"])
                lag_field_dict = make_lagfields(config, save_to_disk=False, z=z)
            else:
                lag_field_dict = None

            field_dict, D = measure_basis_spectra(config, lag_field_dict)

            if do_surrogates:
                config_surr["particledir"] = pdirs[i]
                measure_basis_spectra(config_surr, field_dict2=field_dict, field_D2=D)

    if do_rsd:

        config["rsd"] = True
        config_surr["rsd"] = True

        for i in range(nsnaps):
            if (not do_rs) & (i < start_snapnum):
                continue
            if rank == 0:
                print("Processing snapshot {}".format(i), flush=True)

            config["particledir"] = pdirs[i]
            field_dict, D = measure_basis_spectra(config)

            if do_surrogates:
                config_surr["particledir"] = pdirs[i]
                measure_basis_spectra(config_surr, field_dict2=field_dict, field_D2=D)
