import numpy as np
import sys, os
from fields.make_lagfields import make_lagfields
from fields.measure_basis import advect_fields_and_measure_spectra
from fields.measure_hmf_and_bias import measure_hmf_and_bias
from fields.common_functions import get_snap_z
from mpi4py import MPI
from copy import copy
from glob import glob
from yaml import Loader
import sys, yaml
import h5py
import gc

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
        
        if 'halodir' in config:
            savelist = np.genfromtxt(config['savelist'])
            halodir = config['halodir']
            bgcdirs = glob("{}/outbgc2_*list".format(halodir))
            outdirs = glob("{}/out_*list".format(halodir))
            bgcdirs = np.array(glob(bgcdirs))
            outdirs = np.array(glob(outdirs))

            bgcstrs = np.array([d.split("_")[-1].split('.list') for d in bgcdirs])
            bgcnums = np.array([int(d) for d in bgcstrs])
            outstrs = np.array([d.split("_")[-1].split('.list') for d in outdirs])
            outnums = np.array([int(d) for d in outstrs])            
            
            idx = np.argsort(bgcnums)
            bgcdirs = bgcdirs[idx]
            bgcstrs = bgcstrs[idx]    
            idx = np.argsort(outnums)                
            outdirs = outdirs[idx]
            outstrs = outstrs[idx]
            
            nhalo_files = len(outdirs)
            halo_counter = 0
            do_hmf_and_bias = True
            

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
        # just use d_lin from IC code here
        config["scale_dependent_growth"] = False
        out = make_lagfields(config, save_to_disk=True)
        del out
        gc.collect()
        config["scale_dependent_growth"] = scale_dependent_growth

        if do_surrogates:
            out = make_lagfields(config_surr, save_to_disk=True)
            del out
            gc.collect()

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

            field_dict, field_D, _, _, _, pm = advect_fields_and_measure_spectra(
                config, lag_field_dict=lag_field_dict
            )
            if do_surrogates:
                config_surr["particledir"] = pdirs[i]
                _ = advect_fields_and_measure_spectra(
                    config_surr, field_dict2=field_dict, field_D2=field_D
                )
                
            if (do_hmf_and_bias) & (i == savelist[halo_counter]):
                measure_hmf_and_bias(config, bgcdirs[halo_counter], outdirs[halo_counter], field_dict, field_D, pm, comm=comm)

    if do_rsd:

        config["rsd"] = True
        config_surr["rsd"] = True

        for i in range(nsnaps):
            if (not do_rsd) | (i < start_snapnum):
                continue
            if rank == 0:
                print("Processing snapshot {}".format(i), flush=True)

            if scale_dependent_growth:
                z = get_snap_z(pdirs[i], config["sim_type"])
                lag_field_dict = make_lagfields(config, save_to_disk=False, z=z)
            else:
                lag_field_dict = None

            config["particledir"] = pdirs[i]
            field_dict, field_D, _, _, _ = advect_fields_and_measure_spectra(
                config, lag_field_dict=lag_field_dict
            )

            if do_surrogates:
                config_surr["particledir"] = pdirs[i]
                _ = advect_fields_and_measure_spectra(
                    config_surr, field_dict2=field_dict, field_D2=field_D
                )
