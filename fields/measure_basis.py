from .common_functions import readGadgetSnapshot
from nbodykit.algorithms.fftpower import FFTPower
from classy import Class
from mpi4py import MPI
from glob import glob
import numpy as np
import time, sys, gc, psutil, os, yaml
import pmesh, h5py


def get_memory(rank):
    process = psutil.Process(os.getpid())
    print(
        process.memory_info().rss / 1e9, "GB is current memory usage, rank ", rank
    )  # in bytes


def mpiprint(text, rank):
    if rank == 0:
        print(text)
        sys.stdout.flush()
    else:
        pass


def CompensateCICAliasing(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the CIC window function in configuration space,
            as well as the approximate aliasing correction
    From the nbodykit documentation.
    """
    for i in range(3):
        wi = w[i]
        v = v / (1 - 2.0 / 3 * np.sin(0.5 * wi) ** 2) ** 0.5
    return v


def get_Nparts(snapfiles, sim_type, parttype):
    """Count up the number of particles that will be read in by this rank.

    Args:
        snapfiles list: List of blocks assigned to this rank
        sim_type str: Type of simulation (format)

    Returns:
        npart int: Number of particles assigned to this rank.
    """
    npart = 0
    for f in snapfiles:
        if sim_type == "Gadget_hdf5":
            block = h5py.File(f, "r")
            npart += block["Header"].attrs["NumPart_ThisFile"][parttype]
            z_this = block["Header"].attrs["Redshift"]
            mass = block["Header"].attrs["MassTable"][parttype]
            block.close()
        elif sim_type == "Gadget":
            if parttype != 1:
                raise (
                    ValueError(
                        "Neutrino functionality not yet implemented for classic gadget outputs."
                    )
                )
            header = readGadgetSnapshot(f, read_id=False, read_pos=False)
            npart += header["npart"][1]
            z_this = header["redshift"]
            mass = 1

    return npart, z_this, mass


def load_particles(
    basedir,
    sim_type,
    rank,
    size,
    parttype=1,
    cv_surrogate=False,
    icfile=None,
    ic_format=None,
    D=None,
    nmesh=None,
    lbox=None,
    boltz=None,
    z_ic=None,
):

    if sim_type == "Gadget_hdf5":
        snapfiles = glob(basedir + "*hdf5")
    elif sim_type == "Gadget":
        snapfiles = glob(basedir + "*")

    if cv_surrogate:
        assert icfile is not None
        assert ic_format is not None
        assert D is not None
        assert nmesh is not None
        assert lbox is not None

    snapfiles_this = snapfiles[rank::size]
    nfiles_this = len(snapfiles_this)
    npart_this, z_this, mass = get_Nparts(snapfiles_this, sim_type, parttype)

    D = boltz.scale_independent_growth_factor(z_this)
    D = D / boltz.scale_independent_growth_factor(z_ic)

    pos = np.zeros((npart_this, 3))
    if parttype == 1:
        ids = np.zeros(npart_this, dtype=np.int)
    else:
        # don't need ids for neutrinos, since not weighting
        ids = None

    if not cv_surrogate:
        npart_counter = 0
        for i in range(nfiles_this):

            if sim_type == "Gadget_hdf5":
                block = h5py.File(snapfiles_this[i], "r")
                npart_block = block["Header"].attrs["NumPart_ThisFile"][parttype]
                pos[npart_counter : npart_counter + npart_block] = block[
                    "PartType{}/Coordinates".format(parttype)
                ]
                if parttype == 1:
                    ids[npart_counter : npart_counter + npart_block] = block[
                        "PartType{}/ParticleIDs".format(parttype)
                    ]
                block.close()

            elif sim_type == "Gadget":
                if parttype != 1:
                    raise (
                        ValueError(
                            "Neutrino functionality not yet implemented for classic gadget outputs."
                        )
                    )
                hdr, pos_i, ids_i = readGadgetSnapshot(
                    snapfiles_this[i], read_id=True, read_pos=True
                )
                npart_block = hdr["npart"][1]
                pos[npart_counter : npart_counter + npart_block] = pos_i
                ids[npart_counter : npart_counter + npart_block] = ids_i
            else:
                raise (ValueError("Sim type must be either Gadget or GadgetHDF5"))

            npart_counter += npart_block
    else:
        if ic_format == "monofonic":
            n_ = [nmesh, nmesh, nmesh]
            get_cell_idx = lambda i, j, k: (i * n_[1] + j) * n_[2] + k
            with open(icfile, "r") as ics:
                # read in displacements, rescale by D=D(z_this)/D(z_ini)
                grid = np.meshgrid(
                    np.arange(rank, nmesh, size),
                    np.arange(nmesh),
                    np.arange(nmesh),
                    indexing="ij",
                )
                pos_x = (
                    (grid[0] / nmesh + D * ics["DM_dx_filt"][rank::size, ...]) % 1
                ) * lbox
                pos_y = (
                    (grid[1] / nmesh + D * ics["DM_dy_filt"][rank::size, ...]) % 1
                ) * lbox
                pos_z = (
                    (grid[2] / nmesh + D * ics["DM_dz_filt"][rank::size, ...]) % 1
                ) * lbox
                pos = np.stack([pos_x, pos_y, pos_z])
                pos = pos.reshape(3, -1).T
                del pos_x, pos_y, pos_z
                gc.collect()

                ids = get_cell_idx(grid[0], grid[1], grid[2]).flatten()
                del grid[0], grid[1], grid[2]
                gc.collect()

                mass = 1

        else:
            raise (ValueError("ic_format {} is unsupported".format(ic_format)))

    return pos, ids, npart_this, z_this, mass, D


def measure_basis_spectra(configs):

    lindir = configs["outdir"]
    nmesh = configs["nmesh_in"]
    Lbox = configs["lbox"]
    compensate = bool(configs["compensate"])
    fdir = configs["particledir"]
    componentdir = configs["outdir"]
    cv_surrogate = configs["compute_cv_surrogate"]
    # don't use neutrinos for CV surrogate. cb field should be fine.
    if cv_surrogate:
        use_neutrinos = False
        basename = "mpi_icfields_nmesh_filt"
    else:
        use_neutrinos = configs["use_neutrinos"]
        basename = "mpi_icfields_nmesh"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    start_time = time.time()

    # ParticleMeshes for 1, delta, deltasq, tidesq, nablasq
    # Make the late-time component fields

    ################################################################################################
    #################################### Advecting weights #########################################
    pm = pmesh.pm.ParticleMesh(
        [nmesh, nmesh, nmesh], Lbox, dtype="float32", resampler="cic", comm=comm
    )

    if rank == 0:
        get_memory(rank)
        print("starting loop")
        sys.stdout.flush()

    pkclass = Class()
    pkclass.set(configs["Cosmology"])
    pkclass.compute()

    z_ic = configs["z_ic"]

    # Load in a subset of the total gadget snapshot.
    posvec, idvec, npart_this, zbox, m_cb, D = load_particles(
        fdir,
        configs["sim_type"],
        rank,
        nranks,
        cv_surrogate=cv_surrogate,
        icfile=configs["icdir"],
        ic_format=configs["ic_format"],
        boltz=pkclass,
        nmesh=configs["nmesh_in"],
        lbox=configs["lbox"],
        z_ic=z_ic
    )

    # if use_neutrinos=True, compute an additional set of basis spectra,
    # where the unweighted field is the total matter field
    # rather than the cb field. Separate this out to save memory.
    if use_neutrinos:
        posvec_nu, _, _, _, m_nu, _ = load_particles(
            fdir, configs["sim_type"], rank, nranks, parttype=2,
            boltz=pkclass, z_ic=z_ic
        )
        posvec_tot = np.vstack([posvec, posvec_nu])
        del posvec_nu
        gc.collect()
        m = np.zeros(len(posvec_tot))
        m[:npart_this] = m_cb
        m[npart_this:] = m_nu

        keynames = ["1m"]
        fieldlist = [pm.create(type="real")]
        layout = pm.decompose(posvec_tot)
        p = layout.exchange(posvec_tot)
        del posvec_tot
        gc.collect()
        w = layout.exchange(m)
        del m
        gc.collect()

        pm.paint(p, out=fieldlist[-1], mass=w, resampler="cic")
        del m
    else:
        keynames = []
        fieldlist = []

    keynames.extend(["1cb", "delta", "deltasq", "tidesq", "nablasq"])
    fieldlist.extend(
        [
            pm.create(type="real"),
            pm.create(type="real"),
            pm.create(type="real"),
            pm.create(type="real"),
            pm.create(type="real"),
        ]
    )

    # Gadget has IDs starting with ID=1.
    # FastPM has ID=0
    # idfac decides which one to use
    idfac = 1
    if (configs["sim_type"] == "FastPM") | (configs["ic_format"] == "monofonic"):
        idfac = 0

    a_ic = ((idvec - idfac) // nmesh**2) % nmesh
    b_ic = ((idvec - idfac) // nmesh) % nmesh
    c_ic = (idvec - idfac) % nmesh

    a_ic = a_ic.astype(int)
    b_ic = b_ic.astype(int)
    c_ic = c_ic.astype(int)
    # Figure out where each particle position is going to be distributed among mpi ranks
    layout = pm.decompose(posvec)

    # Exchange positions
    p = layout.exchange(posvec)

    mpiprint(("posvec shapes", posvec.shape), rank)
    mpiprint(("idvec shapes", idvec.shape), rank)
    del posvec
    gc.collect()

    for k in range(len(fieldlist)):
        if keynames[k] == "1m":
            continue  # already handled this above

        if rank == 0:
            print(k)
        if keynames[k] == "1cb":
            pm.paint(p, out=fieldlist[k], mass=1, resampler="cic")
        else:
            # Now load specific compfield. 1,2,3 is delta, delta^2, s^2
            if configs["np_weightfields"]:
                arr = np.load(
                    lindir + "{}_{}_{}_np.npy".format(basename, nmesh, keynames[k]),
                    mmap_mode="r",
                )
            else:
                arr = h5py.File(lindir + "{}_{}".format(basename, nmesh), "r")[k]["3D"][
                    "2"
                ]

            # Get weights
            w = arr[a_ic, b_ic, c_ic]

            mpiprint(("w shapes", w.shape))

            # distribute weights properly
            m = layout.exchange(w)
            del w
            gc.collect()

            get_memory(rank)

            pm.paint(p, out=fieldlist[k], mass=m, resampler="cic")
            sys.stdout.flush()
            del m
            gc.collect()

        # print('painted! ', rank)
        sys.stdout.flush()

    if rank == 0:
        print(fieldlist[0].shape)
    del p
    gc.collect()
    if rank == 0:
        print("pasted")
        sys.stdout.flush()
    get_memory(rank)

    # Normalize and mean-subtract the normal particle field.
    fieldlist[0] = fieldlist[0] / fieldlist[0].cmean() - 1
    if "1m" in keynames:
        fieldlist[1] = fieldlist[1] / fieldlist[1].cmean() - 1

    for k in range(len(fieldlist)):
        if rank == 0:
            print(np.mean(fieldlist[k].value), np.std(fieldlist[k].value))
            sys.stdout.flush()
        if configs["save_advected_fields"]:
            np.save(
                componentdir
                + "latetime_weight_%s_%s_rank%s" % (k, nmesh, rank),
                fieldlist[k].value,
            )

        if compensate:
            fieldlist[k] = fieldlist[k].r2c()
            fieldlist[k] = fieldlist[k].apply(CompensateCICAliasing, kind="circular")

    get_memory(rank)
    sys.stdout.flush()

    #######################################################################################################################
    #################################### Adjusting for growth #############################################################

    mpiprint(D, rank)

    if use_neutrinos:
        labelvec = [
            "1m",
            "1cb",
            r"$\delta_L$",
            r"$\delta^2$",
            r"$s^2$",
            r"$\nabla^2\delta$",
        ]
        field_D = [1, 1, D, D**2, D**2, D]
    else:
        labelvec = ["1cb", r"$\delta_L$", r"$\delta^2$", r"$s^2$", r"$\nabla^2\delta$"]
        field_D = [1, D, D**2, D**2, D]

    field_dict = dict(zip(labelvec, fieldlist))

    #######################################################################################################################
    #################################### Measuring P(k) ###################################################################
    kpkvec = []
    pkcounter = 0
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if (i < j) | (use_neutrinos & (j == 0) & (i == 1)):
                continue
            elif i >= j:
                pk = FFTPower(
                    field_dict[labelvec[i]],
                    "1d",
                    second=field_dict[labelvec[j]],
                    BoxSize=Lbox,
                    Nmesh=nmesh,
                )

                pkpower = pk.power["power"].real
                if i == 0 and j == 0:
                    kpkvec.append(pk.power["k"])

                kpkvec.append(pkpower * field_D[i] * field_D[j])

                pkcounter += 1
                mpiprint(("pk done ", pkcounter), rank)

    kpkvec = np.array(kpkvec)
    if rank == 0:
        np.savetxt(
            componentdir + "lakelag_mpi_pk_a%.2f_nmesh%s.txt" % (1 / (zbox + 1), nmesh),
            kpkvec,
        )

        print(time.time() - start_time)


if __name__ == "__main__":
    ################################### YAML /Initial Config stuff #################################
    yamldir = sys.argv[1]
    fieldnameadd = sys.argv[2]

    configs = yaml.load(open(yamldir, "r"), yaml.FullLoader)
    measure_basis_spectra(configs)
