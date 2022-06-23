from .common_functions import readGadgetSnapshot
from classy import Class
from mpi4py import MPI
from glob import glob
import numpy as np
import time, sys, gc, psutil, os, yaml
import pmesh, h5py

try:
    from nbodykit.algorithms.fftpower import FFTPower
except Exception as e:
    print(e)

from pypower import MeshFFTPower


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

    if npart == 0:
        return npart, 0, 0
    else:
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
    rsd=False,
):

    if sim_type == "Gadget_hdf5":
        snapfiles = glob(basedir + "*hdf5")
    elif sim_type == "Gadget":
        snapfiles = glob(basedir + "*")

    if cv_surrogate:
        assert icfile is not None
        assert ic_format is not None
        assert nmesh is not None
        assert lbox is not None

    snapfiles_this = snapfiles[rank::size]
    nfiles_this = len(snapfiles_this)
    npart_this, z_this, mass = get_Nparts(snapfiles_this, sim_type, parttype)

    D = boltz.scale_independent_growth_factor(z_this)
    D = D / boltz.scale_independent_growth_factor(z_ic)
    f = boltz.scale_independent_growth_factor_f(z_this)
    Ha = boltz.Hubble(z_this) * 299792.458

    if rsd:
        v_fac = (1 + z_this) ** 0.5 / Ha * boltz.h()  # (v_p = v_gad * a^(1/2))

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
                if rsd:
                    pos[npart_counter : npart_counter + npart_block, 2] += (
                        v_fac * block["PartType{}/Velocities".format(parttype)][:, 2]
                    )
                    pos[npart_counter : npart_counter + npart_block, 2] %= lbox

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
                if rsd:
                    hdr, pos_i, vel_i, ids_i = readGadgetSnapshot(
                        snapfiles_this[i], read_id=True, read_pos=True, read_vel=True
                    )
                    npart_block = hdr["npart"][1]
                    pos[npart_counter : npart_counter + npart_block] = pos_i
                    pos[npart_counter : npart_counter + npart_block, 2] += (
                        v_fac * vel_i[:, 2]
                    )
                    pos[npart_counter : npart_counter + npart_block, 2] %= lbox
                    ids[npart_counter : npart_counter + npart_block] = ids_i
                else:
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
            with h5py.File(icfile, "r") as ics:
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
                if rsd:
                    pos_z += f * D * ics["DM_dz_filt"][rank::size, ...] * lbox
                    pos_z %= lbox
                pos = np.stack([pos_x, pos_y, pos_z])
                pos = pos.reshape(3, -1).T
                del pos_x, pos_y, pos_z
                gc.collect()

                ids = get_cell_idx(grid[0], grid[1], grid[2])
                del grid
                gc.collect()

                mass = 1

        else:
            raise (ValueError("ic_format {} is unsupported".format(ic_format)))

    return pos, ids, npart_this, z_this, mass, D


def measure_pk(mesh1, mesh2, lbox, nmesh, rsd, use_pypower, D1, D2):

    k_edges = np.linspace(0, nmesh * np.pi / lbox, int(nmesh // 2))
    if not rsd:
        edges = k_edges
        poles = (0,)
        mode = "1d"
        Nmu = 1
    else:
        Nmu = 100
        mu_edges = np.linspace(0, 1, Nmu)
        edges = (k_edges, mu_edges)
        poles = (0, 2, 4)
        mode = "2d"

    pkdict = {}

    if use_pypower:
        pk = MeshFFTPower(mesh1, mesh2=mesh2, edges=edges, los="z", ells=poles)

        pkdict["k"] = pk.poles.k
        pkdict["mu"] = pk.wedges.mu
        pkdict["nmodes"] = pk.poles.nmodes
        pkdict["nmodes_wedges"] = pk.wedges.nmodes
        pkdict["power_poles"] = pk.poles.power.real * D1 * D2
        pkdict["power_wedges"] = pk.wedges.power.real * D1 * D2

    else:
        pk = FFTPower(
            mesh1, mode, second=mesh2, BoxSize=lbox, Nmesh=nmesh, Nmu=Nmu, poles=poles
        )

        pkdict["k"] = pk.power["k"].real
        pkdict["nmodes"] = pk.power["modes"].real
        pkdict["power_wedges"] = pk.power["power"].real * D1 * D2

        if rsd:
            pkdict["mu"] = pk.power["mu"].real
            pkdict["power_poles"] = np.stack(
                [pk.poles["power_{}".format(ell)].real * D1 * D2 for ell in poles]
            )

    return pkdict


def exchange(send_counts, send_offsets, recv_counts, recv_offsets, data, comm):

    newlength = recv_counts.sum()

    duplicity = np.product(np.array(data.shape[1:], "intp"))
    itemsize = duplicity * data.dtype.itemsize

    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()
    dtype = np.dtype((data.dtype, data.shape[1:]))
    recvbuffer = np.empty(newlength, dtype=dtype, order="C")

    _ = comm.Alltoallv(
        (data, (send_counts, send_offsets), dt),
        (recvbuffer, (recv_counts, recv_offsets), dt),
    )
    dt.Free()
    comm.Barrier()

    return recvbuffer


def send_parts_to_weights(idvec, posvec, nmesh, comm, idfac, overload):

    nranks = comm.size
    # determine what rank each particles weight is on
    a_ic = (((idvec - idfac) // overload) // nmesh ** 2) % nmesh
    a_ic = a_ic.astype(int)

    slabs_per_rank = nmesh // nranks
    send_rank = a_ic // slabs_per_rank

    # presort before communication
    idx = np.argsort(send_rank)
    posvec = posvec.take(idx, axis=0)
    idvec = idvec.take(idx, axis=0)
    send_rank = send_rank[idx]

    # figure out how many things we're sending where
#    send_counts, _ = np.histogram(send_rank + 1, np.arange(nranks + 1))
    send_counts = np.bincount(send_rank, minlength=nranks)
    recv_counts = np.zeros_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    send_offsets = np.zeros_like(send_counts)
    recv_offsets = np.zeros_like(recv_counts)
    send_offsets[1:] = send_counts.cumsum()[:-1]
    recv_offsets[1:] = recv_counts.cumsum()[:-1]

    # send
    posvec = exchange(
        send_counts, send_offsets, recv_counts, recv_offsets, posvec, comm
    )

    idvec = exchange(send_counts, send_offsets, recv_counts, recv_offsets, idvec, comm)

    return posvec, idvec


def measure_basis_spectra(
    configs, lag_field_dict=None, field_dict2=None, field_D2=None
):

    lindir = configs["outdir"]
    nmesh = configs["nmesh_in"]
    Lbox = configs["lbox"]
    compensate = bool(configs["compensate"])
    fdir = configs["particledir"]
    componentdir = configs["outdir"]
    cv_surrogate = configs["compute_cv_surrogate"]
    try:
        use_pypower = configs["use_pypower"]
    except:
        use_pypower = False

    try:
        rsd = configs["rsd"]
    except:
        rsd = False

    # don't use neutrinos for CV surrogate. cb field should be fine.
    if cv_surrogate:
        use_neutrinos = False
        basename = "mpi_icfiles_nmesh_filt"
        outname = "basis_spectra_za_surrogate"
    else:
        use_neutrinos = configs["use_neutrinos"]
        basename = "mpi_icfields_nmesh"
        outname = "basis_spectra_nbody"

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
        z_ic=z_ic,
        rsd=rsd,
    )

    # if use_neutrinos=True, compute an additional set of basis spectra,
    # where the unweighted field is the total matter field
    # rather than the cb field. Separate this out to save memory.
    if use_neutrinos:
        posvec_nu, _, _, _, m_nu, _ = load_particles(
            fdir,
            configs["sim_type"],
            rank,
            nranks,
            parttype=2,
            boltz=pkclass,
            z_ic=z_ic,
            lbox=configs["lbox"],
            rsd=rsd,
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
    overload = 1 << configs["lattice_type"]

    if lag_field_dict:
        posvec, idvec = send_parts_to_weights(idvec, posvec, nmesh, comm, idfac, overload)
        a_ic = (
            (((idvec - idfac) // overload) // nmesh ** 2)
            % nmesh
            % (nmesh // nranks)
        )
    else:
        a_ic = (((idvec - idfac) // overload) // nmesh ** 2) % nmesh

    b_ic = (((idvec - idfac) // overload) // nmesh) % nmesh
    c_ic = ((idvec - idfac) // overload) % nmesh

    del idvec
    gc.collect()

    # Figure out where each particle position is going to be distributed among mpi ranks
    layout = pm.decompose(posvec)
    p = layout.exchange(posvec)

    if lag_field_dict:
        kdict = lag_field_dict.keys()        
        lag_field_dict_new = {}
        for k in kdict:
            lag_field_dict_new[k] = layout.exchange(lag_field_dict[k][a_ic, b_ic, c_ic])
        del lag_field_dict
        lag_field_dict = lag_field_dict_new
        
    del posvec
    gc.collect()
    

    for k in range(len(fieldlist)):
        if keynames[k] == "1m":
            continue  # already handled this above

        if keynames[k] == "1cb":
            pm.paint(p, out=fieldlist[k], mass=1, resampler="cic")
        else:

            if lag_field_dict:
                m = lag_field_dict[keynames[k]]

            elif configs["np_weightfields"]:
                arr = np.load(
                    lindir + "{}_{}_{}_np.npy".format(basename, nmesh, keynames[k]),
                    mmap_mode="r",
                )

                if cv_surrogate:
                    w = arr[rank::nranks, ...].flatten()
                else:
                    w = arr[a_ic, b_ic, c_ic]

                m = layout.exchange(w)
                del w
            else:
                arr = h5py.File(lindir + "{}_{}.h5".format(basename, nmesh), "r")[
                    keynames[k]
                ]["3D"]["2"]
                w = arr[rank::nranks, ...].flatten()

                # distribute weights properly
                m = layout.exchange(w)
                del w
            gc.collect()

            pm.paint(p, out=fieldlist[k], mass=m, resampler="cic")
            sys.stdout.flush()
            del m
            gc.collect()

        sys.stdout.flush()

    del p
    gc.collect()

    # Normalize and mean-subtract the normal particle field.
    fieldlist[0] = fieldlist[0] / fieldlist[0].cmean() - 1
    if "1m" in keynames:
        fieldlist[1] = fieldlist[1] / fieldlist[1].cmean() - 1

    for k in range(len(fieldlist)):
        if rank == 0:
            sys.stdout.flush()
        if configs["save_advected_fields"]:
            if cv_surrogate:
                np.save(
                    componentdir + "latetime_zeldovich_weight_{}_z{}_{}_rank{}".format(k, zbox, nmesh, rank),
                    fieldlist[k].value,
                )                
            else:
                np.save(
                    componentdir + "latetime_nbody_weight_{}_z{}_{}_rank{}".format(k, zbox, nmesh, rank),
                    fieldlist[k].value,
                )

        if compensate:
            fieldlist[k] = fieldlist[k].r2c()
            fieldlist[k] = fieldlist[k].apply(CompensateCICAliasing, kind="circular")

    sys.stdout.flush()
    #######################################################################################################################
    #################################### Adjusting for growth #############################################################

    if use_neutrinos:
        labelvec = [
            "1m",
            "1cb",
            r"$\delta_L$",
            r"$\delta^2$",
            r"$s^2$",
            r"$\nabla^2\delta$",
        ]
        field_D = [1, 1, D, D ** 2, D ** 2, D]

    else:
        labelvec = ["1cb", r"$\delta_L$", r"$\delta^2$", r"$s^2$", r"$\nabla^2\delta$"]
        field_D = [1, D, D ** 2, D ** 2, D]

    if configs["scale_dependent_growth"]:
        field_D = [1, 1, 1, 1, 1, 1]
        
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

                pkdict = measure_pk(
                    field_dict[labelvec[i]],
                    field_dict[labelvec[j]],
                    Lbox,
                    nmesh,
                    rsd,
                    use_pypower,
                    field_D[i],
                    field_D[j],
                )
                kpkvec.append(pkdict)
                pkcounter += 1
                mpiprint(("pk done ", pkcounter), rank)

    if rank == 0:
        np.save(
            componentdir
            + "{}_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                outname, rsd, use_pypower, 1 / (zbox + 1), nmesh
            ),
            kpkvec,
        )

        print("measuring spectra took {}s".format(time.time() - start_time))

    # if we passed another field dict in to this function, cross correlate everything with everything
    if field_dict2:
        kpkvec = []
        pkcounter = 0
        if field_dict2 is not None:
            labelvec2 = list(field_dict2.keys())

            for i in range(len(labelvec)):
                for j in range(len(labelvec2)):

                    pkdict = measure_pk(
                        field_dict[labelvec[i]],
                        field_dict2[labelvec2[j]],
                        Lbox,
                        nmesh,
                        rsd,
                        use_pypower,
                        field_D[i],
                        field_D2[j],
                    )
                    kpkvec.append(pkdict)
                    pkcounter += 1
                    mpiprint(("pk done ", pkcounter), rank)

        if rank == 0:
            np.save(
                componentdir
                + "{}_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                    outname + "_crosscorr", rsd, use_pypower, 1 / (zbox + 1), nmesh
                ),
                kpkvec,
            )
            print("measuring cross spectra took {}s".format(time.time() - start_time))

    return field_dict, field_D


if __name__ == "__main__":
    ################################### YAML /Initial Config stuff #################################
    yamldir = sys.argv[1]
    fieldnameadd = sys.argv[2]

    configs = yaml.load(open(yamldir, "r"), yaml.FullLoader)
    measure_basis_spectra(configs)
