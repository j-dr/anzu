from nbodykit.algorithms.fftpower import FFTPower
from common_functions import readGadgetSnapshot
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


def get_Nparts(snapfiles, sim_type):
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
            npart += block["Header"].attrs["NumPart_ThisFile"][1]
            z_this = block["Header"].attrs["Redshift"]
            block.close()
        if sim_type == "Gadget":
            header = readGadgetSnapshot(f, read_id=False, read_pos=False)
            npart += header["npart"][1]
            z_this = header["redshift"]

    return npart, z_this


def load_particles(basedir, sim_type, rank, size):

    if sim_type == "Gadget_hdf5":
        snapfiles = glob(basedir + "*hdf5")
    elif sim_type == "Gadget":
        snapfiles = glob(basedir + "*")

    snapfiles_this = snapfiles[rank::size]
    nfiles_this = len(snapfiles_this)
    npart_this, z_this = get_Nparts(snapfiles_this, sim_type)

    pos = np.zeros((npart_this, 3))
    ids = np.zeros(npart_this, dtype=np.int)

    npart_counter = 0

    for i in range(nfiles_this):
        if sim_type == "Gadget_hdf5":
            block = h5py.File(snapfiles_this[i], "r")
            npart_block = block["Header"].attrs["NumPart_ThisFile"][1]
            pos[npart_counter : npart_counter + npart_block] = block[
                "PartType1/Coordinates"
            ]
            ids[npart_counter : npart_counter + npart_block] = block[
                "PartType1/ParticleIDs"
            ]
            block.close()

        elif sim_type == "Gadget":
            hdr, pos_i, ids_i = readGadgetSnapshot(
                snapfiles_this[i], read_id=True, read_pos=True
            )
            npart_block = hdr["npart"][1]
            pos[npart_counter : npart_counter + npart_block] = pos_i
            ids[npart_counter : npart_counter + npart_block] = ids_i
        else:
            raise (ValueError("Sim type must be either Gadget or GadgetHDF5"))

        npart_counter += npart_block

    return pos, ids, npart_this, z_this


def measure_basis_spectra(configs):
    
    lindir = configs["outdir"]
    nmesh = configs["nmesh_in"]
    Lbox = configs["lbox"]
    compensate = bool(configs["compensate"])
    fdir = configs["particledir"]

    # Save to wherever particles are
    componentdir = configs["outdir"]
    boxno = configs["aem_box"]
    try:
        testvar = configs["aem_testno"]
    except:
        testvar = ""

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

    fieldlist = [
        pm.create(type="real"),
        pm.create(type="real"),
        pm.create(type="real"),
        pm.create(type="real"),
        pm.create(type="real"),
    ]

    if rank == 0:
        get_memory(rank)
        print("starting loop")
        sys.stdout.flush()

    # Load in a subset of the total gadget snapshot.
    posvec, idvec, npart_this, zbox = load_particles(
        fdir, configs["sim_type"], rank, nranks
    )

    # Gadget has IDs starting with ID=1.
    # FastPM has ID=0
    # idfac decides which one to use
    idfac = 1
    if configs["sim_type"] == "FastPM":
        idfac = 0

    a_ic = ((idvec - idfac) // nmesh**2) % nmesh
    b_ic = ((idvec - idfac) // nmesh) % nmesh
    c_ic = (idvec - idfac) % nmesh
    mpiprint(a_ic[3], rank)
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

    # f = h5py.File(lindir+'mpi_icfields_nmesh%s.h5'%nmesh, 'r')
    # keynames = list(f.keys())
    keynames = ["1", "delta", "deltasq", "tidesq", "nablasq"]
    for k in range(len(fieldlist)):
        if rank == 0:
            print(k)
        if k == 0:
            pm.paint(p, out=fieldlist[k], mass=1, resampler="cic")
        else:
            # Now only load specific compfield. 1,2,3 is delta, delta^2, s^2
            # compfield = np.load(componentdir+'reshape_componentfields_%s_%s.npy'%(rank,k), mmap_mode='r')
            # Load in the given weight field
            if config["np_weightfields"]:
                arr = np.load(lindir + keynames[k] + "_np.npy", mmap_mode="r")
            else:
                arr = h5py.File(lindir + "mpi_icfields_nmesh%s.h5" % nmesh, "r")[k][
                    "3D"
                ]["2"]

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
    for k in range(len(fieldlist)):
        if rank == 0:
            print(np.mean(fieldlist[k].value), np.std(fieldlist[k].value))
            sys.stdout.flush()
        if config["save_advected_fields"]:
            np.save(
                componentdir
                + "latetime_weight_%s_%s_%s_rank%s" % (k, nmesh, fieldnameadd, rank),
                fieldlist[k].value,
            )

        if compensate:
            fieldlist[k] = fieldlist[k].r2c()
            fieldlist[k] = fieldlist[k].apply(CompensateCICAliasing, kind="circular")

    get_memory(rank)
    sys.stdout.flush()

    #######################################################################################################################
    #################################### Adjusting for growth #############################################################

    field_dict = {
        "1": fieldlist[0],
        r"$\delta_L$": fieldlist[1],
        r"$\delta^2$": fieldlist[2],
        r"$s^2$": fieldlist[3],
        r"$\nabla^2\delta$": fieldlist[4],
    }
    labelvec = ["1", r"$\delta_L$", r"$\delta^2$", r"$s^2$", r"$\nabla^2\delta$"]

    pkclass = Class()
    pkclass.set(configs["Cosmology"])
    pkclass.compute()

    z_ic = configs["z_ic"]
    D = pkclass.scale_independent_growth_factor(zbox)
    D = D / pkclass.scale_independent_growth_factor(z_ic)

    mpiprint(D, rank)
    # If not including nabla field
    # growthratvec = np.array([1, D, D**2, D**2, D**3, D**4, D**2, D**3, D**4, D**4])

    growthratvec = np.array(
        [
            1,
            D,
            D**2,
            D**2,
            D**3,
            D**4,
            D**2,
            D**3,
            D**4,
            D**4,
            D,
            D**2,
            D**3,
            D**3,
            D**2,
        ]
    )
    #######################################################################################################################
    #################################### Measuring P(k) ###################################################################
    kpkvec = []
    #    rxivec = []
    pkcounter = 0
    for i in range(5):
        for j in range(5):
            if i < j:
                pass
            if i >= j:
                pk = FFTPower(
                    field_dict[labelvec[i]],
                    "1d",
                    second=field_dict[labelvec[j]],
                    BoxSize=Lbox,
                    Nmesh=nmesh,
                )

                # The xi measurements don't work great for now.
                # xi = FFTCorr(field_dict[labelvec[i]], '1d', second = field_dict[labelvec[j]], BoxSize=Lbox, Nmesh=nmesh)
                pkpower = pk.power["power"].real
                # xicorr = xi.corr['corr'].real
                if i == 0 and j == 0:
                    kpkvec.append(pk.power["k"])
                #    rxivec.append(xi.corr['r'])
                kpkvec.append(pkpower * growthratvec[pkcounter])
                # rxivec.append(xicorr*growthratvec[pkcounter])
                pkcounter += 1
                mpiprint(("pk done ", pkcounter), rank)

    kpkvec = np.array(kpkvec)
    # rxivec = np.array(rxivec)
    if rank == 0:
        np.savetxt(
            componentdir
            + "lakelag_mpi_pk_box%s_a%.2f_nmesh%s.txt" % (boxno, 1 / (zbox + 1), nmesh),
            kpkvec,
        )

        print(time.time() - start_time)


if __name__ == "__main__":
    ################################### YAML /Initial Config stuff #################################
    yamldir = sys.argv[1]
    fieldnameadd = sys.argv[2]

    configs = yaml.load(open(yamldir, "r"), yaml.FullLoader)
    measure_basis_spectra(configs)