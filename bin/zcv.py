import numpy as np
import sys, os
from fields.make_lagfields import make_lagfields
from fields.measure_basis import advect_fields, measure_basis_spectra
from fields.common_functions import get_snap_z, measure_pk
from fields.field_level_bias import measure_field_level_bias
from mpi4py_fft import PFFT
from anzu.utils import combine_real_space_spectra, combine_measured_rsd_spectra
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from classy import Class
from mpi4py import MPI
import pmesh
from copy import copy
from glob import glob
from yaml import Loader
import sys, yaml
import h5py
import gc

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


def pk_list_to_vec(pk_ij_list):

    nspec = len(pk_ij_list)
    keys = list(pk_ij_list[0].keys())
    k = pk_ij_list[0]['k']
    nk = k.shape[0]
    
    if 'mu' in keys:
        mu = pk_ij_list[0]['mu']  
        nmu = mu.shape[-1]
    else:
        nmu = 1
        mu = None
    
    if 'power_poles' in keys:
        npoles = pk_ij_list[0]['power_poles'].shape[0]
        has_poles = True
        pk_pole_array = np.zeros((nspec, npoles, nk))
        
    else:
        npoles = 1
        has_poles = False
        
        
    pk_wedge_array = np.zeros((nspec, nk, nmu))
        
    for i in range(nspec):
        #power_wedges is always defined, even if only using 1d pk (then wedge is [0,1])
        pk_wedges = pk_ij_list[i]['power_wedges']
        pk_wedge_array[i,...] = pk_wedges.reshape(nk,-1)
        
        if has_poles:
            pk_poles = pk_ij_list[i]['power_poles']
            pk_pole_array[i,...] = pk_poles
            
    if has_poles:
        return k, mu, pk_wedge_array, pk_pole_array
    else:
        return k, mu, pk_wedge_array, None

def measure_2pt_bias(k, pk_ij_heft, pk_tt, kmax, rsd=False):
    
    kidx = k.searchsorted(kmax)
    kcut = k[:kidx[0]]
    pk_tt_kcut = pk_tt[:kidx]
    pk_ij_heft_kcut = pk_ij_heft[:,...,:kidx,np.newaxis]
    
    if not rsd:
        loss = lambda bvec : np.sum((pk_tt_kcut - combine_real_space_spectra(kcut, pk_ij_heft_kcut, bvec)[:,0])**2/(pk_tt_kcut**2))
        bvec0 = [1, 0, 0, 0, 0]
    else:
        loss = lambda bvec : np.sum((pk_tt_kcut - combine_measured_rsd_spectra(kcut, pk_ij_heft_kcut, None, bvec)[:,0])**2/(pk_tt_kcut**2))
        bvec0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    out = minimize(loss, bvec0)
    
    return out

def get_linear_field(config, lag_field_dict, rank, size, nmesh, bias_vec=None):
    
    boltz = Class()
    boltz.set(config["Cosmology"])
    boltz.compute()
    z_ic = config["z_ic"]        
    z_this = get_snap_z(config["particledir"], config["sim_type"])
    D = boltz.scale_independent_growth_factor(z_this)
    D = D / boltz.scale_independent_growth_factor(z_ic)        
    f = boltz.scale_independent_growth_factor_f(z_this)

    if bias_vec is not None:
        b = bias_vec[0]
    else:
        b = 1

    if config['rsd']:
        delta, fft = real_to_redshift_space(lag_field_dict['delta'], nmesh, Lbox, rank, size, f, b=b)
    else:
        delta = lag_field_dict['delta']

    grid = np.meshgrid(
        np.arange(nmesh)[rank * nmesh // size : (rank + 1) * nmesh // size],
        np.arange(nmesh),
        np.arange(nmesh),
        indexing="ij",
    )
    pos_x = (
        (grid[0] / nmesh) % 1
    ) * Lbox
    pos_y = (
        (grid[1] / nmesh) % 1
    ) * Lbox
    pos_z = (
        (grid[2] / nmesh) % 1
    ) * Lbox
    pos = np.stack([pos_x, pos_y, pos_z])
    pos = pos.reshape(3, -1).T
    del pos_x, pos_y, pos_z
        
    layout = pm.decompose(pos)
    p = layout.exchange(pos)
    d = layout.exchange(delta.flatten())

    mesh = pm.paint(p, mass=d)
    del p, d, pos

    mesh = mesh.r2c() #don't need to dealias, because 'particles' are all on grid points
    field_dict = {'delta':mesh}
    field_D = [D]
    
    return field_dict, field_D, z_this

def real_to_redshift_space(field, nmesh, lbox, rank, nranks, f, fft=None, b=1):
    
    if fft is None:
        N = np.array([nmesh, nmesh, nmesh], dtype=int)
        fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype="float32", grid=(-1,))

    field_k = fft.forward(field)

    kvals = np.fft.fftfreq(nmesh) * (2 * np.pi * nmesh) / lbox
    kvalsmpi = kvals[rank * nmesh // nranks : (rank + 1) * nmesh // nranks]
    kvalsr = np.fft.rfftfreq(nmesh) * (2 * np.pi * nmesh) / lbox

    kx, ky, kz = np.meshgrid(kvalsmpi, kvals, kvalsr)
    knorm = kx ** 2 + ky ** 2 + kz ** 2
    mu = kz / np.sqrt(knorm)
    
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1
        mu[0][0][0] = 0

    rsdfac = b + f * mu**2
    del kx, ky, kz, mu
        
    field_k_rsd = field_k * rsdfac
    field_rsd = fft.backward(field_k_rsd)

    return field_rsd, fft

if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size    

    config = sys.argv[1]
    if len(sys.argv)>2:
        tracer_files = sys.argv[2:]
    else:
        tracer_files = None

    with open(config, "r") as fp:
        config = yaml.load(fp, Loader=Loader)
    
#    config['compute_cv_surrogate'] = True
#    config['scale_dependent_growth'] = False
    lattice_type = int(config.get('lattice_type', 0))
    config['lattice_type'] = lattice_type
    lindir = config["outdir"]

    if tracer_files is None:
        tracer_files= [config['tracer_file']]
        
    kmax = np.atleast_1d(config['field_level_kmax'])
    nmesh = int(config['nmesh_out'])
    Lbox = float(config['lbox'])    
    linear_surrogate = config.get('linear_surrogate', False)
    measure_cross_spectra = config.get('measure_cross_spectra', True)
    
    if config['compute_cv_surrogate']:
        basename = "mpi_icfields_nmesh_filt"
    else:
        basename = "mpi_icfields_nmesh"
        
    if 'bias_vec' in config:
        bias_vec = config['bias_vec']
        field_level_bias = False
        M = None
    else:
        bias_vec=[None]*len(tracer_files)
        field_level_bias = config.get('field_level_bias', False)
        M_file = config.get('field_level_M', None)
        if M_file:
            M = np.load(M_file)
        else:
            M = None
            

    if config['scale_dependent_growth']:
        z = get_snap_z(config["particledir"], config["sim_type"])
        lag_field_dict = make_lagfields(config, save_to_disk=False, z=z)
    else:
        linfields = glob(lindir + "{}_{}_*_np.npy".format(basename, nmesh))
        if len(linfields)==0:
            lag_field_dict = make_lagfields(config, save_to_disk=True)
        elif linear_surrogate:
            lag_field_dict = {}
            arr = np.load(
                lindir + "{}_{}_{}_np.npy".format(basename, nmesh, 'delta'),
                mmap_mode="r",
            )        
            lag_field_dict['delta'] = arr[rank * nmesh // size : (rank + 1) * nmesh // size, :, :]
            keynames = ['delta']
            labelvec = ['delta']
        else:
            lag_field_dict = None    
        
    #advect ZA fields
    if not linear_surrogate:
        pm, field_dict, field_D, keynames, labelvec, zbox = advect_fields(config, lag_field_dict=lag_field_dict)
    else:
        pm = pmesh.pm.ParticleMesh(
            [nmesh, nmesh, nmesh], Lbox, dtype="float32", resampler="cic", comm=comm
        )
        if len(tracer_files)==1:
            field_dict, field_D, zbox = get_linear_field(config, lag_field_dict, rank, size, nmesh, bias_vec=bias_vec[0])

    # load tracers and deposit onto mesh.
    # TODO: generalize to accept different formats

    if linear_surrogate:
        stype = 'l'
    elif config['compute_cv_surrogate']:
        stype = 'z'
    else:
        stype = 'heft'
    
    for ii, tracer_file in enumerate(tracer_files):

        if linear_surrogate and (len(tracer_files)>1):
            field_dict, field_D, zbox = get_linear_field(config, lag_field_dict, rank, size, nmesh, bias_vec=bias_vec[ii])            
    
        if config['rsd']:
            tracer_pos = h5py.File(tracer_file)['pos_zspace'][rank::size,:]
        else:
            tracer_pos = h5py.File(tracer_file)['pos_rspace'][rank::size,:]
        
        layout = pm.decompose(tracer_pos)
        p = layout.exchange(tracer_pos)
        tracerfield = pm.paint(p, mass=1, resampler="cic")
        tracerfield = tracerfield / tracerfield.cmean() - 1
        tracerfield = tracerfield.r2c()
        tracerfield.apply(CompensateCICAliasing, kind='circular')
        del tracer_pos, p
        
        #measure tracer auto-power
        pk_tt_dict = measure_pk(tracerfield, tracerfield, Lbox, nmesh, config['rsd'], config['use_pypower'], 1, 1)
    
        field_dict2 = {'t':tracerfield}
        field_D2 = [1]

        np.save(
            lindir
            + "{}_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
                tracer_file.split('/')[-1], config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
            ),
            [pk_tt_dict],
        )
        
        if measure_cross_spectra:
    
            pk_auto_vec, pk_cross_vec = measure_basis_spectra(
                config,
                field_dict,
                field_D,
                keynames,
                labelvec,
                zbox,
                field_dict2=field_dict2,
                field_D2=field_D2,
                save=False
            )
    
            np.save(
                lindir
                + "{}cv_surrogate_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(stype,
                                                                                        config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
                ),
                pk_auto_vec,
            )    
    
    
            np.save(
                lindir
                + "{}cv_cross_{}_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(stype,
                                                                                  tracer_file.split('/')[-1], config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
                ),
                pk_cross_vec,
            )        

        if (bias_vec[ii] is None) & field_level_bias:
            if '1m' in field_dict:
                dm = field_dict.pop('1m')
                d = field_D[0]
                field_D = field_D[1:]
            else:
                dm = None
            M, A, bv, zafield = measure_field_level_bias(comm, pm, tracerfield, field_dict, field_D, nmesh, kmax, Lbox, M=M)

            if dm is not None:
                field_dict['1m'] = dm
                temp = np.copy(field_D)
                field_D = np.zeros(len(field_D)+1)
                field_D[0] = d
                field_D[1:] = temp

            np.save('{}/b_{}cv_rsd={}_kmax{:.4f}_{}_a{:.4f}.npy'.format(config['outdir'], stype, config['rsd'], kmax[0], tracer_file.split('/')[-1], 1 / (zbox + 1)), bv)
            np.save('{}/M_{}cv_rsd={}_kmax{:.4f}_a{:.4f}.npy'.format(config['outdir'], stype, config['rsd'], kmax[0], 1 / (zbox + 1)), M)
            np.save('{}/A_{}cv_rsd={}_kmax{:.4f}_{}_a{:.4f}.npy'.format(config['outdir'], stype, config['rsd'], kmax[0], tracer_file.split('/')[-1], 1 / (zbox + 1)), A)                
            
        elif bias_vec is not None:
            zafield = field_dict['1cb'].copy()
            counter = 0
            for j, k in enumerate(field_dict):
                if (k=='1m') | (k=='1cb'): continue

                try:
                    zafield += bias_vec[ii][counter] * field_D[counter] * field_dict[k]
                    counter += 1
                except IndexError as e:
                    continue

        eps = tracerfield - zafield
        pk_ee = measure_pk(eps, eps, Lbox, nmesh, config['rsd'], config['use_pypower'], 1, 1)
        pk_zz_fl = measure_pk(zafield, zafield, Lbox, nmesh, config['rsd'], config['use_pypower'], 1, 1)

        np.save(
            lindir
            + "{}cv_surrogate_{}_resid_pk_rsd={}_kmax{:0.4f}_opmax{}_pypower={}_a{:.4f}_nmesh{}.npy".format(stype, tracer_file.split('/')[-1], config['rsd'], kmax[0], 4, config['use_pypower'], 1 / (zbox + 1), nmesh),
            [pk_ee],
        )
        np.save(
            lindir
            + "{}cv_surrogate_{}_fieldsum_pk_rsd={}_kmax{:.4f}_opmax{}_pypower={}_a{:.4f}_nmesh{}.npy".format(stype, tracer_file.split('/')[-1], config['rsd'], kmax[0], 4, config['use_pypower'], 1 / (zbox + 1), nmesh),
            [pk_zz_fl],
        )
                    

