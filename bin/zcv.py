import numpy as np
import sys, os
from fields.make_lagfields import make_lagfields
from fields.measure_basis import advect_fields, measure_basis_spectra
from fields.common_functions import get_snap_z, measure_pk
from fields.field_level_bias import measure_field_level_bias
from scipy.optimize import minimize
from mpi4py import MPI
from copy import copy
from glob import glob
from yaml import Loader
import sys, yaml
import h5py
import gc

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

def measure_2pt_bias(k, pk_ij_heft, pk_tt, kmax):
    
    return None

if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size    

    config = sys.argv[1]

    with open(config, "r") as fp:
        config = yaml.load(fp, Loader=Loader)
    
    config['compute_cv_surrogate'] = True
    scale_dependent_growth = bool(config.get("scale_dependent_growth", False))
    lattice_type = int(config.get('lattice_type', 0))
    lindir = config["outdir"]
    tracer_file = config['tracer_file']
    kmax = np.atleast_1d(config['field_level_kmax'])
    nmesh = int(config['nmesh_out'])
    Lbox = float(config['lbox'])    
    basename = "mpi_icfields_nmesh_filt"
    
    if 'bias_vec' in config:
        bias_vec = config['bias_vec']
    else:
        bias_vec=None
        field_level_bias = config.get('field_level_bias', False)
        M_file = config.get('field_level_M', None)
        if M_file:
            M = np.load(M_file)
    
    #create/load surrogate linear fields
    linfields = glob(lindir + "{}_{}_*_np.npy".format(basename, nmesh))
    if not os.path.exists(linfields[0]):
        lag_field_dict = make_lagfields(config, save_to_disk=True)
    else:
        lag_field_dict = None
        
    #advect ZA fields
    pm, field_dict, field_D, keynames, labelvec, zbox = advect_fields(config, lag_field_dict=lag_field_dict)

    # load tracers and deposit onto mesh. 
    # TODO: generalize to accept different formats
    
    if config['rsd']:
        tracer_pos = h5py.File(tracer_file)['pos_zspace'][rank::size,:]
    else:
        tracer_pos = h5py.File(tracer_file)['pos_rspace'][rank::size,:]
        
    layout = pm.decompose(tracer_pos)
    p = layout.exchange(tracer_pos)
    tracerfield = pm.paint(p, mass=1, resampler="cic")
    tracerfield = tracerfield.r2c()
    del tracer_pos, p                
        
    #measure tracer auto-power
    pk_tt_dict = measure_pk(tracerfield, tracerfield, Lbox, nmesh, config['rsd'], config['use_pypower'], 1, 1)
    
    field_dict2 = {'t':tracerfield}
    field_D2 = [1]
    
    pk_auto_vec, pk_cross_vec = measure_basis_spectra(
        config,
        field_dict,
        field_D,
        keynames,
        labelvec,
        zbox,
        lag_field_dict=lag_field_dict,
        field_dict2=field_dict2,
        field_D2=field_D2,
        save=False
    )
    
    np.save(
        lindir
        + "zcv_surrogate_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
            config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
        ),
        pk_auto_vec,
    )    
    
    np.save(
        lindir
        + "{}_auto_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
            tracer_file.split('/')[-1], config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
        ),
        pk_tt_dict,
    )
    
    np.save(
        lindir
        + "zcv_cross_{}_pk_rsd={}_pypower={}_a{:.4f}_nmesh{}.npy".format(
            tracer_file.split('/')[-1], config['rsd'], config['use_pypower'], 1 / (zbox + 1), nmesh
        ),
        pk_cross_vec,
    )        
    
    if bias_vec is None:
        if field_level_bias:
            bias_vec, M, A = measure_field_level_bias(comm, pm, tracerfield, field_dict, field_D, nmesh, kmax, Lbox, M=M)
        else:
            k, mu, pk_tt_wedge_array, pk_tt_pole_array = pk_list_to_vec([pk_tt_dict])
            k, mu, pk_ij_wedge_array, pk_ij_pole_array = pk_list_to_vec(pk_auto_vec)

            if config['rsd']:
                bias_vec = measure_2pt_bias(k, pk_ij_pole_array, pk_tt_pole_array[0,...], kmax)
            else:
                bias_vec = measure_2pt_bias(k, pk_ij_wedge_array, pk_tt_wedge_array[0,...], kmax)
                
            


