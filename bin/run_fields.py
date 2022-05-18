import numpy as np
from fields.make_lagfields import make_lagfields
from fields.measure_basis import measure_basis_spectra
from mpi4py import MPI
from copy import copy
from glob import glob
from yaml import Loader 
import sys

if __name__ == '__main__':
    
    config = sys.argv[1]
    if len(sys.argv) > 2:
        globsnaps = True
    else:
        globsnaps = False
        
    if globsnaps:
        snapdir = '/'.join(config['particledir'].split('/')[:-1])
        snapglob = '_'.join(snapdir.split('_')[:-1]) + '*'
        snapdirs = glob(snapglob)
        snapstrs = [d.split('_')[-1] for d in snapdirs]
        snapnums = [int(d) for d in snapstrs]
        idx = np.argsort(snapnums)
        snapdirs = snapdirs[idx]
        snapstrs = snapstrs[idx]
        nsnaps = len(snapstrs)
        pdirs = [snapdirs[i] + '/snapshot_{}'.format(snapstrs[i]) \
                    for i in range(nsnaps)]
    else:
        nsnaps = 1
        pdirs = [config['particledir']]
        
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if config['compute_surrogate_cv']:
        config_surr = copy(config)
        config['compute_surrogate_cv'] = False
    
    if rank==0:
        print('Constructing lagrangian fields')
        make_lagfields(config)
        
        if config['compute_surrogate_cv']:
            make_lagfields(config_surr)

    if rank==0:
        print('Processing basis spectra for {} snapshots'.format(nsnaps))
        
        
    for i in range(nsnaps):
        if rank==0:
            print('Processing snapshot {}'.format(i))
            
        config['particledir'] = pdirs[i]
        measure_basis_spectra(config)
        
        if config['compute_surrogate_cv']:
            measure_basis_spectra(config_surr)
