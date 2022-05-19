import numpy as np
#import sys
#sys.path.insert(0, '/pscratch/sd/j/jderose/anzu/')
from fields.make_lagfields import make_lagfields
from fields.measure_basis import measure_basis_spectra
from mpi4py import MPI
from copy import copy
from glob import glob
from yaml import Loader 
import sys, yaml



if __name__ == '__main__':
    
    config = sys.argv[1]
    with open(config, 'r') as fp:
        config = yaml.load(fp, Loader=Loader)
        globsnaps = config['globsnaps']
        
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
    
    if config['compute_cv_surrogate']:
        config_surr = copy(config)
        config['compute_cv_surrogate'] = False
    
    if rank==0:
        print('Constructing lagrangian fields', flush=True)

    make_lagfields(config)
        
    if config['compute_cv_surrogate']:
        make_lagfields(config_surr)

    if rank==0:
        print('Processing basis spectra for {} snapshots'.format(nsnaps), flush=True)
        
        
    for i in range(nsnaps):
        if rank==0:
            print('Processing snapshot {}'.format(i), flush=True)
            
        config['particledir'] = pdirs[i]
        measure_basis_spectra(config)
        
        if config['compute_cv_surrogate']:
            measure_basis_spectra(config_surr)
