import numpy as np
from make_lagfields import make_lagfields
from measure_basis import measure_basis_spectra
from mpi4py import MPI
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

    if rank==0:
        print('Processing basis spectra for {} snapshots'.format(nsnaps))
        
    for i in range(nsnaps):
        if rank==0:
            print('Processing snapshot {}'.format(i))
            
        config['particledir'] = pdirs[i]
        make_lagfields(config)