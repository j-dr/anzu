from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
from scipy.interpolate import interp1d
from scipy.integrate import simps
from classy import Class
import numpy as np
import yaml 

def _cleft_pk(k, p_lin, D=None, cleftobj=None, kecleft=True):
    '''
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            If kecleft==True, then should be for z=0, and redshift evolution is
            handled by passing the appropriate linear growth factor to D.
        D: float
            Linear growth factor. Only required if kecleft==True.
        kecleft: bool
            Whether to use kecleft or not. Setting kecleft==True
            allows for faster computation of spectra keeping cosmology
            fixed and varying redshift if the cleftobj from the
            previous calculation at the same cosmology is provided to
            the cleftobj keyword.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    '''

    if kecleft:
        if D is None:
            raise(ValueError('Must provide growth factor if using kecleft'))

        # If using kecleft, check if we're only varying the redshift

        if cleftobj is None:
            # Function to obtain the no-wiggle spectrum.
            # Not implemented yet, maybe Wallisch maybe B-Splines?
            # pnw = p_nwify(pk)
            # For now just use Stephen's standard savgol implementation.
            cleftobj = RKECLEFT(k, p_lin)

        # Adjust growth factors
        cleftobj.make_ptable(D=D, kmin=k[0], kmax=k[-1], nk=1000)
        cleftpk = cleftobj.pktable.T

    else:
        # Using "full" CLEFT, have to always do calculation from scratch
        cleftobj = CLEFT(k, p_lin, N=2700, jn=10, cutoff=1)
        cleftobj.make_ptable()

        cleftpk = cleftobj.pktable.T

        # Different cutoff for other spectra, because otherwise different
        # large scale asymptote

        cleftobj = CLEFT(k, p_lin, N=2700, jn=5, cutoff=10)
        cleftobj.make_ptable()

    cleftpk[3:, :] = cleftobj.pktable.T[3:, :]
    cleftpk[2, :] /= 2
    cleftpk[6, :] /= 0.25
    cleftpk[7, :] /= 2
    cleftpk[8, :] /= 2

    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')

    return cleftspline, cleftobj


def _lpt_pk(k, p_lin, f, cleftobj=None, kecleft=False, zenbu=True, 
            third_order=True, one_loop=True, cutoff=np.pi*700/525.):
    '''
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            If kecleft==True, then should be for z=0, and redshift evolution is
            handled by passing the appropriate linear growth factor to D.
        D: float
            Linear growth factor. Only required if kecleft==True.
        kecleft: bool
            Whether to use kecleft or not. Setting kecleft==True
            allows for faster computation of spectra keeping cosmology
            fixed and varying redshift if the cleftobj from the
            previous calculation at the same cosmology is provided to
            the cleftobj keyword.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    '''

    lpt = LPT_RSD(k, p_lin, kIR=0.2, cutoff=np.pi * 700. / 525.,
                          extrap_min=-4, extrap_max = 3,
                          threads=1, jn=5, third_order=third_order,
                         one_loop=one_loop)
    lpt.make_pltable(f, kv=k, nmax=4, 
                     apar=1, aperp=1)
    p0table = lpt.p0ktable
    p2table = lpt.p2ktable
    p4table = lpt.p4ktable
    
    p0table[:, 1] /= 2
    p0table[:, 5] /= 0.25
    p0table[:, 6] /= 2
    p0table[:, 7] /= 2    
    
    p2table[:,1] /= 2
    p2table[:,5] /= 0.25
    p2table[:,6] /= 2
    p2table[:,7] /= 2   
    
    p4table[:,1] /= 2
    p4table[:,5] /= 0.25
    p4table[:,6] /= 2
    p4table[:,7] /= 2  
    
    pktable = np.zeros((len(p0table), 3, p0table.shape[-1]))
    pktable[:,0,:] = p0table
    pktable[:,1,:] = p2table
    pktable[:,2,:] = p4table

    pellspline = interp1d(k, pktable.T, fill_value='extrapolate')
    p0spline = interp1d(k, p0table.T, fill_value='extrapolate')
    p2spline = interp1d(k, p2table.T, fill_value='extrapolate')
    p4spline = interp1d(k, p4table.T, fill_value='extrapolate')

#    return lptspline, lpt_pkell
    return lpt, p0spline, p2spline, p4spline, pellspline

def lpt_spectra(k, z, anzu_config, pkclass=None):
    
    if pkclass==None:
        with open(anzu_config, 'r') as fp:
            cfg = yaml.load(fp, Loader=Loader)
        pkclass = Class()
        pkclass.set(cfg["Cosmology"])
        pkclass.compute()
        
    kt = np.logspace(-3,1,100)
    cutoff = np.pi * cfg['nmesh_in'] / cfg['lbox']
    pk_m_lin = np.array([pkclass.pk_cb_lin(ki, np.array([0])) * cfg['Cosmology']['h']**3 for ki in kt * cfg['Cosmology']['h']])
    Dthis = pkclass.scale_independent_growth_factor(z)
    f = pkclass.scale_independent_growth_factor_f(z)

    lptobj, p0spline, p2spline, p4spline, pellspline = _lpt_pk(kt, pk_m_lin*Dthis**2, f, cutoff=cutoff, third_order=False, one_loop=False)
    pk_zenbu = pellspline(k)

    lptobj, p0spline, p2spline, p4spline, pellspline = _lpt_pk(kt, pk_m_lin*Dthis**2, f, cutoff=cutoff, third_order=True)
    pk_3lpt = pellspline(k)
    
    return pk_zenbu[:11], pk_3lpt[:11], pkclass


def combine_real_space_spectra(k, spectra, bias_params, cross=False):
    
    pkvec = np.zeros((14, spectra.shape[1], spectra.shape[2]))
    pkvec[:10, ...] = spectra
    # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
    nabla_idx = [0, 1, 3, 6]

    # Higher derivative terms
    pkvec[10:, ...] = -k[np.newaxis, :,
                         np.newaxis]**2 * pkvec[nabla_idx, ...]

    b1, b2, bs, bk2, sn = bias_params
    if not cross:
        bterms = [1,
                  2*b1, b1**2,
                  b2, b2*b1, 0.25*b2**2,
                  2*bs, 2*bs*b1, bs*b2, bs**2,
                  2*bk2, 2*bk2*b1, bk2*b2, 2*bk2*bs]
    else:
        # hm correlations only have one kind of <1,delta_i> correlation
        bterms = [1,
                  b1, 0,
                  b2/2, 0, 0,
                  bs, 0, 0, 0,
                  bk2, 0, 0, 0]

    p = np.einsum('b, bkz->kz', bterms, pkvec)

    if not cross:
        p += sn

    return p    

def combine_measured_rsd_spectra(k, spectra_poles, spectra_wedges, bias_params, ngauss=3, nus=None):
    
    pkvec = np.zeros((17, spectra_poles.shape[1], spectra_poles.shape[2]))
    pkvec[:10, ...] = spectra_poles[:10,...]
    
    # TODO: implement counter terms when summing over numerical spectra
    #if nus is None:
    #    nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
    #    nus_calc = nus[0:ngauss]
    #    nus = nus[0:ngauss]
    #    n_nu = ngauss
    #    leggauss = True
    #else:
    #    nus = nus.T
    #    n_nu = nus.shape[0]
    #    leggauss = False
    #    
    #L0 = np.polynomial.legendre.Legendre((1))(nus)
    #L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
    #L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)        
    #    
    #pk_counter = np.zeros((4, n_nu, spectra_poles.shape[2]))
    #pk_stoch = np.zeros((3, n_nu, spectra_poles.shape[2]))
    #
    #pk_counter[0,:,:] = k[np.newaxis,:]**2 * spectra_wedges[0,:,:].T
    #pk_counter[1,:,:] = k[np.newaxis,:]**2 * nus**2 * spectra_wedges[0,:,:].T   
    #pk_counter[2,:,:] = k[np.newaxis,:]**2 * nus**4 * spectra_wedges[0,:,:].T    
    #pk_counter[3,:,:] = k[np.newaxis,:]**2 * nus**6 * spectra_wedges[0,:,:].T     
    #
    #pk_stoch[0,:,:] = 1
    #pk_stoch[1,:,:] = k[np.newaxis,:]**2 * nus**2 
    #pk_stoch[2,:,:] = k[np.newaxis,:]**4 * nus**4
    #
    #if leggauss:
    #    pkvec[10:14,0,...] = 0.5 * np.sum((ws*L0)[np.newaxis,:ngauss,np.newaxis]*pk_counter,axis=1)
    #    pkvec[10:14,1,...] = 2.5 * np.sum((ws*L2)[np.newaxis,:ngauss,np.newaxis]*pk_counter,axis=1)
    #    pkvec[10:14,2,...] = 4.5 * np.sum((ws*L4)[np.newaxis,:ngauss,np.newaxis]*pk_counter,axis=1)

    #    pkvec[14:,0,...] = 0.5 * np.sum((ws*L0)[np.newaxis,:ngauss,np.newaxis]*pk_stoch,axis=1)
    #    pkvec[14:,1,...] = 2.5 * np.sum((ws*L2)[np.newaxis,:ngauss,np.newaxis]*pk_stoch,axis=1)
    #    pkvec[14:,2,...] = 4.5 * np.sum((ws*L4)[np.newaxis,:ngauss,np.newaxis]*pk_stoch,axis=1) 
    #else:
    #    pkvec[10:14,0,...] = 0.5 * simps(L0[np.newaxis,...]*pk_counter, x=nus, axis=1)
    #    pkvec[10:14,1,...] = 2.5 * simps(L2[np.newaxis,...]*pk_counter, x=nus, axis=1)
    #    pkvec[10:14,2,...] = 4.5 * simps(L4[np.newaxis,...]*pk_counter, x=nus, axis=1)

    #    pkvec[14:,0,...] = 0.5 * simps(L0[np.newaxis,...]*pk_stoch, x=nus, axis=1)
    #    pkvec[14:,1,...] = 2.5 * simps(L2[np.newaxis,...]*pk_stoch, x=nus, axis=1)
    #    pkvec[14:,2,...] = 4.5 * simps(L4[np.newaxis,...]*pk_stoch, x=nus, axis=1)        
    #
    ## IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.

    b1,b2,bs,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bias_params
    bias_monomials = np.array([1, 2*b1, b1**2, b2, b1*b2, 0.25*b2**2, 2*bs, 2*b1*bs, b2*bs, bs**2, alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])
    
    p0 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,0,:], axis=0)
    p2 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,1,:], axis=0)
    p4 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,2,:], axis=0)

    return p0, p2, p4

