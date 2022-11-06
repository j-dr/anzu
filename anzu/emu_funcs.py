import time
import os
import numpy as np
import chaospy as cp
import warnings
import GPy

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def norm(x, x_mean=None, x_mult=None):

    if x_mean is None:
        x_mean = np.mean(x, axis=0)

    if x_mult is None:
        x_mult = 2 / (np.max(x, axis=0) - np.min(x, axis=0))

    x_normed = (x - x_mean[np.newaxis, ...]) * x_mult[np.newaxis, ...]

    return x_normed, x_mean, x_mult


def unnorm(x_normed, x_mean, x_mult):

    x = x_normed / x_mult[np.newaxis, ...] + x_mean[np.newaxis, ...]

    return x


class LPTEmulator(object):

    """ Main emulator object """

    def __init__(
        self,
        nbody_training_data_file=None,
        lpt_training_data_file=None,
        kbin_file=None,
        zs=None,
        training_cosmo_file=None,
        surrogate_type="PCE",
        smooth_spectra=True,
        window=11,
        savgol_order=3,
        kmin=0.1,
        kmax=1.0,
        extrap=True,
        kmin_pl=0.5,
        kmax_pl=0.6,
        use_physical_densities=True,
        usez=False,
        zmax=2.0,
        use_sigma_8=True,
        forceLPT=True,
        offset=False,
        tanh=True,
        kecleft=False,
        hyperparams=None,
        aemulus_alpha_settings=False
    ):
        """
        Initialize the emulator object. Default values for all kwargs were
        used for fiducial results in 2101.11014, so don't change these unless
        you have a good reason!

        Kwargs:
            nbody_training_data_file : string
                File name containing the spectra that will be used to train the emulator.
            lpt_training_data_file : string
                File name containing the LPT spectra at the same cosmologies as the spectra
                in the nbody_training_data_file.
            kbin_file : string
                File containing the array of k values that spectra are measured at.
            zs : array like
                Array containing the redshifts that spectra are measured at.
            training_cosmo_file : string
                File name containing the cosmologies that the training spectra are measured at.
            surrogate_type: string
                Type of surrogate model to use. Only "PCE" is currently supported.
            smooth_spectra : bool
                Whether to apply a Savitsky-Golay smoothing to the training spectra
            window : int
                If smooth_spectra==True, then window specifies the window size to use
                when smoothing.
            savgol_order : int
                If smooth_spectra==True, then window specifies the order to use
                when smoothing.
            kmin : float
                Minimum k value that we will build the emulator for. For k<kmin
                pure LPT will be used.
            kmax : float
                Maximum k value to build the emulator for. The model will not make predictions
                for k>kmax.
            extrap : bool
                Whether to apply a power law extrapolation to the 1-1, 1-delta, and delta-delta
                LPT spectra at high k before constructing n-body / LPT ratios.
            kmin_pl : float
                Minimum k value to fit the power law extrapolation to.
            kmax_pl : float
                Maximum k value to fit the power law extrapolation to.
            use_physical_densities : bool
                Whether or not to use ombh^2 and omch^2 instead of om and oc to train emulator.
            usez : bool
                Whether or not to use redshift (as opposed to scale factor) to train the emulator.
            zmax : bool
                Maximum redshift value that will be used in training.
            use_sigma_8 : bool
                Whether to use sigma_8 instead of A_s when training the emulator.
            forceLPT : bool
            offset : bool
            tanh : bool
            kecleft: bool
                Sets whether to use "full" CLEFT or "k-expanded" CLEFT to make LPT predictions. 
                KECLEFT mode allows you to quickly compute spectra at fixed cosmology as a function
                of redshift.


        """

        if nbody_training_data_file is None:
            nbody_training_data_file = "spectra_aem_compensated_43.npy"

        if lpt_training_data_file is None:
            if kecleft:
                lpt_training_data_file = "kecleft_spectra_43.npy"
            else:
                lpt_training_data_file = "cleft_spectra_43.npy"

        if kbin_file is None:
            kbin_file = "kbins.npy"

        if training_cosmo_file is None:
            training_cosmo_file = "cosmos_43.txt"

        self.nbody_training_data_file = nbody_training_data_file
        self.kbin_file = kbin_file
        self.lpt_training_data_file = lpt_training_data_file
        self.training_cosmo_file = training_cosmo_file
        self.surrogate_type = surrogate_type
        self.use_physical_densities = use_physical_densities
        self.use_sigma_8 = use_sigma_8

        if zs is None:
            self.zs = np.array([3.0, 2.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
        else:
            self.zs = zs

        self.smooth_spectra = smooth_spectra
        self.window = window
        self.savgol_order = savgol_order
        self.usez = usez
        self.zmax = zmax
        self.kmin = kmin
        self.kmax = kmax
        self.extrap = extrap
        self.kmin_pl = kmin_pl
        self.kmax_pl = kmax_pl
        self.forceLPT = forceLPT
        if aemulus_alpha_settings:
            self.nspec = 10
        else:
            self.nspec = 14
        self.aemulus_alpha_settings = aemulus_alpha_settings

        self.param_mean = None
        self.param_mult = None
        self.offset = offset
        self.tanh = tanh

        # KECLEFT attributes
        self.kecleft = kecleft
        if self.kecleft and self.extrap:
            warnings.warn("kecleft and extrap are both set. Setting extrap to False.")
        if self.kecleft:
            self.extrap = False

        self._load_data()

        self._build_emulator(hyperparams=hyperparams)

    def _powerlaw_extrapolation(self, spectra, k=None):
        """    

        fit power law indices to all the 1,1 and 1,delta spectra for high k extrapolation
        kmin, kmax are the values used for this extrapolation.

        optionally, feed in a 'k' parameter which sets kbins to re-compute the extrapolated spectra
        """
        k_idx = np.where((self.kmin_pl < self.k) & (self.k < self.kmax_pl))[0]

        # this assumes that 1,1 and 1,delta are the first two spectra and
        # that these are the only ones that need power law extrapolation
        alpha = np.log(spectra[..., :2, k_idx[0]] / spectra[..., :2, k_idx[-1]]) / (
            np.log(self.k[k_idx[0]]) - np.log(self.k[k_idx[-1]])
        )
        p0 = spectra[..., :2, k_idx[-1]] / (self.k[k_idx[-1]] ** alpha)
        k_idx = self.k > self.kmax_pl

        spectra[..., :2, k_idx] = (
            p0[..., np.newaxis] * self.k[k_idx] ** alpha[..., np.newaxis]
        )
        if k is not None:
            specspline = interp1d(self.k, spectra, axis=-1, fill_value="extrapolate")
            spectra_out = specspline(k)
        else:
            spectra_out = 1.0 * spectra
        return spectra_out

    def _load_data(self):
        try:
            aem_file = "/".join(
                [
                    os.path.dirname(os.path.realpath(__file__)),
                    "data",
                    self.nbody_training_data_file,
                ]
            )
            lpt_file = "/".join(
                [
                    os.path.dirname(os.path.realpath(__file__)),
                    "data",
                    self.lpt_training_data_file,
                ]
            )
            k_file = "/".join(
                [os.path.dirname(os.path.realpath(__file__)), "data", self.kbin_file]
            )

            self.spectra_aem = np.load(aem_file)
            self.spectra_lpt = np.load(lpt_file)
            self.k = np.load(k_file)

        except:
            aem_file = self.nbody_training_data_file
            lpt_file = self.lpt_training_data_file
            k_file = self.kbin_file

            self.spectra_aem = np.load(aem_file)
            self.spectra_lpt = np.load(lpt_file)
            self.k = np.load(k_file)

    def _get_pcs(self, evec_spec, spectra, npc):

        nout = np.prod(spectra.shape[:2])
        pcs_spec = np.zeros((nout, self.nspec, self.npc))

        for si in range(self.nspec):
            pcs_spec[:, si, :] = np.dot(
                spectra[:, :, si, :].reshape(-1, self.nk), evec_spec[si, :, :npc]
            )

        return pcs_spec

    def _ratio_and_smooth(self, spectra_aem, spectra_lpt):

        simoverlpt = spectra_aem / spectra_lpt

        # smooth the ratios before taking log
        if self.smooth_spectra:
            simoverlpt = savgol_filter(
                simoverlpt, self.window, self.savgol_order, axis=-1
            )

        simoverlpt = np.log10(simoverlpt)
        simoverlpt[~np.isfinite(simoverlpt)] = 0

        self.zidx = np.min(np.where(self.zs <= self.zmax))
        self.nz = len(self.zs[self.zidx :])

        self.kmax_idx = np.searchsorted(self.k, self.kmax)
        self.kmin_idx = np.searchsorted(self.k, self.kmin)
        self.nk = self.kmax_idx - self.kmin_idx

        simoverlpt = simoverlpt[:, self.zidx :, :, self.kmin_idx : self.kmax_idx]

        return simoverlpt

    def _smooth_transition(self, simoverlpt):
        """
        Additional post-processing on ratios so they're smooth near the transition between LPT
        and the emulator. 

        Does two things:
        1) Computes the offset for log(Nbody/LPT) in the IR range from the mean at every redshift bin
        2) Applies a "high-pass" type filter at low-k so PCs don't go insane.

        Notes:
        Could also try to do a Savgol pass for (2) at low-k instead of the current filter
        """
        nsim, nz, nspec, nk = simoverlpt.shape
        if not self.offset and not self.tanh:
            return simoverlpt

        # kstar = 0.125 where we broadly want the transition to be final
        kstar = 0.125

        # Hard-coded offset, just use window from kmin_idx to kmin_idx+4 for now

        kvals = self.k[self.kmin_idx : self.kmax_idx]

        offidx = (kvals > self.kmin) & (kvals < kstar)

        filter_tanh = 0.5 * (1 + np.tanh(2.5 * (kvals - kstar) / kstar))
        if not self.tanh:
            filter_tanh = np.ones_like(filter_tanh)

        newsimoverlpt = 1.0 * simoverlpt
        for i in range(nz):
            for j in range(nspec):
                meanratio = np.mean(simoverlpt, axis=0)[i, j]

                offset = np.mean(meanratio[offidx])
                if not self.offset:
                    offset = 0

                newsimoverlpt[:, i, j] -= offset
                # Filter only the cubic spectra
                if j in range(nspec):
                    newsimoverlpt[:, i, j] *= filter_tanh

        return newsimoverlpt

    def _setup_training_data(self, spectra_lpt, spectra_aem):

        # apply power law extrapolation to LPT spectra where they diverge at high k
        if self.extrap:

            spectra_lpt = self._powerlaw_extrapolation(spectra_lpt)

        simoverlpt = self._ratio_and_smooth(spectra_aem, spectra_lpt)

        # Smooth the ratios even more/calibrate them to LPT to remove kink
        simoverlpt = self._smooth_transition(simoverlpt)

        self.simoverlpt = simoverlpt

        nsim = len(simoverlpt)

        # Non mean-subtracted PCs
        Xs = np.zeros((self.nspec, self.nk, self.nk))
        for i in range(self.nspec):
            Xs[i, :, :] = np.cov(self.simoverlpt[:, :, i, :].reshape(self.nz * nsim, -1).T)

        # PC basis for each type of spectrum, independent of z and cosmo
        evec_spec = np.zeros((self.nspec, self.nk, self.nk))

        # variance per PC
        vars_spec = np.zeros((self.nspec, self.nk))

        # computing PCs
        for si in range(self.nspec):
            var, pcs = np.linalg.eig(Xs[si, ...])

            evec_spec[si, :, :] = pcs
            vars_spec[si, :] = var

        self.evec_spec = evec_spec
        self.evec_spline = interp1d(
            self.k[self.kmin_idx : self.kmax_idx],
            self.evec_spec[..., : self.npc],
            axis=1,
            fill_value="extrapolate",
        )
        self.vars_spec = vars_spec
        self.pcs_spec = self._get_pcs(self.evec_spec, simoverlpt, self.npc)
        self.pcs_spec_normed, self.pcs_mean, self.pcs_mult = norm(self.pcs_spec)

    def _setup_design(self, cosmofile, param_mean=None, param_mult=None):

        try:
            cosmo_file = "/".join(
                [os.path.dirname(os.path.realpath(__file__)), "data", cosmofile]
            )

            cosmos = np.genfromtxt(cosmo_file, names=True)
        except:
            cosmo_file = cosmofile

            cosmos = np.genfromtxt(cosmo_file, names=True)

        ncosmos = len(cosmos)
        self.training_cosmos = cosmos

        if not self.use_physical_densities:
            if not self.use_sigma_8:
                dt = np.dtype(
                    [
                        ("omegab", np.float),
                        ("omegam", np.float),
                        ("w0", np.float),
                        ("ns", np.float),
                        ("As", np.float),
                        ("H0", np.float),
                        ("nu_mass_ev", np.float),
                    ]
                )
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp["omegab"] = cosmos["ombh2"] / (cosmos["H0"] / 100) ** 2
                cosmos_temp["omegam"] = (cosmos["omch2"] + cosmos["ombh2"]) / (
                    cosmos["H0"] / 100
                ) ** 2
                cosmos_temp["w0"] = cosmos["w0"]
                cosmos_temp["ns"] = cosmos["ns"]
                cosmos_temp["As"] = cosmos["As"]
                cosmos_temp["H0"] = cosmos["H0"]
                if self.aemulus_alpha_settings:
                    cosmos_temp["nu_mass_ev"] = cosmos["Neff"]
                    
                else:
                    cosmos_temp["nu_mass_ev"] = cosmos["nu_mass_ev"]
                cosmos = cosmos_temp
            else:
                dt = np.dtype(
                    [
                        ("omegab", np.float),
                        ("omegam", np.float),
                        ("w0", np.float),
                        ("ns", np.float),
                        ("sigma8", np.float),
                        ("H0", np.float),
                        ("nu_mass_ev", np.float),
                    ]
                )
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp["omegab"] = cosmos["ombh2"] / (cosmos["H0"] / 100) ** 2
                cosmos_temp["omegam"] = (cosmos["omch2"] + cosmos["ombh2"]) / (
                    cosmos["H0"] / 100
                ) ** 2
                cosmos_temp["w0"] = cosmos["w0"]
                cosmos_temp["ns"] = cosmos["ns"]
                cosmos_temp["sigma8"] = cosmos["sigma8"]
                cosmos_temp["H0"] = cosmos["H0"]
                if self.aemulus_alpha_settings:
                    cosmos_temp["nu_mass_ev"] = cosmos["Neff"]
                else:                    
                    cosmos_temp["nu_mass_ev"] = cosmos["nu_mass_ev"]
                cosmos = cosmos_temp
        else:
            if not self.use_sigma_8:
                dt = np.dtype(
                    [
                        ("ombh2", np.float),
                        ("omch2", np.float),
                        ("w0", np.float),
                        ("ns", np.float),
                        ("As", np.float),
                        ("H0", np.float),
                        ("nu_mass_ev", np.float),
                    ]
                )
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp["ombh2"] = cosmos["ombh2"]
                cosmos_temp["omch2"] = cosmos["omch2"]
                cosmos_temp["w0"] = cosmos["w0"]
                cosmos_temp["ns"] = cosmos["ns"]
                cosmos_temp["As"] = cosmos["As"]
                cosmos_temp["H0"] = cosmos["H0"]
                if self.aemulus_alpha_settings:
                    cosmos_temp["nu_mass_ev"] = cosmos["Neff"]
                else:                    
                    cosmos_temp["nu_mass_ev"] = cosmos["nu_mass_ev"]
                cosmos = cosmos_temp

            else:
                dt = np.dtype(
                    [
                        ("ombh2", np.float),
                        ("omch2", np.float),
                        ("w0", np.float),
                        ("ns", np.float),
                        ("sigma8", np.float),
                        ("H0", np.float),
                        ("nu_mass_ev", np.float),
                    ]
                )
                cosmos_temp = np.zeros(ncosmos, dtype=dt)
                cosmos_temp["ombh2"] = cosmos["ombh2"]
                cosmos_temp["omch2"] = cosmos["omch2"]
                cosmos_temp["w0"] = cosmos["w0"]
                cosmos_temp["ns"] = cosmos["ns"]
                cosmos_temp["sigma8"] = cosmos["sigma8"]
                cosmos_temp["H0"] = cosmos["H0"]
                if self.aemulus_alpha_settings:
                    cosmos_temp["nu_mass_ev"] = cosmos["Neff"]
                else:                    
                    cosmos_temp["nu_mass_ev"] = cosmos["nu_mass_ev"]
                cosmos = cosmos_temp

        param_ranges = np.array(
            [[np.min(cosmos[k]), np.max(cosmos[k])] for k in cosmos.dtype.names]
        )
        if self.usez:
            param_ranges = np.vstack([param_ranges, [0, self.zmax]])
        else:
            param_ranges = np.vstack([param_ranges, [1 / (self.zmax + 1), 1]])

        # if self.param_mean and self.param_mult are already defined
        # we will use those. This is mostly useful for our test suites
        param_ranges_scaled, self.param_mean, self.param_mult = norm(
            param_ranges.T, self.param_mean, self.param_mult
        )

        self.param_ranges_scaled = param_ranges_scaled.T

        # Design matrix for the PCs, with 7 parameters (wCDM + z)
        zidx = np.min(np.where(self.zs <= self.zmax))
        z = self.zs[zidx:]
        a = 1 / (1 + z)
        
        if self.degree_cv < 2:
            idx = np.arange(ncosmos) != self.ncv
        else:
            idx = (np.arange(ncosmos) < self.ncv) & (self.ncv + self.degree_cv < np.arange(ncosmos))

        if self.degree_cv<2:
            idx = np.arange(ncosmos) != self.ncv
        else:
            if self.ncv is not None:
                idx = (np.arange(ncosmos) < self.ncv) | (np.arange(ncosmos) >= self.degree_cv + ncv)
            else:
                idx = (np.arange(ncosmos) >= self.degree_cv)
            
        if self.usez:
            design = np.hstack(
                [
                    np.tile(cosmos.view(("<f8", 7)), self.nz)[
                        idx
                    ].reshape(self.nz * (ncosmos - self.degree_cv), 7),
                    np.tile(z, ncosmos - self.degree_cv)[:, np.newaxis],
                ]
            )
        else:
            design = np.hstack(
                [
                    np.tile(cosmos.view(("<f8", 7)), self.nz)[
                        idx
                    ].reshape(self.nz * (ncosmos - self.degree_cv), 7),
                    np.tile(a, ncosmos - self.degree_cv)[:, np.newaxis],
                ]
            )

        design_scaled = (design - self.param_mean[np.newaxis, :]) * self.param_mult[
            np.newaxis, :
        ]

        return design, design_scaled

    def _train_surrogates(self):

        self.surrogates = []
        
        if self.surrogate_type == "PCE":

            distribution = cp.J(
                *[
                    cp.Uniform(
                        self.param_ranges_scaled[i][0], self.param_ranges_scaled[i][1]
                    )
                    for i in range(8)
                ]
            )

            # PCE coefficient regression
            for i in range(self.nspec):
                pce = cp.orth_ttr(
                    self.npoly[i], distribution, cross_truncation=self.qtrunc
                )
                surrogate = cp.fit_regression(
                    pce, self.design_scaled.T, np.real(self.pcs_spec_normed[:, i, :])
                )

                self.surrogates.append(surrogate)

        elif self.surrogate_type == "GP":

            for i in range(self.nspec):
                print('fitting spec {}'.format(i),flush=True)
                if self.independent_pcs:
                    self.surrogates.append([])
                    for j in range(self.npc):
                        print('fitting pc {}'.format(j), flush=True)
                        K = GPy.kern.Matern32(input_dim=len(self.param_mean)) + GPy.kern.White(1)
                        m = GPy.models.GPRegression(self.design_scaled,
                                                    np.real(self.pcs_spec[:, i, j, np.newaxis]),
                                                    normalizer=None,
                                                    kernel=K)

                        if self.optimize_kern:
                            m.optimize(optimizer='bfgs')
                        self.surrogates[i].append(m)
                else:
                    K = GPy.kern.RBF(input_dim=len(self.param_mean),
                                          variance=self.kern_var[i],
                                          lengthscale=self.kern_lenscale[i])
                    m = GPy.models.GPRegression(self.design_scaled,
                                                np.real(self.pcs_spec[:, i, :]),
                                                kernel=K)

                    if self.optimize_kern:
                        m.optimize('bfgs')
                    self.surrogates.append(m)

        else:

            raise (
                ValueError(
                    "Surrogate type {} not implemented!".format(self.surrogate_type)
                )
            )

    def _build_emulator(self, hyperparams=None):
        """
        Trains the emulator with polynomial chaos expansion regression.
        Default values for all kwargs were used for fiducial results in
        2101.11014, so don't change these unless you have a good reason!

        Kwargs:
            npc : int
                number of principal components. This must always be defined,
                because all models we consider build surrogates for principal components.
            npoly : int/array like
                Polynomial order for the PCE. If of int type, the same order
                is used for all parameters. If array like, then needs to be of size
                (n_spec, n_param), where n_spec is the number of bias basis spectra,
                i.e. 10, and n_param is the number of cosmological/redshift
                parameters that the emulator is a function of, i.e. 8. The order is
                the same as in the training cosmology file, redshift/scale factor is last.
            qtrunc : float
                hyperbolic truncation parameter for PCE regression.
            ncv: int
                number of cosmologies to leave out when training, for cross-validation.

        """

        if hyperparams is None:
            self.npc = 2
            ncv = None
            self.ncv = ncv
            self.degree_cv = 0
        else:
            self.npc = hyperparams["npc"]
            ncv = hyperparams["ncv"]
            self.ncv = ncv
            
            if 'degree_cv' in hyperparams:
                self.degree_cv = hyperparams['degree_cv']
            else:
                self.degree_cv = None

            if 'independent_pcs' in hyperparams:
                self.independent_pcs = hyperparams['independent_pcs']
            else:
                self.independent_pcs = True
                
        if self.surrogate_type == "PCE":
            if hyperparams is None:
                npoly = np.array([1, 2, 1, 1, 3, 2, 1, 3])
                npoly = np.tile(npoly, [self.nspec, 1])
                qtrunc = 1
            else:
                if "npoly" not in hyperparams.keys():
                    npoly = np.array([1, 2, 1, 1, 3, 2, 1, 3])
                    npoly = np.tile(npoly, [self.nspec, 1])

                else:
                    npoly = hyperparams["npoly"]
                    if len(npoly.shape) == 1:
                        npoly = np.tile(npoly, [self.nspec, 1])

                if "qtrunc" not in hyperparams.keys():
                    qtrunc = 1
                else:
                    qtrunc = hyperparams["qtrunc"]

            self.npoly = npoly
            self.qtrunc = qtrunc
        elif self.surrogate_type == 'GP':
            if hyperparams is None:
                self.kern_var = np.ones(self.nspec)
                self.kern_lenscale = np.ones(self.nspec)
                self.optimize_kern = True
            else:
                if 'kern_var' in hyperparams.keys():
                    self.kern_var = hyperparams['kern_var']
                else:
                    self.kern_var = np.ones(self.nspec)
                    
                if 'kern_lenscale' in hyperparams.keys():
                    self.kern_lenscale = hyperparams['kern_lenscale']
                else:
                    self.kern_lenscale = np.ones(self.nspec)

                if 'optimize_kern' in hyperparams.keys():
                    self.optimize_kern = bool(hyperparams['optimize_kern'])
                else:
                    self.optimize_kern = True
                    

        # Pulling all of the measured P(k) into a file
        spectra_aem = np.copy(self.spectra_aem)
        spectra_lpt = np.copy(self.spectra_lpt)
        ncosmos = spectra_aem.shape[0]
        
        if self.degree_cv < 2:
            idx = np.arange(ncosmos) != self.ncv
        else:
            if self.ncv is not None:
                idx = (np.arange(ncosmos) < self.ncv) | (np.arange(ncosmos) >= self.degree_cv + ncv)
            else:
                idx = (np.arange(ncosmos) >= self.degree_cv)
        

        spectra_aem = spectra_aem[idx]
        spectra_lpt = spectra_lpt[idx]

        self._setup_training_data(spectra_lpt, spectra_aem)
        self.design, self.design_scaled = self._setup_design(self.training_cosmo_file)
        self._train_surrogates()

        self.trained = True

    def predict(self, k, cosmo, **kwargs):
        """
        Make predictions from a trained emulator given a vector of wavenumbers and
        a cosmology.

        Args:
            k : array-like
                1d vector of wave-numbers. Maximum k cannot be larger than 
                self.kmax. For k < self.kmin, predictions will be made using
                velocileptors, for self.kmin <= k < self.kmax predictions
                use the emulator.
            cosmo : array-like
                Vector containing cosmology/scale factor in the order
                (ombh2, omch2, w0, ns, sigma8, H0, Neff, a).
                If self.use_sigma_8 != True, then ln(A_s/10^{-10})
                should be provided instead of sigma8. If self.usez==True 
                then a should be replaced with redshift.
        Kwargs:
            Kwargs to be passed to _pce_predict.

        Output: 
            Emulator predictions for the 10 basis spectra of the 2nd order lagrangian bias expansion.
            Order of spectra is 1-1, delta-1, delta-delta, delta2-1, delta2-delta, delta2-delta2
            s2-1, s2-delta, s2-delta2, s2-s2.


        """

        if not self.trained:
            raise (ValueError("Need to call build_emulator before making predictions"))

        pk_emu, lambda_surr, pk_var = self._predict(k, cosmo, **kwargs)

        return pk_emu, pk_var

    def basis_to_full(self, k, btheta, emu_spec, cross=True):
        """
        Take an LPTemulator.predict() array and combine with bias parameters to obtain predictions for P_hh and P_hm. 


        Inputs:
        -k: set of wavenumbers used to generate emu_spec.
        -btheta: vector of bias + shot noise. See notes below for structure of terms
        -emu_spec: output of LPTemu.predict() at a cosmology / set of k values
        -halomatter: whether we compute only P_hh or also P_hm

        Outputs:
        -pfull: P_hh (k) or a flattened [P_hh (k),P_hm (k)] for given spectrum + bias params.


        Notes:
        Bias parameters can either be

        btheta = [b1, b2, bs2, SN]

        or

        btheta = [b1, b2, bs2, bnabla2, SN]

        Where SN is a constant term, and the bnabla2 terms follow the approximation

        <X, nabla^2 delta> ~ -k^2 <X, 1>. 

        Note the term <nabla^2, nabla^2> isn't included in the prediction since it's degenerate with even higher deriv
        terms such as <nabla^4, 1> which in principle have different parameters. 


        To-do:
        Include actual measured nabla^2 correlators once the normalization issue has been properly worked out.

        """
        if len(btheta) == 4:
            b1, b2, bs, sn = btheta
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [
                0,
                1,
                0,
                2 * b1,
                b1 ** 2,
                0,
                b2,
                b2 * b1,
                0.25 * b2 ** 2,
                0,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs ** 2,
            ]

            # hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [1, 0, b1, 0, 0, b2 / 2, 0, 0, 0, bs, 0, 0, 0, 0]

            pkvec = emu_spec

        else:
            b1, b2, bs, bk2, sn = btheta
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [
                0,
                1,
                0,
                2 * b1,
                b1 ** 2,
                0,
                b2,
                b2 * b1,
                0.25 * b2 ** 2,
                0,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs ** 2,
                0,
                2 * bk2,
                2 * bk2 * b1,
                bk2 * b2,
                2 * bk2 * bs,
            ]

            # hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [
                1,
                0,
                b1,
                0,
                0,
                b2 / 2,
                0,
                0,
                0,
                bs,
                0,
                0,
                0,
                0,
                bk2,
                0,
                0,
                0,
                0,
            ]

            pkvec = np.zeros(shape=(self.nspec + 4, len(k)))
            pkvec[: self.nspec] = emu_spec

            # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
            if cross:
                nabla_idx = [0, 2, 5, 9]
            else:
                nabla_idx = [1, 3, 6, 10]

            # Higher derivative terms
            pkvec[self.nspec :] = -(k ** 2) * pkvec[nabla_idx]

        if cross:
            bterms_hm = np.array(bterms_hm)
            pfull = np.einsum("b, bk->k", bterms_hm, pkvec)

        else:
            bterms_hh = np.array(bterms_hh)
            pfull = np.einsum("b, bk->k", bterms_hh, pkvec) + sn

        return pfull

    def _predict(
        self,
        k,
        cosmo,
        spec_lpt,
        k_lpt=None,
        lambda_surr=None,
        evec_spec=None,
        simoverlpt=None,
        timing=False,
    ):
        """
        Args:
            k : array-like
                1d vector of wave-numbers.
            cosmo : array-like
                Vector containing cosmology/scale factor in the order
                (ombh2, omch2, w0, ns, sigma8, H0, Neff, a).
                If self.use_sigma_8 != True, then ln(A_s/10^{-10})
                should be provided instead of sigma8. If self.usez==True 
                then a should be replaced with redshift.
        Kwargs:
            lambda_surr : array-like
                Array of shape (n_spec, n_pc) of PC coefficients to use
                to make predictions. Mostly used for validation of PCE/GP procedure.
            spec_lpt : array-like
                LPT predictions for spectra from velocileptors at the specified cosmology
                call. 
            evec_spec : array-like
                Array of PC spectra. For use when validating PCA procedure.
            simoverlpt : array-like
                Array of n-body/lpt ratios. For use when validating PCA procedure.
            timing : bool
                If True, then print timing info.

        Output:
            pk_emu : array-like
                Emulator predictions for the 10 basis spectra of the 2nd order lagrangian bias expansion. 


        """
        if np.max(k) > self.k[self.kmax_idx]:
            raise (
                ValueError(
                    "Trying to compute spectra beyond the maximum value of the emulator!"
                )
            )

        evecs = self.evec_spline(k)
        cosmo_scaled = (cosmo - self.param_mean[np.newaxis, :]) * self.param_mult[
            np.newaxis, :
        ]
        if (k_lpt is not None) & (np.sum(k != k_lpt) > 0):
            lpt_interp = interp1d(k_lpt, spec_lpt, axis=-1, fill_value="extrapolate")
            spectra_lpt = lpt_interp(k)
        else:
            spectra_lpt = spec_lpt

        if spectra_lpt.shape[-1] != len(k) and self.extrap == False:
            raise (
                ValueError(
                    "Trying to feed in lpt spectra computed at different k than the desired outcome!"
                )
            )

        # if we already have PCs, just make prediction using them

        if lambda_surr is None:

            # otherwise, check to see if we have PC vecs and spectra, in which case
            # compute PCs with them
            if evec_spec is not None:
                if simoverlpt is None:
                    raise (
                        ValueError(
                            "need to provide non-linear ratios if want PCA only resids"
                        )
                    )

                lambda_surr = self._get_pcs(evec_spec, simoverlpt, self.npc)
                lambda_surr_normed = None
                lambda_var = np.zeros_like(lambda_surr)
                simoverlpt_var = np.einsum("bkp, cbp->cbk", evecs**2, lambda_var)                

            # otherwise just use the surrogates to compute PCs
            else:
                lambda_surr_normed = np.zeros((len(cosmo), self.nspec, self.npc))
                lambda_var_normed = np.zeros((len(cosmo), self.nspec, self.npc))

                for i in range(self.nspec):
                    start = time.time()
                    if self.surrogate_type == "PCE":

                        lambda_surr_normed[:, i, ...] = self.surrogates[i](
                            *cosmo_scaled.T
                        ).T

                    elif self.surrogate_type == "GP":
                        if self.independent_pcs:
                            for j in range(self.npc):
                                m, v = self.surrogates[i][j].predict(cosmo_scaled)
                                lambda_surr_normed[:, i, j] = m[:,0]
                                lambda_var_normed[:, i, j] = v[:,0]
                        else:
                            (
                                lambda_surr_normed[:, i, ...],
                                lambda_var_normed[:, i, ...],
                            ) = self.surrogates[i].predict(cosmo_scaled)

                    end = time.time()

                    if timing:
                        print("took {}s".format(end - start))

                if self.surrogate_type == 'PCE':
                    lambda_surr = unnorm(lambda_surr_normed, self.pcs_mean, self.pcs_mult)
                else:
                    lambda_surr = lambda_surr_normed
                lambda_var = unnorm(lambda_var_normed, self.pcs_mean, self.pcs_mult**2)
                simoverlpt_var = np.einsum("bkp, cbp->cbk", evecs**2, lambda_var)
                
        simoverlpt_emu = np.einsum("bkp, cbp->cbk", evecs, lambda_surr)


        if self.extrap:
            # Extrap and rebin.
            spectra_lpt = self._powerlaw_extrapolation(spectra_lpt, k)

        pk_emu = np.zeros_like(spectra_lpt)
        var_emu = np.zeros_like(spectra_lpt)
        pk_emu[:] = spectra_lpt
        # Enforce agreement with LPT
        if self.forceLPT:
            pk_emu[..., k > self.kmin] = (10 ** (simoverlpt_emu) * pk_emu)[
                ..., k > self.kmin
            ]
            var_emu[..., k > self.kmin] = (pk_emu**2 * spectra_lpt**2 * (np.log(10) * (simoverlpt_var))**2)[
                ..., k > self.kmin
            ]
        else:
            pk_emu[...] = 10 ** (simoverlpt_emu) * pk_emu[...]
            var_emu[...] = (pk_emu**2 * spectra_lpt**2 * (np.log(10) * (simoverlpt_var))**2)[...]

        return pk_emu, lambda_surr, var_emu

    
