#### Author of this code: Eva-Maria Ahrer, adapted from Tiberius TransitGPPM model (author: J. Kirk)

import numpy as np

from fitting_utils import lightcurve

import dynesty

from scipy import optimize,stats
import matplotlib.pyplot as plt

from fitting_utils import parametric_fitting_functions as pf
from fitting_utils import plotting_utils as pu
from fitting_utils import priors

class Sampling(object):
    def __init__(self,lightcurve,pars_dict,prior_dict,sampling_arguments,sampling_method):

        """


        Inputs:
        lightcurve         - light curve class which includes the full model (transit, systematics, etc.)
        pars_dict          - the dictionary of fitting parameters
        prior_dict         - dictionary with all the information needed to set up the priors (see fitting_input)
        sampling_arguments - dict, parameters needed for dynesty / emcee; e.g. live points, precision criterion, nsteps, nwalkers
        sampling_method    - str, either dynesty, emcee, LM



        Can return:
        - dynesty result
        - emcee result
        """

        self.lightcurve = lightcurve
        self.pars_dict = pars_dict
        self.prior_dict = prior_dict
        self.sampling_method = sampling_method
        self.sampling_arguments = sampling_arguments

        if sampling_method == 'dynesty':
            self.nDims = len(pars_dict)


        self.namelist = [k for k in self.pars_dict.keys() if self.pars_dict[k] is not None and not isinstance(self.pars_dict[k],float)]


    # -------------------- Dynesty methods -------------------- #
    def prior_setup(self, x):

        if sampling_method == 'dynesty':
            theta = [0] * self.nDims

            for i in range(self.nDims):
                if self.prior_dict['%s_prior'%self.pars_dict[i]] == 'N':
                    theta[i] = priors.GaussianPrior(self.prior_dict['%s_1'%self.pars_dict[i]], self.prior_dict['%s_2'%self.pars_dict[i]])(np.array(x[i]))
                elif self.prior_dict['%s_prior'%pars_dict[i]] == 'U':
                    theta[i] = priors.UniformPrior(self.prior_dict['%s_1'%self.pars_dict[i]], self.prior_dict['%s_2'%self.pars_dict[i]])(np.array(x[i]))

            return theta



    def loglikelihood_dynesty(self,theta):
        self.lightcurve.update_model(theta)
        noise = lightcurve.return_flux_err()

        residuals = self.lightcurve.calc_residuals()

        N = len(noise)
        logL = -N/2. *  np.log(2*np.pi)
        logL += - np.nansum(np.log(noise)) - np.nansum(residuals**2 / (2 * noise**2))

        return logL


    def run_dynesty(self):
        live_points = self.sampling_arguments['live_points']
        precision_criterion = self.sampling_arguments['precision_criterion']
        sampler = dynesty.NestedSampler(self.loglikelihood_dynesty, self.prior_setup, self.nDims,nlive=live_points*self.nDims, bootstrap=0)#,sample='rslice')
        sampler.run_nested(dlogz=precision_criterion, print_progress=True)
        results = sampler.results
        return results


    # -------------------- EMCEE methods -------------------- #
    def logprior_emcee(self, theta):
        """Compute log prior based on self.prior_dict."""
        lp = 0
        for i, pname in enumerate(self.namelist):
            prior_type = self.prior_dict[f'{pname}_prior']
            if prior_type == 'N':
                mu = self.prior_dict[f'{pname}_1']
                sigma = self.prior_dict[f'{pname}_2']
                lp += -0.5 * ((theta[i] - mu)/sigma)**2 - np.log(sigma*np.sqrt(2*np.pi))
            elif prior_type == 'U':
                lower = self.prior_dict[f'{pname}_1']
                upper = self.prior_dict[f'{pname}_2']
                if not (lower <= theta[i] <= upper):
                    return -np.inf
        return lp

    def loglikelihood_emcee(self, theta):
        """Compute log likelihood using the lightcurve residuals and errors."""
        self.lightcurve.update_model(theta)
        residuals = self.lightcurve.calc_residuals()
        noise = self.lightcurve.get_noise()  # TO BE IMPLEMENTED -- this is Evie's noise inflation term
        N = len(residuals)
        logL = -N/2. * np.log(2*np.pi) - np.sum(np.log(noise)) - np.sum(residuals**2 / (2*noise**2))
        return logL

    def logprobability_emcee(self, theta):
        """Full log-probability for emcee: lnprior + lnlike."""
        lp = self.logprior_emcee(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglikelihood_emcee(theta)


    def _advance_chain(self, sampler, p0, nsteps, burn, save_chain, bin_number):
        """Internal method to advance the emcee sampler chain."""
        for i, (pos, prob, state) in enumerate(sampler.sample(p0, iterations=nsteps, store=True)):
            # Optional: progress bar
            if not burn and save_chain:
                with open(f'prod_chain_wb{str(bin_number+1).zfill(4)}.txt', 'a') as f:
                    for k in range(pos.shape[0]):
                        thisPos = pos[k]
                        thisProb = prob[k]
                        f.write(f"{k} {' '.join(map(str,thisPos))} {thisProb}\n")
        return pos, np.max(prob)

    def run_emcee(self, burn=False, save_chain=True, bin_number=0):
        """Run emcee MCMC sampling."""
        nsteps = self.sampling_arguments['nsteps']
        nwalk = self.sampling_arguments['nwalkers']
        npars = len(self.namelist)
        nwalk_total = nwalk * npars

        # Scatter walkers around starting parameters
        starting_values = np.array([self.pars_dict[k] for k in self.namelist])
        if burn:
            p0 = emcee.utils.sample_ball(starting_values, 1e-3*starting_values, size=nwalk_total)
        else:
            p0 = [starting_values + 1e-8*np.random.randn(npars) for j in range(nwalk_total)]

        # Initialize sampler
        sampler = emcee.EnsembleSampler(nwalk_total, npars, self.logprobability_emcee)

        # Advance chain
        self._advance_chain(sampler, p0, nsteps, burn, save_chain, bin_number)

        # Flatten and get samples
        samples = sampler.get_chain(discard=int(nsteps/4), thin=10, flat=True)
        return samples


    # -------------------- Statistical Evaluation Methods -------------------- #
    def chisq(self, theta, time, flux, flux_err, sys_model_inputs=None):
        self.lightcurve.update_model(theta)
        if self.lightcurve.GP_used:
            mu, _ = self.lightcurve.calc_gp_component(time, flux, flux_err, sys_model_inputs)
            resids = (flux - self.lightcurve.calc(time, sys_model_inputs) - mu) / flux_err
        else:
            resids = (flux - self.lightcurve.calc(time, sys_model_inputs)) / flux_err
        return np.sum(resids**2)

    def reducedChisq(self, theta, time, flux, flux_err, sys_model_inputs=None):
        return self.chisq(theta, time, flux, flux_err, sys_model_inputs) / (len(flux) - self.lightcurve.npars)

    def rms(self, theta, time, flux, flux_err=None, sys_model_inputs=None):
        self.lightcurve.update_model(theta)
        if self.lightcurve.GP_used:
            mu, _ = self.lightcurve.calc_gp_component(time, flux, flux_err, sys_model_inputs)
            resids = flux - self.lightcurve.calc(time, sys_model_inputs) - mu
        else:
            resids = flux - self.lightcurve.calc(time, sys_model_inputs)
        return np.sqrt(np.mean(resids**2))

    def BIC(self, theta, time, flux, flux_err, sys_model_inputs=None):
        # note we can use loglikelihood_emcee also for LM fit since the statistic is independent of the sampling method
        return self.lightcurve.npars * np.log(len(flux)) - 2 * self.loglikelihood_emcee(theta, flux, flux_err, time, sys_model_inputs)

    def AIC(self, theta, time, flux, flux_err, sys_model_inputs=None):
        return 2 * self.lightcurve.npars - 2 * self.loglikelihood_emcee(theta, flux, flux_err, time, sys_model_inputs)

    def red_noise_beta(self, theta, time, flux, flux_err, sys_model_inputs=None):
        # Get the RMS of the residuals using the existing function
        rms_val = self.rms(theta, time, flux, flux_err, sys_model_inputs)

        time_diff = np.diff(time).min() * 24 * 60  # in minutes
        max_points = int(np.round(30 / time_diff))
        bins = np.linspace(time[0], time[-1], int(len(time) / max_points))

        # Rebin residuals
        _, binned_y, _ = pu.rebin(bins, time, flux - self.lightcurve.calc(time, sys_model_inputs), flux_err, weighted=False)
        max_rms = np.sqrt(np.nanmean(binned_y**2))

        gaussian_white_noise = np.array([1, 1/np.sqrt(max_points)])
        offset = np.max(gaussian_white_noise) / rms_val
        gaussian_white_noise /= offset
        beta_factor = max(max_rms / gaussian_white_noise)

        return beta_factor

    # -------------------- Parameter Opimisation -------------------- #
def optimise_params(self, time, flux, flux_err, reset_starting_gp=False, contact1=None, contact4=None,
                    full_model=False, sys_priors=None, verbose=True, LM_fit=False):
    """
    Optimise model parameters using Nelder-Mead or Levenberg-Marquardt.
    Can optimise GP hyperparameters only, full transit model, or all together.
    """

    # Select data for GP-only optimisation if out-of-transit
    if not full_model and contact1 is not None:
        mask = np.ones_like(time, dtype=bool)
        mask[contact1:contact4] = False
        time_opt, flux_opt, flux_err_opt = time[mask], flux[mask], flux_err[mask]
        kern_inputs = [ki[mask] for ki in self.lightcurve.systematics_model_inputs]
        evaluated_model = self.calc(time)[mask]
    else:
        time_opt, flux_opt, flux_err_opt = time, flux, flux_err
        kern_inputs = self.lightcurve.systematics_model_inputs
        evaluated_model = self.calc(time)

    y = flux_opt - evaluated_model if not full_model else flux_opt
    error = flux_err_opt

    # Construct GP if used
    if self.lightcurve.GP_used:
        gp = self.lightcurve.construct_gp()
        if self.lightcurve.gp_ndim > 1:
            gp.compute(np.array(kern_inputs).T, error)
        else:
            gp.compute(kern_inputs[0], error)
        if verbose:
            print('Initial ln-likelihood =', gp.lnlikelihood(y))
            print('Initial GP hyperparams =', gp.get_parameter_vector())

    # Initial parameter vector
    p0 = gp.get_parameter_vector() if self.lightcurve.GP_used and not full_model else self.extract_model_values()

    # Build bounds systematically
    bnds = self.build_bounds(full_model=full_model)

    # Verbosity
    disp = verbose and not LM_fit

    # Run optimisation
    if not LM_fit:
        results = optimize.minimize(
            nll, p0, args=(self, y, True, time_opt, flux_err_opt, sys_priors, False),
            method='Nelder-Mead', bounds=tuple(bnds), options=dict(maxiter=1e4, disp=disp)
        )
        self.lightcurve.update_model(results.x)
        return self, results.x
    else:
        results = optimize.least_squares(
            nll, p0, args=(self, y, True, time_opt, flux_err_opt, sys_priors, False, True),
            method='lm'
        )
        self.lightcurve.update_model(results.x)
        try:
            J = results.jac
            cov = np.linalg.inv(J.T.dot(J)) * self.reducedChisq(time, flux, flux_err)
            uncertainties = np.sqrt(np.diagonal(cov))
        except:
            print("Unable to estimate uncertainties from covariance matrix")
            uncertainties = np.zeros_like(results.x)
        return self, results.x, uncertainties


    def build_bounds(self, full_model=False):
        """
        Build parameter bounds using prior definitions and GP/kernel settings.
        Returns a list of (min, max) tuples in the order of self.namelist.
        """
        bnds = []

        for name in self.namelist:

            # Use prior definitions if available
            prior_type = self.prior_dict.get(f'{name}_prior', None)
            if prior_type == 'U':
                bnds.append((self.prior_dict[f'{name}_1'], self.prior_dict[f'{name}_2']))
            elif prior_type == 'N':
                # Optional: ±3σ around mean as bounds
                mu, sigma = self.prior_dict[f'{name}_1'], self.prior_dict[f'{name}_2']
                bnds.append((mu - 3*sigma, mu + 3*sigma))
            else:
                # Fallback: ±10% around current value
                curr = self.pars[name].currVal
                bnds.append((curr*0.9, curr*1.1))

        # GP bounds
        if self.lightcurve.GP_used:
            if self.lightcurve.wn_kernel:
                bnds.append((np.log((self.lightcurve.kernel_priors['min_WN_sigma'])**2),
                             np.log((self.lightcurve.kernel_priors['max_WN_sigma'])**2)))

            for i in range(self.gp_ndim):
                for key in [f'min_lniL_{i+1}', f'max_lniL_{i+1}']:
                    if key in self.lightcurve.kernel_priors:
                        bnds.append((self.lightcurve.kernel_priors[key], self.lightcurve.kernel_priors[key]))

            # GP amplitude
            if 'min_A' in self.lightcurve.kernel_priors and 'max_A' in self.lightcurve.kernel_priors:
                bnds.append((self.lightcurve.kernel_priors['min_A'], self.lightcurve.kernel_priors['max_A']))

        return bnds
