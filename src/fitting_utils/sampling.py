#### Author of this code: Eva-Maria Ahrer, adapted from Tiberius TransitGPPM model (author: J. Kirk)

import numpy as np

# from fitting_utils import lightcurve

import dynesty
import emcee

import sys
import pickle

from scipy import optimize,stats
# import matplotlib.pyplot as plt

# from fitting_utils import parametric_fitting_functions as pf
from fitting_utils import plotting_utils as pu
from fitting_utils import priors

class Sampling(object):
    def __init__(self,lightcurve,sampling_arguments,sampling_method):

        """


        Inputs:
        lightcurve         - light curve class which includes the full model (transit, systematics, etc.)
        sampling_arguments - dict, parameters needed for dynesty / emcee; e.g. live points, precision criterion, nsteps, nwalkers
        sampling_method    - str, either dynesty, emcee, LM



        Can return:
        - dynesty result
        - emcee result
        """

        self.lightcurve = lightcurve
        self.param_dict = self.lightcurve.param_dict
        self.param_list_free = self.lightcurve.param_list_free
        self.prior_dict = self.lightcurve.prior_dict
        self.sampling_method = sampling_method
        self.sampling_arguments = sampling_arguments

        if self.sampling_method == 'dynesty':
            self.nDims = len( self.param_list_free)

    # -------------------- Dynesty methods -------------------- #
    def prior_setup(self, x):
        if self.sampling_method == 'dynesty':
            theta = [0] * self.nDims

            for i in range(self.nDims):
                if self.prior_dict['%s_prior'%self.param_list_free[i]] == 'N':
                    theta[i] = priors.GaussianPrior(self.prior_dict['%s_1'%self.param_list_free[i]], self.prior_dict['%s_2'%self.param_list_free[i]])(np.array(x[i]))
                elif self.prior_dict['%s_prior'%self.param_list_free[i]] == 'U':
                    theta[i] = priors.UniformPrior(self.prior_dict['%s_1'%self.param_list_free[i]], self.prior_dict['%s_2'%self.param_list_free[i]])(np.array(x[i]))
            return theta



    def loglikelihood_dynesty(self,theta):
        self.lightcurve.update_model(theta)
        noise = self.lightcurve.return_flux_err()

        if self.lightcurve.GP_used:
            model_calc = self.lightcurve.calc(with_GP=False)
            logL = self.lightcurve.GP_model.lnlike(model_calc,noise)
            return logL

        else:
            residuals = self.lightcurve.calc_residuals()

            N = len(noise)
            logL = -N/2. *  np.log(2*np.pi)
            logL += - np.nansum(np.log(noise)) - np.nansum(residuals**2 / (2 * noise**2))

            return logL


    def run_dynesty(self):
        live_points = self.sampling_arguments['nlive_pdim']
        precision_criterion = self.sampling_arguments['precision_crit']
        sampler = dynesty.NestedSampler(self.loglikelihood_dynesty, self.prior_setup, self.nDims,nlive=live_points*self.nDims, bootstrap=0) #,sample='rslice')
        sampler.run_nested(dlogz=precision_criterion, print_progress=True)
        results = sampler.results
        return results


    # -------------------- EMCEE methods -------------------- #
    def logprior_emcee(self, theta):
        """Compute log prior based on self.prior_dict."""
        lp = 0
        for i, pname in enumerate(self.param_list_free):
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
        if theta is not None:
            self.lightcurve.update_model(theta)
        residuals = self.lightcurve.calc_residuals()
        flux_error = self.lightcurve.return_flux_err()
        N = len(residuals)
        logL = -N/2. * np.log(2*np.pi) - np.sum(np.log(flux_error)) - np.sum(residuals**2 / (2*flux_error**2))
        return logL

    def logprobability_emcee(self, theta):
        """Full log-probability for emcee: lnprior + lnlike."""
        lp = self.logprior_emcee(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglikelihood_emcee(theta)


    def advance_chain(self,sampler,p0,nsteps,burn,save_chain,wavelength_bin):
        """The function that advances the emcee sampler chain with a progress bar

        Inputs:
        sampler - the emcee sampler, intitiated in run_emcee
        p0 - the array of (starting) parameter nvalues
        nsteps - the number of steps to advance the chain over
        burn - is this a burn chain? If so, don't save to file
        save_chain - True/False, do we want to save the chain to file?
        wavelength_bin - the number of the wavelength bin we're fitting, so that we can save the output correctly

        Returns:
        sampler - the inputted emcee sampler advanced by nsteps"""

        width = 100 # for progress bar
        highest_prob = 0
        print('Progress:') # for progress bar
        for i,(pos, prob, state) in enumerate(sampler.sample(p0,iterations=nsteps,store=True)):
            n = int((width+1) * float(i) / nsteps)
            sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n))) # for progress bar

            if np.max(prob) > highest_prob:
                highest_prob_pars = pos[np.argmax(prob)]
                highest_prob = np.max(prob)

            if not burn and save_chain:
                f = open('prod_chain_wb%s.txt'%(str(wavelength_bin+1).zfill(4)),'a')
            for k in range(pos.shape[0]):
                # loop over all walkers and append to file
                thisPos = pos[k]
                thisProb = prob[k]
                if not burn and save_chain: # only save production chain to file and only file steps otherwise these files are huge!
                    if nsteps > 500 and i > nsteps/2.:
                        f.write("{0:4d} {1:s} {2:f}\n".format(k," ".join(map(str,thisPos)),thisProb ))
                    if nsteps < 500 and i > nsteps - 100:
                        f.write("{0:4d} {1:s} {2:f}\n".format(k," ".join(map(str,thisPos)),thisProb ))
            if not burn and save_chain:
                f.close()

        return sampler, highest_prob_pars, highest_prob

    def run_emcee(self, burn=False, save_chain=True, wavelength_bin=0):

        """Run emcee MCMC sampling."""
        nsteps = self.sampling_arguments['nsteps']
        nwalkers = self.sampling_arguments['nwalkers']
        nthreads = self.sampling_arguments['nthreads']
        npars = len(self.param_list_free)
        namelist = self.param_list_free
        nwalkers_total = nwalkers * npars

        # if wavelength_bin > 0 and burn:
        #     diagnostic_tab = open('burn_statistics.txt','a')
        #
        # elif wavelength_bin > 0 and not burn:
        #     diagnostic_tab = open('prod_statistics.txt','a')
        #
        # else: # starting fresh
        #     if burn:
        #         diagnostic_tab = open('burn_statistics.txt','w')
        #     else:
        #         diagnostic_tab = open('prod_statistics.txt','w')
        #
        # diagnostic_tab.close()
        #
        # if burn:
        #     diagnostic_tab = open('burn_statistics.txt','a')
        # else:
        #     diagnostic_tab = open('prod_statistics.txt','a')

        # Scatter walkers around starting parameters
        starting_values = np.array([self.param_dict[k].currVal for k in self.param_list_free])
        if burn:
            p0 = emcee.utils.sample_ball(starting_values, 1e-3*starting_values, size=nwalkers_total)
        else:
            p0 = [starting_values + 1e-8*np.random.randn(npars) for j in range(nwalkers_total)]

        # Initialize sampler
        # sampler = emcee.EnsembleSampler(nwalk_total, npars, self.logprobability_emcee)
        # intiate emcee sampler object
        if npars > 1:
            sampler = emcee.EnsembleSampler(nwalkers_total,npars,self.logprobability_emcee,threads=nthreads)
        else: # from my own tests I find that for a single parameter, the acceptance fraction is too high. Increasing the stretch scale factor decreases the acceptance fraction to a more acceptable value. This is relevant for ingress/egress fitting for ingress/egress with just Rp/Rs
            sampler = emcee.EnsembleSampler(nwalkers_total,npars,self.logprobability_emcee,threads=nthreads,moves=emcee.moves.StretchMove(10))

        # run chains
        print('################')
        if burn:
            print("Running burn-in for bin %d..."%(wavelength_bin+1))
            # f = open('burn_chain_%d.txt'%(wavelength_bin+1),'w') # deciding to only save production chain

        else:
            print("Running production for bin %d..."%(wavelength_bin+1))
            if save_chain:
                f = open('prod_chain_wb%s.txt'%(str(wavelength_bin+1).zfill(4)),'w')
                f.close()

        if nsteps == "auto":
            not_converged = True
            chain_number = 1
            nsteps = 2000

            while not_converged:

                if chain_number == 1:
                    sampler, highest_prob_pars, highest_prob = self.advance_chain(sampler,p0,nsteps,burn,save_chain,wavelength_bin)
                else:
                    sampler, highest_prob_pars, highest_prob = self.advance_chain(sampler,sampler.get_last_sample(),nsteps,burn,save_chain,wavelength_bin)

                total_steps = chain_number*nsteps

                try:
                    auto_corr_time = np.round(total_steps/np.median(sampler.acor))

                    # ideal scenario, we're >= 50x the median autocorr time
                    if auto_corr_time >= 50: # taking DFM's estimate
                        not_converged = False
                        print("\n\nChains run for %d total steps"%(chain_number*nsteps))
                        nsteps = total_steps # updated nsteps for calculation of corner plots and parameter values later on

                    # not so good scenario but chains are getting long
                    elif auto_corr_time >= 20 and total_steps >= 10000:
                        print("\n\n After %d steps the number of steps is %dX the autocorrelation time, finishing chain"%(chain_number*nsteps,auto_corr_time))
                        nsteps = total_steps # updated nsteps for calculation of corner plots and parameter values later on
                        not_converged = False

                    # chains too long, probably won't converge now
                    elif total_steps >= 20000:
                        print("\n\n After %d steps the chains have not yet converged, exiting"%(chain_number*nsteps))
                        nsteps = total_steps
                        not_converged = False

                    else:
                        print("\n\n After %d steps the number of steps is %dX the autocorrelation time, running chain again"%(chain_number*nsteps,auto_corr_time))
                        chain_number += 1
                except:
                    if total_steps >= 20000:
                        print("\n\n After %d steps the chains have not yet converged, exiting"%(chain_number*nsteps))
                        nsteps = total_steps
                        not_converged = False
                    else:
                        print("\n\n After %d steps the chains have not yet converged, running chain again"%(chain_number*nsteps))
                        chain_number += 1
        else:
            sampler, highest_prob_pars, highest_prob = self.advance_chain(sampler,p0,nsteps,burn,save_chain,wavelength_bin)

        # save plots of chains
        pu.plot_chains(sampler,burn,wavelength_bin,npars,namelist)

        if nsteps >= 500:
            if burn:
                samples = sampler.chain[:, int(nsteps/2):, :].reshape((-1, ndim))
            else:
                samples = sampler.get_chain(discard=int(nsteps/4), thin=10, flat=True)
        else:
            samples = sampler.chain[:, -100:, :].reshape((-1, npars))

        print('\n')
        # generate median, upper and lower bounds
        med, up, low, mode = recover_quartiles_single(samples,namelist,bin_number=(wavelength_bin+1),verbose=True,save_result=True,burn=burn)

        if not burn and npars > 1:
            # generate and save corner plot
            samples_corner = samples
            pu.make_corner_plot(samples_corner,bin_number=(wavelength_bin+1),save_fig=True,namelist=namelist,parameter_modes=mode)

        self.lightcurve.update_model(med)

        # fitted_chi2 = self.chisq(med)
        # fitted_reducedChi2 = self.reducedChisq(med)
        # fitted_rms = self.rms(med)*1e6
        # fitted_lnlike = self.loglikelihood_emcee(med)
        # fitted_lnprob = self.logprobability_emcee(med)
        # fitted_BIC = self.BIC(med)
        # fitted_AIC = self.AIC(med)
        #
        # print("\n--- Using medians of posteriors ---")
        # print('chi2 = %f' % fitted_chi2)
        # print('Reduced chi2 = %f' % fitted_reducedChi2)
        # print('Lnlikelihood = %f' % fitted_lnlike)
        # print('Lnprobability = %f' % fitted_lnprob)
        # print('Residual RMS (ppm) = %f' % fitted_rms)
        # print('BIC = %f' % fitted_BIC)
        # print('AIC = %f' % fitted_AIC)
        #
        # diagnostic_tab.write("\n### Bin %d ###\n" % (wavelength_bin+1))
        # diagnostic_tab.write("\n--- Using medians of posteriors --- \n")
        # diagnostic_tab.write('Chi2 = %f \n' % fitted_chi2)
        # diagnostic_tab.write('Reduced chi2 = %f \n' % fitted_reducedChi2)
        # diagnostic_tab.write('Lnlikelihood = %f \n' % fitted_lnlike)
        # diagnostic_tab.write('Lnprobability = %f \n' % fitted_lnprob)
        # diagnostic_tab.write('Residual RMS (ppm) = %f \n' % fitted_rms)
        # diagnostic_tab.write('BIC = %f \n' % fitted_BIC)
        # diagnostic_tab.write('AIC = %f \n' % fitted_AIC)

        # mode_model = copy.deepcopy(self.lightcurve)
        # mode_model = self.lightcurve.update_model(mode)
        #
        # mode_chi2 = mode_model.chisq(x,y,e)
        # mode_reducedChi2 = mode_model.reducedChisq(x,y,e)
        # mode_rms = mode_model.rms(x,y,e)*1e6
        # mode_lnlike = mode_model.lnlike(x,y,e)
        # mode_lnprob = lnprob_emcee(mode,mode_model,x,y,e,None,sys_priors,typeII)
        # mode_BIC = mode_model.BIC(x,y,e)
        #
        # print("\n--- Using modes of posteriors ---")
        # print('chi2 = %f' % mode_chi2)
        # print('Reduced chi2 = %f' % mode_reducedChi2)
        # print('Lnlikelihood = %f' % mode_lnlike)
        # print('Lnprobability = %f' % mode_lnprob)
        # print('Residual RMS (ppm) = %f' % mode_rms)
        # print('BIC = %f' % mode_BIC)
        #
        # diagnostic_tab.write("\n--- Using modes of posteriors --- \n")
        # diagnostic_tab.write('Chi2 = %f \n' % mode_chi2)
        # diagnostic_tab.write('Reduced chi2 = %f \n' % mode_reducedChi2)
        # diagnostic_tab.write('Lnlikelihood = %f \n' % mode_lnlike)
        # diagnostic_tab.write('Lnprobability = %f \n' % mode_lnprob)
        # diagnostic_tab.write('Residual RMS (ppm) = %f \n' % mode_rms)
        # diagnostic_tab.write('BIC = %f \n' % mode_BIC)

        write_fit_diagnostics(self,wavelength_bin,emcee_fit=True,burn=burn,emcee_sampler=sampler,nsteps=nsteps)


        # try:
        #     print('\nAutocorrelation time for each parameter = ',np.round(sampler.acor).astype(int))
        #     # Alternatively something like: emcee.autocorr.integrated_time(sampler.chain, low=10, high=None, step=1, c=5, full_output=True,axis=0, fast=False)
        #     diagnostic_tab.write('\nAutocorrelation time for each parameter = ')
        #     for ac in np.round(sampler.acor).astype(int):
        #         diagnostic_tab.write('%d '%ac)
        #     diagnostic_tab.write('\n')
        #
        #     print('nsamples/median(autocorrelation time) = %d'%np.round(nsteps/np.median(sampler.acor)))
        #     diagnostic_tab.write('nsamples/median(autocorrelation time) = %d \n'%(np.round(nsteps/np.median(sampler.acor))))
        # except:
        #     print("\nAutocorrelation time can't be calculated - chains likely too short")
        #     diagnostic_tab.write("\nAutocorrelation time can't be calculated - chains likely too short \n")
        #
        # print('Acceptance fraction = %f'%(np.mean(sampler.acceptance_fraction)))
        #
        # diagnostic_tab.write('Acceptance fraction = %f \n'%(np.mean(sampler.acceptance_fraction)))
        # diagnostic_tab.write('Total steps = %d \n'%(nsteps))
        #
        # diagnostic_tab.close()

        if not burn:
            pickle.dump(self.lightcurve,open('fitted_lightcurve_model_wb%s.pickle'%(str(wavelength_bin+1).zfill(4)),'wb'))
            # try:
            #     pickle.dump(mode_model,open('parameter_modes_model_wb%s.pickle'%(str(wavelength_bin+1).zfill(4)),'wb'))
            # except:
            #     pass

        if burn:
            print("...burn-in complete for bin %d"%(wavelength_bin+1))
        else:
            print("...production complete for bin %d"%(wavelength_bin+1))

        sampler.reset()

        return self.lightcurve

    ### -------- Levenberg-Marquadt methods -------- ###
    def run_LM(self,wavelength_bin=0):
        """
        Run Levenberg-Marquardt optimization to minimize residuals
        """

        # Initial parameter vector
        theta0 = np.array([self.param_dict[p].currVal for p in self.param_list_free])

        # Define residual function
        def residuals(theta):

            self.lightcurve.update_model(theta)

            prior_val = self.logprior_emcee(theta)

            if not np.isfinite(prior_val):
                return np.ones_like(self.lightcurve.flux_array)*np.inf

            return self.lightcurve.calc_residuals()/self.lightcurve.flux_err

        # Run Levenberg-Marquardt fit
        result = optimize.least_squares(residuals, theta0, method='lm')

        # Update model with best-fit parameters
        self.lightcurve.update_model(result.x)

        # Estimate uncertainties from covariance matrix
        try:
            J = result.jac
            cov = np.linalg.inv(J.T.dot(J))*self.reducedChisq()
            uncertainties = np.sqrt(np.diag(cov))
        except:
            print("Unable to estimate uncertainties from Jacobian")
            uncertainties = np.zeros_like(result.x)

        write_fit_diagnostics(self,wavelength_bin,LM_fit=True)
        save_LM_results(self.lightcurve,result.x,uncertainties,wavelength_bin,verbose=True)

        return self.lightcurve


    def build_bounds(self, full_model=False):
        """
        Build parameter bounds using prior definitions and GP/kernel settings.
        Returns a list of (min, max) tuples in the order of self.param_list_free.
        """
        bnds = []

        for name in self.param_list_free:

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


    # -------------------- Statistical Evaluation Methods -------------------- #
    def chisq(self, theta=None):
        if theta is not None:
            self.lightcurve.update_model(theta)

        if self.lightcurve.GP_used:
            mu, _ = self.lightcurve.calc_gp_component()
            resids = (self.lightcurve.calc_residuals() - mu) / self.lightcurve.flux_err
        else:
            resids = self.lightcurve.calc_residuals() / self.lightcurve.flux_err

        return np.sum(resids**2)

    def reducedChisq(self, theta=None):
        return self.chisq(theta) / (len(self.lightcurve.flux_array) - self.lightcurve.npars)

    def rms(self, theta=None):
        if theta is not None:
            self.lightcurve.update_model(theta)

        if self.lightcurve.GP_used:
            mu, _ = self.lightcurve.calc_gp_component()
            resids = (self.lightcurve.calc_residuals() - mu)
        else:
            resids = self.lightcurve.calc_residuals()

        return np.sqrt(np.mean(resids**2))

    def BIC(self, theta=None):
        # note we can use loglikelihood_emcee also for LM fit since the statistic is independent of the sampling method
        return self.lightcurve.npars * np.log(len(self.lightcurve.flux_array)) - 2 * self.loglikelihood_emcee(theta)

    def AIC(self, theta=None):
        return 2 * self.lightcurve.npars - 2 * self.loglikelihood_emcee(theta)

    def red_noise_beta(self, theta=None):
        # Get the RMS of the residuals using the existing function
        rms_val = self.rms(theta)

        time_diff = np.diff(self.lightcurve.time_array).min() * 24 * 60  # in minutes
        max_points = int(np.round(30 / time_diff))
        bins = np.linspace(self.lightcurve.time_array[0],
                           self.lightcurve.time_array[-1],
                           int(len(self.lightcurve.time_array) / max_points))

        # Rebin residuals
        _, binned_y, _ = pu.rebin(bins,
                                  self.lightcurve.time_array,
                                  self.lightcurve.flux_array - self.lightcurve.calc(),
                                  self.lightcurve.flux_err, weighted=False)

        max_rms = np.sqrt(np.nanmean(binned_y**2))

        gaussian_white_noise = np.array([1, 1/np.sqrt(max_points)])
        offset = np.max(gaussian_white_noise) / rms_val
        gaussian_white_noise /= offset
        beta_factor = max(max_rms / gaussian_white_noise)

        return beta_factor



def save_LM_results(fitted_lightcurve,param_medians,param_uncertainties,bin_number,verbose=True):
    """Function to save the results from an LM fit to a best_fit_parameters.dat and LM_statistics.dat tables equivalent to emcee results.

    Inputs:
    fitted_lightcurve - the best fitting, resulting, Lightcurvemodel object
    param_medians - the best fitting parameters
    param_uncertainties - the 1 sigma uncertainties on the parameters
    bin_number - the bin number (correcting for Python indexing, i.e. adding 1)
    verbose - True/False - print the best-fitting results to terminal?

    Returns:
    Nothing but saving the results to best_fit_parameters.dat and LM_statistics.txt"""

    ndim = len(param_medians)
    namelist = fitted_lightcurve.param_list_free

    if bin_number == 0:
        new_tab = open('best_fit_parameters.txt','w')
    else:
        new_tab = open('best_fit_parameters.txt','a')

    print('\nSaving best fit parameters to table...\n')

    for i in range(ndim):
        # note, we repeat the uncertainties column twice here even though there is only one uncertainty value, this is so the other functions can better handle this table
        new_tab.write("%s_%d = %f + %f - %f \n"%(namelist[i].replace('$','').replace("\\",''),bin_number+1,param_medians[i],param_uncertainties[i],param_uncertainties[i]))

        if verbose:
            print("%s_%d = %f +/- %f"%(namelist[i].replace('$','').replace("\\",''),bin_number+1,param_medians[i],param_uncertainties[i]))

    new_tab.write('#------------------ \n')
    new_tab.close()

    return


def write_fit_diagnostics(sampling_model,wavelength_bin,emcee_fit=False,burn=False,LM_fit=False,emcee_sampler=None,nsteps=None):

    if wavelength_bin == 0:
        read_mode = 'a'
    else:
        read_mode = 'w'

    if emcee_fit:

        if burn:
            diagnostic_tab = open('burn_statistics.txt',read_mode)
        else:
            diagnostic_tab = open('prod_statistics.txt',read_mode)

    if LM_fit:

        diagnostic_tab = open('LM_statistics.txt',read_mode)

    fitted_chi2 = sampling_model.chisq()
    fitted_reducedChi2 = sampling_model.reducedChisq()
    fitted_rms = sampling_model.rms()*1e6
    fitted_BIC = sampling_model.BIC()
    fitted_AIC = sampling_model.AIC()

    print('\nCalculating statistics for best fit...')
    print('chi2 = %.3f' % fitted_chi2)
    print('Reduced chi2 = %.3f' % fitted_reducedChi2)
    print('Residual RMS (ppm) = %d' % fitted_rms)
    print('BIC = %f' % fitted_BIC)
    print('AIC = %f' % fitted_AIC)

    diagnostic_tab.write("\nBin %d \n" % (wavelength_bin))
    diagnostic_tab.write('Chi2 = %.3f \n' % fitted_chi2)
    diagnostic_tab.write('Reduced chi2 = %.3f \n' % fitted_reducedChi2)
    diagnostic_tab.write('Residual RMS (ppm) = %d \n' % fitted_rms)
    diagnostic_tab.write('BIC = %f \n' % fitted_BIC)
    diagnostic_tab.write('AIC = %f \n' % fitted_AIC)

    if emcee_sampler is not None:
        try:
            print('\nAutocorrelation time for each parameter = ',np.round(emcee_sampler.acor).astype(int))
            # Alternatively something like: emcee.autocorr.integrated_time(sampler.chain, low=10, high=None, step=1, c=5, full_output=True,axis=0, fast=False)
            diagnostic_tab.write('\nAutocorrelation time for each parameter = ')
            for ac in np.round(emcee_sampler.acor).astype(int):
                diagnostic_tab.write('%d '%ac)
            diagnostic_tab.write('\n')

            print('nsamples/median(autocorrelation time) = %d'%np.round(nsteps/np.median(emcee_sampler.acor)))
            diagnostic_tab.write('nsamples/median(autocorrelation time) = %d \n'%(np.round(nsteps/np.median(emcee_sampler.acor))))
        except:
            print("\nAutocorrelation time can't be calculated - chains likely too short")
            diagnostic_tab.write("\nAutocorrelation time can't be calculated - chains likely too short \n")

        print('Acceptance fraction = %f'%(np.mean(emcee_sampler.acceptance_fraction)))

        diagnostic_tab.write('Acceptance fraction = %f \n'%(np.mean(emcee_sampler.acceptance_fraction)))
        diagnostic_tab.write('Total steps = %d \n'%(nsteps))

    diagnostic_tab.write('#------------------ \n')
    diagnostic_tab.close()



def nll(p,model,y,full_model=False,x=None,e=None,sys_priors=None,typeII=False,LM_fit=False):
    """Function to calculate the negative ln-likelihood of the george.GP object. This is a neccessary step for optimising the GP hyperparameters and is following the procedure given in the george documentation.

    Inputs:
    p - the parameter values of the model
    model - the TransitModelGPPM object
    y - the y (flux) data points at which to evaluate the model
    full_model - True/False - are we fitting the full model (including mean/transit model params) or just the GP hyperparams?
    x - the array of times, needed for full_model=True
    e - the array of flux uncertainties, needed for full_model=True
    sys_priors - the priors on the system parameters for full_model=True, default=None (no priors)
    typeII - is this a typeII maximum likelihood fit?
    LM_fit - True/False, are we using a Levenberg-Marquardt fit? If so, we need to return the weighted residuals, not likelihood/chi2

    Returns:
    if full_model=False:
        the negative ln-likelihood evaluated by george
    if full_model and not LM_fit:
        the chi2 of the model fit
    if full_model and LM_fit:
        the weighted residuals of the model fit"""

    if full_model:
        for i in range(model.npars):
            model[i] = p[i]
        if np.isfinite(model.lnprior(sys_priors)):
            if LM_fit:
                if model.GP_used: # note: LM fit is not working 100% with GPs
                    mu, std = model.calc_gp_component(x,y,e)
                    residuals = (y - model.calc(x) - mu)
                else:
                    residuals = (y-model.calc(x))/e
                return residuals
            else:
                chi2 = model.chisq(x,y,e)
                return chi2
        else:
            if LM_fit:
                return np.ones_like(y)*np.inf
            return np.inf

    else:
        model.set_parameter_vector(p)
        ll = model.lnlikelihood(y, quiet=True)

        # The scipy optimizer doesn't play well with infinities.
        return -ll if np.isfinite(ll) else 1e25

def grad_nll(p,gp,y):
    """A function to compute the gradient of the objective function/ln-likelihood. This is a neccessary step for optimising the GP hyperparameters and is following the procedure given in the george documentation.

    Inputs:
    p - the GP hyperparameter values
    gp - the GP object
    y - the y (flux) data points at which to evaluate the GP

    Returns:
    the negative grad_lnlikelihood evaluated by george"""

    gp.set_parameter_vector(p)
    return -gp.grad_lnlikelihood(y, quiet=True)


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


# def update_prior_file(input_prior_file,lightcurve_model,best_fit_params):
#
#     import shutil
#
#     shutil.copyfile(input_prior_file, new_prior_file_name)
#
#     new_prior_file_name = input_prior_file.split(".txt")[0] + "_wb.txt"
#
#     new_prior_file = open(new_prior_file_name,"a")
#
#     fitted_param_names = np.loadtxt("best_fit_parameters.txt",usecols=0,dtype="str").replace("_1","")
#     param_medians,param_uncertainties =
#
#     for i,line in enumerate(new_prior_file):
#
#         if line[1] == "free":
#             line[2] = lightcurve_model[]








def recover_quartiles_single(samples,namelist,bin_number,verbose=True,save_result=False,burn=False):
    """
    Function that calculates the 16th, 50th and 84th percentiles from a numpy array / emcee chain and saves these to a table.

    Inputs:
    samples - the samples/chains from emcee
    namelist - the names of the parameters that were fit - needed for printing and saving to file
    bin_number - the number of the wavelength bin we're considering. Necessary for printing and saving to file.
    verbose - True/False: do we want to print the results to screen?
    save_result - True/False: do we want to save the results to a table?
    burn - True/False: is this a burn-in chain? If so, save to burn_parameters.dat, else save to best_fit_parameters.dat

    Returns:
    (median, upper bound, lower bound) with shape (nparameters,3)
    """
    lower = []
    median = []
    upper = []
    mode = []

    ndim = np.shape(samples)[1] # this is equal to the number of params

    length_nl = len(namelist)

    # generate dictionary of how many decimal places we want to round each parameter to before calculating the mode of the rounded distribution
    namelist_decimal_places = {"t0":6,"per":6,"rp":6,"a":2,"inc":2,"ecc":3,"w":2,\
                               "u1":2,"u2":2,"u3":2,"u4":2,"f":4,"s":3,"A":3,"step1":3,"step2":3,"breakpoint":0,"lniL":2,
                               "infl":3}

    # now pad the dictionray with the systematics coefficients which could be a large number of parameters (although much less than the 100 allowed for below)
    for i in range(100):
        namelist_decimal_places["r%s"%(i)] = 2
        namelist_decimal_places["c%s"%(i)] = 6
        namelist_decimal_places["lniL_%s"%(i)] = 2

    if save_result:
        if bin_number == 1:
            if burn:
                new_tab = open('burn_parameters.txt','w')
                # new_tab_2 = open('parameter_modes_burn.txt','w')
            else:
                new_tab = open('best_fit_parameters.txt','w')
                new_tab_2 = open('parameter_modes_prod.txt','w')
        else:
            if burn:
                new_tab = open('burn_parameters.txt','a')
                # new_tab_2 = open('parameter_modes_burn.txt','a')
            else:
                new_tab = open('best_fit_parameters.txt','a')
                new_tab_2 = open('parameter_modes_prod.txt','a')

    for i in range(ndim):
        par = samples[:,i]
        lolim,best,uplim = np.percentile(par,[16,50,84])
        lower.append(lolim)
        median.append(best)
        upper.append(uplim)

        # calculate mode of rounded sample array
        key = namelist[i].replace('$','').replace("\\",'')
        key = key.split("_")[0]
        rounded_par = np.round(par,namelist_decimal_places[key])
        mode_value, mode_count = stats.mode(rounded_par,keepdims=True)
        mode.append(mode_value[0])

        if save_result:
            new_tab.write("%s_%d = %f + %f - %f \n"%(namelist[i].replace('$','').replace("\\",''),bin_number,best,uplim-best,best-lolim))
            if not burn:
                new_tab_2.write("%s_%d = %f (%d counts = %d%%) \n"%(namelist[i].replace('$','').replace("\\",''),bin_number,mode_value[0],mode_count[0],100*mode_count[0]/len(par)))

        if verbose:
            print("%s_%d = %f + %f - %f"%(namelist[i].replace('$','').replace("\\",''),bin_number,best,uplim-best,best-lolim))
            if not burn:
                print("%s_%d (mode of posterior) = %f (%d counts = %.2f%%) \n"%(namelist[i].replace('$','').replace("\\",''),bin_number,mode_value[0],mode_count[0],100*mode_count[0]/len(par)))

    if save_result:
        print('\nSaving best fit parameters to table...\n')
        new_tab.write('#------------------ \n')
        new_tab.close()

        if not burn:
            new_tab_2.write('#------------------ \n')
            new_tab_2.close()

    return np.array(median),np.array(upper),np.array(lower),np.array(mode)
