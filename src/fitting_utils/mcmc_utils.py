#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import pickle
import matplotlib.pyplot as plt
import emcee
from corner import corner,overplot_lines
import sys
from fitting_utils import TransitModelGPPM as tmgp
import copy
from scipy import stats

def parseParam(parString):
    """Function to convert input_dicts / parameters saved as 'mean +err -err' to floats with upper and lower errors. Currently used by plotting_utils.

    Inputs:
    parString - the string of 'mean +err -err'

    Returns:
    val - the mean as a float
    upper_error - the upper error as a float
    lower_error - the lower error as a float

    """

    fields = parString.split()
    val = float(fields[0])
    upper_error   = float(fields[2])
    lower_error   = float(fields[4])

    return val, upper_error, lower_error


def save_LM_results(best_pars,uncertainties,namelist,bin_number,best_model,time,flux,flux_err,verbose=True):
    """Function to save the results from an LM fit to a best_fit_parameters.dat and LM_statistics.dat tables equivalent to emcee results.

    Inputs:
    best_pars - the best fitting parameters
    uncertainties - the 1 sigma uncertainties on the parameters
    namelist - the names of the values
    bin_number - the bin number (correcting for Python indexing, i.e. adding 1)
    best_model - the best fitting, resulting, TransitModelGPPM object
    time - the time array
    flux - the flux array
    flux_err - the flux error array
    verbose - True/False - print the best-fitting results to terminal?

    Returns:
    Nothing but saving the results to best_fit_parameters.dat and LM_statistics.txt"""

    ndim = len(best_pars)

    if bin_number == 1:
        new_tab = open('best_fit_parameters.txt','w')
        diagnostic_tab = open('LM_statistics.txt','w')
    else:
        new_tab = open('best_fit_parameters.txt','a')
        diagnostic_tab = open('LM_statistics.txt','a')

    fitted_chi2 = best_model.chisq(time,flux,flux_err)
    fitted_reducedChi2 = best_model.reducedChisq(time,flux,flux_err)
    fitted_rms = best_model.rms(time,flux,flux_err)*1e6
    fitted_BIC = best_model.BIC(time,flux,flux_err)
    fitted_AIC = best_model.AIC(time,flux,flux_err)

    print('\nCalculating statistics for best fit...')
    print('chi2 = %.3f' % fitted_chi2)
    print('Reduced chi2 = %.3f' % fitted_reducedChi2)
    print('Residual RMS (ppm) = %d' % fitted_rms)
    print('BIC = %f' % fitted_BIC)
    print('AIC = %f' % fitted_AIC)

    diagnostic_tab.write("\nBin %d \n" % (bin_number))
    diagnostic_tab.write('Chi2 = %.3f \n' % fitted_chi2)
    diagnostic_tab.write('Reduced chi2 = %.3f \n' % fitted_reducedChi2)
    diagnostic_tab.write('Residual RMS (ppm) = %d \n' % fitted_rms)
    diagnostic_tab.write('BIC = %f \n' % fitted_BIC)
    diagnostic_tab.write('AIC = %f \n' % fitted_AIC)
    diagnostic_tab.write('#------------------ \n')
    diagnostic_tab.close()

    print('\nSaving best fit parameters to table...\n')

    for i in range(ndim):
        # note, we repeat the uncertainties column twice here even though there is only one uncertainty value, this is so the other functions can better handle this table
        new_tab.write("%s_%d = %f + %f - %f \n"%(namelist[i].replace('$','').replace("\\",''),bin_number,best_pars[i],uncertainties[i],uncertainties[i]))

        if verbose:
            print("%s_%d = %f +/- %f"%(namelist[i].replace('$','').replace("\\",''),bin_number,best_pars[i],uncertainties[i]))

    new_tab.write('#------------------ \n')
    new_tab.close()


    return


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
    namelist_decimal_places = {"t0":6,"period":6,"k":6,"aRs":2,"inc":2,"ecc":3,"omega":2,\
                               "u1":2,"u2":2,"u3":2,"u4":2,"f":4,"s":3,"A":3,"step1":3,"step2":3,"breakpoint":0}

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


def make_corner_plot(sample_chains,bin_number,namelist,parameter_modes,save_fig=False,title=None):
    """Use DFM's corner package to make a corner plot of the emcee chains.

    Input:
    sample_chains - the emcee chains
    bin_number - the wavelength bin number, needed for saving the plot to file
    namelist - a list of parameter names corresponding to the chain
    parameter_modes - the modes of the parameter distributions
    save_fig - True/False: do we want to save the figure to file?
    title - set to a string if wanting to define where the plot is saved to, otherwise default is used. Default=None.

    Returns:
    Nothing - just plots the figure
    """

    print('Generating corner plot...')
    ndim = np.shape(sample_chains)[1]

    fig = corner(sample_chains,labels=namelist,quantiles=[0.16, 0.5, 0.84],verbose=False,show_titles=True)
    overplot_lines(fig, parameter_modes)

    if save_fig:
        if title is not None:
            fig.savefig(title)
        else:
            fig.savefig('cornerplot_wb%s.png'%(str(bin_number).zfill(4)))
        plt.close()
    else:
        plt.show()


def lnprob_emcee(pars,model,x,y,e,sys_model_inputs=None,sys_priors=None,typeII=False):
    """Calculate the lnprobability of the model using the function inbuilt into the transit model classes.

    Inputs:
    pars - the new set of parameters that we're updating the model with
    model - the TransitModel or TransitModelGP object
    x - array of times
    y - array of fluxes
    e - array of errors on fluxes
    sys_model_inputs - the inputs of the systematics model. Can be left blank if these are unchanged since the model was initialised.
    sys_priors - array of standard deviations on Rp/Rs, a/Rs and inclination. Only used for white light fits. Default = None (no prior used).
    typeII - True/False: are we performing typeII maximum likelihood - only used by TransitModelGP

    Returns:
    evaluated log likelihood (float)
    """

    # we need to update the model we're using to use pars as submitted by MCMC

    if typeII:
        for i in range(len(pars)):
            model[i] = pars[i]
    else:
        for i in range(model.npars):
            model[i] = pars[i]

    model_lnprob = model.lnprob(x,y,e,sys_model_inputs,sys_priors,typeII)

    return model_lnprob


def chi2(pars,model,x,y,e):
    """Calculate the chi2 using the class in-built chi2 calculator.

    Inputs:
    pars - the new parameters that we're updating the model with
    model - the TransitModel or TransitModelGP object
    x - array of times
    y - array of fluxes
    e - array of errors on fluxes

    Returns:
    evaluated chi2 (float)"""

    # we need to update the model we're using to use pars as submitted by MCMC
    for i in range(model.npars):
            model[i] = pars[i]

    return model.chisq(x,y,e)


def run_emcee(starting_model,x,y,e,nwalk,nsteps,nthreads,burn=False,wavelength_bin=0,sys_priors=None,typeII=False,save_chain=True):

    """Run the MCMC.

    Inputs:
    starting_model - the TransitModel or TransitModelGP object
    x - array of times
    y - array of fluxes
    e - array of errors on fluxes
    nwalk - the number of emcee walkers
    nsteps - the number of steps in the MCMC chain. Set to 'auto' if wanting to use the autocorrelation time to determine when the chains have burned in
    nthreads - the number of threads over which to run the MCMC
    burn - True/False: is this a burn in run or production run? This changes how the output is saved.
    wavelength_bin - the number of the wavelength bin the fit is running to. Needed for accurate saving.
    sys_priors - array of standard deviations on Rp/Rs, a/Rs and inclination. Only used for white light fits. Default = None (no prior used).
    typeII - True/False: are we performing typeII maximum likelihood - only used by TransitModelGP

    Returns:
    (fitted median parameter values, fitted upper parameter bounds, fitted lower parameter bounds, fitted TransitModel/TransitModelGP object)

    """


    if wavelength_bin > 0 and burn:
        diagnostic_tab = open('burn_statistics.txt','a')

    elif wavelength_bin > 0 and not burn:
        diagnostic_tab = open('prod_statistics.txt','a')

    else: # starting fresh
        if burn:
            diagnostic_tab = open('burn_statistics.txt','w')
        else:
            diagnostic_tab = open('prod_statistics.txt','w')

    diagnostic_tab.close()

    if burn:
        diagnostic_tab = open('burn_statistics.txt','a')
    else:
        diagnostic_tab = open('prod_statistics.txt','a')

    starting_model_values = tmgp.extract_model_values(starting_model,typeII)

    npars = ndim = len(starting_model_values)
    nwalkers = nwalk*npars

    # scatter walkers around starting values
    if burn:
        p0 = emcee.utils.sample_ball(starting_model_values,1e-3*starting_model_values,size=nwalkers)
    else:
        p0 = [np.array(starting_model_values) + 1e-8 * np.random.randn(ndim) for j in range(nwalkers)]

    # intiate emcee sampler object
    if ndim > 1:
        sampler = emcee.EnsembleSampler(nwalkers,npars,lnprob_emcee,args=[starting_model,x,y,e,None,sys_priors,typeII],threads=nthreads)
    else: # from my own tests I find that for a single parameter, the acceptance fraction is too high. Increasing the stretch scale factor decreases the acceptance fraction to a more acceptable value
        sampler = emcee.EnsembleSampler(nwalkers,npars,lnprob_emcee,args=[starting_model,x,y,e,None,sys_priors,typeII],threads=nthreads,moves=emcee.moves.StretchMove(10))

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
                sampler, highest_prob_pars, highest_prob = advance_chain(sampler,p0,nsteps,burn,save_chain,wavelength_bin)
            else:
                sampler, highest_prob_pars, highest_prob = advance_chain(sampler,sampler.get_last_sample(),nsteps,burn,save_chain,wavelength_bin)

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
        sampler, highest_prob_pars, highest_prob = advance_chain(sampler,p0,nsteps,burn,save_chain,wavelength_bin)

    # save plots of chains
    if ndim > 1:
        fig,axes = plt.subplots(ndim,1,sharex=True,figsize=(8,12))
        for j in range(ndim):
            axes[j].plot(sampler.chain[:, :, j].T, color="k", alpha=0.4)
            axes[j].set_ylabel(starting_model.namelist[j],fontsize=20)
            axes[j].set_xlabel("step number")
    else:
        fig,axes = plt.subplots(1,1,figsize=(6,3))
        axes.plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
        axes.set_ylabel(starting_model.namelist[0],fontsize=20)
        axes.set_xlabel("step number")

    fig.tight_layout(h_pad=0.0)
    if burn:
        fig.savefig('burn_chain_wb%s.png'%(str(wavelength_bin+1).zfill(4)))
    else:
        fig.savefig('prod_chain_wb%s.png'%(str(wavelength_bin+1).zfill(4)))
    plt.close()

    if nsteps >= 500:
        if burn:
            samples = sampler.chain[:, int(nsteps/2):, :].reshape((-1, ndim))
        else:
            samples = sampler.get_chain(discard=int(nsteps/4), thin=10, flat=True)
    else:
        samples = sampler.chain[:, -100:, :].reshape((-1, ndim))

    print('\n')
    # generate median, upper and lower bounds
    med, up, low, mode = recover_quartiles_single(samples,starting_model.namelist,bin_number=(wavelength_bin+1),verbose=True,save_result=True,burn=burn)

    if not burn and ndim > 1:
        # pickle.dump(sampler,open('emcee_sampler_wb%s.pickle'%(str(wavelength_bin+1).zfill(2)),'wb'))
        # generate and save corner plot
        samples_corner = samples

        # ~ if nsteps > 6000:
            # ~ samples_corner = sampler.chain[:, -3000:, :].reshape((-1, ndim))
        # ~ else:
            # ~ samples_corner = sampler.chain[:, int(nsteps/2):, :].reshape((-1, ndim))

        make_corner_plot(samples_corner,bin_number=(wavelength_bin+1),save_fig=True,namelist=starting_model.namelist,parameter_modes=mode)

    fitted_model = tmgp.update_model(starting_model,med)

    fitted_chi2 = fitted_model.chisq(x,y,e)
    fitted_reducedChi2 = fitted_model.reducedChisq(x,y,e)
    fitted_rms = fitted_model.rms(x,y,e)*1e6
    fitted_lnlike = fitted_model.lnlike(x,y,e)
    fitted_lnprob = lnprob_emcee(med,fitted_model,x,y,e,None,sys_priors,typeII)
    fitted_BIC = fitted_model.BIC(x,y,e)

    print("\n--- Using medians of posteriors ---")
    print('chi2 = %f' % fitted_chi2)
    print('Reduced chi2 = %f' % fitted_reducedChi2)
    print('Lnlikelihood = %f' % fitted_lnlike)
    print('Lnprobability = %f' % fitted_lnprob)
    print('Residual RMS (ppm) = %f' % fitted_rms)
    print('BIC = %f' % fitted_BIC)

    diagnostic_tab.write("\n### Bin %d ###\n" % (wavelength_bin+1))
    diagnostic_tab.write("\n--- Using medians of posteriors --- \n")
    diagnostic_tab.write('Chi2 = %f \n' % fitted_chi2)
    diagnostic_tab.write('Reduced chi2 = %f \n' % fitted_reducedChi2)
    diagnostic_tab.write('Lnlikelihood = %f \n' % fitted_lnlike)
    diagnostic_tab.write('Lnprobability = %f \n' % fitted_lnprob)
    diagnostic_tab.write('Residual RMS (ppm) = %f \n' % fitted_rms)
    diagnostic_tab.write('BIC = %f \n' % fitted_BIC)

    mode_model = copy.deepcopy(starting_model)
    mode_model = tmgp.update_model(mode_model,mode)

    mode_chi2 = mode_model.chisq(x,y,e)
    mode_reducedChi2 = mode_model.reducedChisq(x,y,e)
    mode_rms = mode_model.rms(x,y,e)*1e6
    mode_lnlike = mode_model.lnlike(x,y,e)
    mode_lnprob = lnprob_emcee(mode,mode_model,x,y,e,None,sys_priors,typeII)
    mode_BIC = mode_model.BIC(x,y,e)

    print("\n--- Using modes of posteriors ---")
    print('chi2 = %f' % mode_chi2)
    print('Reduced chi2 = %f' % mode_reducedChi2)
    print('Lnlikelihood = %f' % mode_lnlike)
    print('Lnprobability = %f' % mode_lnprob)
    print('Residual RMS (ppm) = %f' % mode_rms)
    print('BIC = %f' % mode_BIC)

    diagnostic_tab.write("\n--- Using modes of posteriors --- \n")
    diagnostic_tab.write('Chi2 = %f \n' % mode_chi2)
    diagnostic_tab.write('Reduced chi2 = %f \n' % mode_reducedChi2)
    diagnostic_tab.write('Lnlikelihood = %f \n' % mode_lnlike)
    diagnostic_tab.write('Lnprobability = %f \n' % mode_lnprob)
    diagnostic_tab.write('Residual RMS (ppm) = %f \n' % mode_rms)
    diagnostic_tab.write('BIC = %f \n' % mode_BIC)

    try:
        print('\nAutocorrelation time for each parameter = ',np.round(sampler.acor).astype(int))
        # Alternatively something like: emcee.autocorr.integrated_time(sampler.chain, low=10, high=None, step=1, c=5, full_output=True,axis=0, fast=False)
        diagnostic_tab.write('\nAutocorrelation time for each parameter = ')
        for ac in np.round(sampler.acor).astype(int):
            diagnostic_tab.write('%d '%ac)
        diagnostic_tab.write('\n')

        print('nsamples/median(autocorrelation time) = %d'%np.round(nsteps/np.median(sampler.acor)))
        diagnostic_tab.write('nsamples/median(autocorrelation time) = %d \n'%(np.round(nsteps/np.median(sampler.acor))))
    except:
        print("\nAutocorrelation time can't be calculated - chains likely too short")
        diagnostic_tab.write("\nAutocorrelation time can't be calculated - chains likely too short \n")

    print('Acceptance fraction = %f'%(np.mean(sampler.acceptance_fraction)))

    diagnostic_tab.write('Acceptance fraction = %f \n'%(np.mean(sampler.acceptance_fraction)))
    diagnostic_tab.write('Total steps = %d \n'%(nsteps))

    diagnostic_tab.close()

    if not burn:
        pickle.dump(fitted_model,open('prod_model_wb%s.pickle'%(str(wavelength_bin+1).zfill(4)),'wb'))
        pickle.dump(mode_model,open('parameter_modes_model_wb%s.pickle'%(str(wavelength_bin+1).zfill(4)),'wb'))

    if burn:
        print("...burn-in complete for bin %d"%(wavelength_bin+1))
    else:
        print("...production complete for bin %d"%(wavelength_bin+1))

    sampler.reset()

    return med,up,low,fitted_model


def advance_chain(sampler,p0,nsteps,burn,save_chain,wavelength_bin):
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


def beta_rescale_uncertainties(beta_factors,best_fit_tab,trans_spec_tab=None):
    """Using beta factors found from rms vs bins, rescale the parameter uncertainties given the unaccounted for red noies.

    Inputs:
    beta_factors - the np array of beta factors
    best_fit_tab - the table of best fitting parameters
    trans_spec_Tab - the transmission spectrum table

    Returns:
    nothing but new tables with _rescaled_beta_uncertainties.txt"""

    # First load in the existing table info.
    best_fit_pars = np.genfromtxt(best_fit_tab,dtype=str)
    value_names = best_fit_pars[:,0]
    nbins = len(beta_factors)
    npars = int(len(value_names)/nbins)

    rescaled_pve = best_fit_pars[:,4].astype(float).reshape(nbins,npars) * beta_factors.reshape(nbins,1)
    rescaled_nve = best_fit_pars[:,6].astype(float).reshape(nbins,npars) * beta_factors.reshape(nbins,1)

    rescaled_pve = rescaled_pve.reshape(nbins*npars)
    rescaled_nve = rescaled_nve.reshape(nbins*npars)

    new_tab = open("%s_rescaled_beta_uncertainties.txt"%best_fit_tab[:-4],"w")

    current_bin = 1
    for i in range(npars):
        new_bin = int(best_fit_pars[:,0][i].split("_")[1])
        if new_bin > current_bin:
            new_tab.write("#------------------\n")
            current_bin = new_bin
        new_tab.write("%s = %f + %f - %f \n"%(best_fit_pars[:,0][i],best_fit_pars[:,2][i].astype(float),rescaled_pve[i],rescaled_nve[i]))

    if trans_spec_tab is not None:
        new_tab_2 = open("transmission_spectrum_rescaled_beta_uncertainties.txt","w")
        new_tab_2.write("# Wavelength bin centre, wavelength bin full width, Rp/Rs, Rp/Rs +ve error, Rp/Rs -ve error \n")
        w,we,k,k_up,k_lo = np.loadtxt(trans_spec_tab,unpack=True,usecols=[0,1,2,3,4])
        k_up *= beta_factors
        k_lo *= beta_factors

        nbins = len(k_up)

        for i in range(nbins):
            new_tab_2.write("%f %f %f %f %f \n"%(w[i],we[i],k[i],k_up[i],k_lo[i]))

        new_tab_2.close()

    return
