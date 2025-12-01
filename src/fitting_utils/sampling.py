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

        self.pars_dict = pars_dict
        self.prior_dict = prior_dict
        self.sampling_method = sampling_method
        self.sampling_arguments = sampling_arguments

        if sampling_method == 'dynesty':
            self.nDims = len(pars_dict)


        self.namelist = [k for k in self.pars.keys() if self.pars[k] is not None and not isinstance(self.pars[k],float)]


    def prior_setup(self, x):

        if sampling_method == 'dynesty':
            theta = [0] * self.nDims
            
            for i in range(self.nDims):
                if self.prior_dict['%s_prior'%pars_dict[i]] == 'N':
                    theta[i] = priors.GaussianPrior(self.prior_dict['%s_1'%pars_dict[i]], self.prior_dict['%s_2'%pars_dict[i]])(np.array(x[i]))
                elif self.prior_dict['%s_prior'%pars_dict[i]] == 'U':
                    theta[i] = priors.UniformPrior(self.prior_dict['%s_1'%pars_dict[i]], self.prior_dict['%s_2'%pars_dict[i]])(np.array(x[i]))

            return theta



    def loglikelihood_dynesty(self,theta):
        noise = lightcurve.update_model(theta)
        
        residuals = lightcurve.calc_residuals()
             
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
        
