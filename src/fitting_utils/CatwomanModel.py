#### Author of this code: Eva-Maria Ahrer, adapted from Tiberius TransitGPPM model (author: J. Kirk)

import numpy as np
import catwoman

import dynesty

from scipy import optimize,stats
import matplotlib.pyplot as plt

from fitting_utils import parametric_fitting_functions as pf
from fitting_utils import plotting_utils as pu
from fitting_utils import priors

class CatwomanModel(object):
    def __init__(self,pars_dict,flux,flux_error,time_array,prior_dict,nested_parameters,k_m_e_equal=False,ld_law="quadratic"):

        """
        

        Inputs:
        pars_dict         - the dictionary of the planet's transit parameters
        flux              - the light curve flux data points. 
        flux_error        - the errors on the flux data points to be fitted. 
        time_array        - the array of time
        prior_dict        - dictionary with all the information needed to set up the priors (see fitting_input)
        nested_parameters - parameters needed for dynesty; array of live points per dimension and precision criterion
        ld_law            - the limb darkening law we want to use: linear/quadratic/nonlinear/squareroot, default = quadratic
        k_m_e_equal       - if morning and evening limb are forced to have the same radius (True or False), default=False
        
        

        Can return:
        - model calculated on a certain time array
        - result of catwoman dynesty fitting
        - ... 
        """

        self.pars = pars_dict
        self.flux_array = flux
        self.flux_err = flux_error
        self.time_array = time_array
        self.prior_dict = prior_dict
        self.ld_law = ld_law
        self.k_m_e_equal = k_m_e_equal
        self.nDims_dict = []
        self.live_points, self.precision_criterion = nested_parameters


        self.namelist = [k for k in self.pars.keys() if self.pars[k] is not None and not isinstance(self.pars[k],float)]
        self.data = [v for v in self.pars.values() if v is not None and not isinstance(v,float)]


        if isinstance(self.pars['u1'],float):
            self.fix_u1 = True
        else:
            self.fix_u1 = False

        if self.ld_law != "linear":
            if isinstance(self.pars['u2'],float):
                self.fix_u2 = True
            else:
                self.fix_u2 = False

        if self.ld_law == "nonlinear":
            if isinstance(self.pars['u3'],float):
                self.fix_u3 = True
            else:
                self.fix_u3 = False

            if isinstance(self.pars['u4'],float):
                self.fix_u4 = True
            else:
                self.fix_u4 = False

        ##### Catwoman initialisation - note this is first outside of model calculation as it is the fastest way

        self.catwoman_params = catwoman.TransitParams()

        self.catwoman_params.t0 = self.pars['t0']              #time of inferior conjuction (in days)
        self.catwoman_params.per = self.pars['period']         #orbital period (in days)
        self.catwoman_params.a = self.pars['aRs']              #semi-major axis (in units of stellar radii)
        self.catwoman_params.inc = self.pars['inc']            #orbital inclination (in degrees)
        self.catwoman_params.ecc = self.pars['ecc']            #eccentricity
        self.catwoman_params.w = self.pars['w']                #longitude of periastron (in degrees)
        self.catwoman_params.phi = 0.                          #angle of rotation of top semi-circle (in degrees)

        if self.k_m_e_equal:
            self.catwoman_params.rp = self.pars['k'].currVal   #if morning = evening we only have one radius
            self.catwoman_params.rp2 = self.pars['k'].currVal
            self.nDims_dict.append('k')
        else:
            self.catwoman_params.rp = self.pars['k1'].currVal  #top semi-circle radius (in units of stellar radii)
            self.catwoman_params.rp2 = self.pars['k2'].currVal #bottom semi-circle radius (in units of stellar radii)
            self.nDims_dict.append('k1')
            self.nDims_dict.append('k2')
            
        # limb-darkening 
        gamma = []
        if self.fix_u1:
            gamma.append(self.pars['u1'])
        else:
            gamma.append(self.pars['u1'].currVal)
            self.nDims_dict.append('u1')

        if self.ld_law != "linear":
            if self.fix_u2:
                gamma.append(self.pars['u2'])
            else:
                gamma.append(self.pars['u2'].currVal)
                self.nDims_dict.append('u2')

        if self.ld_law == "nonlinear":
            if self.fix_u3:
                gamma.append(self.pars['u3'])
            else:
                gamma.append(self.pars['u3'].currVal)
                self.nDims_dict.append('u3')

        if self.ld_law == "nonlinear":
            if self.fix_u4:
                gamma.append(self.pars['u4'])
            else:
                gamma.append(self.pars['u4'].currVal)
                self.nDims_dict.append('u4')
        self.catwoman_params.u = gamma                #limb darkening coefficients [u1, u2, ..,]

        if self.ld_law == "squareroot": # change to match Catwoman's naming
            self.catwoman_params.limb_dark = "square-root"       #limb darkening model
        else:
            self.catwoman_params.limb_dark = self.ld_law

        self.catwoman_model = catwoman.TransitModel(self.catwoman_params, time_array)    #initializes model

        self.nDims_dict.append('infl_err') # add for error inflation

        self.nDims = len(self.nDims_dict)

    

    def calc(self,time=None):

        """Calculates and returns the evaluated Mandel & Agol transit model, using catwoman.

        Inputs:
        time - the array of times at which to evaluate the model. Can be left blank if this has not changed from the initial init call.

        Returns:
        model - the modelled catwoman light curve"""


        if self.k_m_e_equal:
            self.catwoman_params.rp = self.pars['k'].currVal   #if morning = evening we only have one radius
            self.catwoman_params.rp2 = self.pars['k'].currVal  
        else:
            self.catwoman_params.rp = self.pars['k1'].currVal  #top semi-circle radius (in units of stellar radii)
            self.catwoman_params.rp2 = self.pars['k2'].currVal #bottom semi-circle radius (in units of stellar radii)

        # set up limb darkening coefficients
        gamma = []
        if self.fix_u1:
            gamma.append(self.pars['u1'])
        else:
            gamma.append(self.pars['u1'].currVal)

        if self.ld_law != "linear":
            if self.fix_u2:
                gamma.append(self.pars['u2'])
            else:
                gamma.append(self.pars['u2'].currVal)

        if self.ld_law == "nonlinear":
            if self.fix_u3:
                gamma.append(self.pars['u3'])
            else:
                gamma.append(self.pars['u3'].currVal)

        if self.ld_law == "nonlinear":
            if self.fix_u4:
                gamma.append(self.pars['u4'])
            else:
                gamma.append(self.pars['u4'].currVal)

        self.catwoman_params.u = gamma                #limb darkening coefficients [u1, u2]

        if time is not None:
            if np.any(time != self.time_array): # optionally recalculating catwoman model if the time array has changed
                self.catwoman_model = catwoman.TransitModel(self.catwoman_params, time, nthreads=1)

        model = self.catwoman_model.light_curve(self.catwoman_params)
        return model

    def prior_setup(self, x):

        """Calculates the priors from a uniform cube 

        Inputs:
        x - the multidimensional cube from the nested sampler 

        Returns:
        theta - the transformed parameters"""
        
        theta = [0] * self.nDims
        
        index_run = 0
        if self.k_m_e_equal:
            if self.prior_dict['k_m_prior'] == 'N':
                theta[index_run] = priors.GaussianPrior(self.prior_dict['k_m_1'],self.prior_dict['k_m_2'])(np.array(x[index_run]))
                index_run += 1
            elif self.prior_dict['k_m_prior'] == 'U':
                theta[index_run] = priors.UniformPrior(self.prior_dict['k_m_1'],self.prior_dict['k_m_2'])(np.array(x[index_run]))
                index_run += 1
            else:
                #add exit here 
                print('Choose either normal or uniform prior') 
        else:
            if self.prior_dict['k_m_prior'] == 'N':
                theta[index_run] = priors.GaussianPrior(self.prior_dict['k_m_1'],self.prior_dict['k_m_2'])(np.array(x[index_run]))
                index_run += 1
            elif self.prior_dict['k_m_prior'] == 'U':
                theta[index_run] = priors.UniformPrior(self.prior_dict['k_m_1'],self.prior_dict['k_m_2'])(np.array(x[index_run]))
                index_run += 1
            else:
                #add exit here 
                print('Choose either normal or uniform prior for k_m')
                
            if self.prior_dict['k_e_prior'] == 'N':
                theta[index_run] = priors.GaussianPrior(self.prior_dict['k_e_1'],self.prior_dict['k_e_2'])(np.array(x[index_run]))
                index_run += 1
            elif self.prior_dict['k_e_prior'] == 'U':
                theta[index_run] = priors.UniformPrior(self.prior_dict['k_e_1'],self.prior_dict['k_e_2'])(np.array(x[index_run]))
                index_run += 1
            else:
                #add exit here 
                print('Choose either normal or uniform prior for k_e')

        if not self.fix_u1:
            theta[index_run] = priors.GaussianPrior(self.prior_dict['u1'],
                                                    self.prior_dict['u1_err']*self.prior_dict['ld_unc_multiplier'])(np.array(x[index_run]))
            index_run += 1
        if self.ld_law != "linear":
            if not self.fix_u2:
                theta[index_run] = priors.GaussianPrior(self.prior_dict['u2'],
                                                        self.prior_dict['u2_err']*self.prior_dict['ld_unc_multiplier'])(np.array(x[index_run]))
                index_run += 1
        if self.ld_law == "nonlinear":
            if not self.fix_u3:
                theta[index_run] = priors.GaussianPrior(self.prior_dict['u3'],
                                                        self.prior_dict['u3_err']*self.prior_dict['ld_unc_multiplier'])(np.array(x[index_run]))
                index_run += 1
            if not self.fix_u4:
                theta[index_run] = priors.GaussianPrior(self.prior_dict['u4'],
                                                        self.prior_dict['u4_err']+self.prior_dict['ld_unc_multiplier'])(np.array(x[index_run]))
                index_run += 1

        if self.prior_dict['error_infl_prior'] == 'N':
            theta[index_run] = priors.GaussianPrior(self.prior_dict['err_1'],self.prior_dict['err_2'])(np.array(x[index_run]))
            index_run += 1
        elif self.prior_dict['error_infl_prior'] == 'U':
            theta[index_run] = priors.UniformPrior(self.prior_dict['err_1'],self.prior_dict['err_2'])(np.array(x[index_run]))
            index_run += 1
        else:
            #add exit here 
            print('Choose either normal or uniform prior for err_inflation')
            
        return theta


    def update_model(self,theta):
        for i in range(len(theta)):
            self.pars[self.nDims_dict[i]].currVal = theta[i]
        return

    def return_curr_parameters(self):
        return self.pars

    def return_fitted_parameters(self):
        return self.nDims_dict


    def loglikelihood(self,theta):

        self.update_model(theta)
        
        model_values = self.calc(self.time_array)
        
        residuals = self.flux_array - model_values

        new_noise = self.flux_err*self.pars['infl_err'].currVal
        
        N = len(self.flux_array)
        logL = -N/2. *  np.log(2*np.pi)
        logL += - np.sum(np.log(new_noise)) - np.sum(residuals**2 / (2 * new_noise**2))
        return logL


    def run_dynesty(self):

        sampler = dynesty.NestedSampler(self.loglikelihood, self.prior_setup, self.nDims,nlive=self.live_points*self.nDims)
        sampler.run_nested(dlogz=self.precision_criterion, print_progress=True) 
        results = sampler.results

        return results
        
    

    def chisq(self,time,flux,flux_err):

        """Evaluate the chi2 of the object.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points

        Returns:
        evaluated chi squared
        """
        resids = (flux - self.calc(time))/flux_err
        return np.sum(resids*resids)
        

    def reducedChisq(self,time,flux,flux_err):
        """Evaluate the reduced chi2 of the object.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points

        Returns:
        the evaluated reduced chi squared"""
        return self.chisq(time,flux,flux_err) / (len(flux) - self.nDims)


    def rms(self,time,flux,flux_err=None):
        """Evaluate the RMS of the residuals

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points

        Returns:
        the evaluated RMS of the residuals"""

        resids = (flux - self.calc(time))

        rms = np.sqrt(np.square(resids).mean())

        return rms

class Param(object):
    '''A Param (parameter) needs a starting value and a current value. However, when first defining the Param object, it takes the starting value as the current value.

    Inputs:
    startVal: the starting value for the parameter

    Returns:
    Param object'''
    def __init__(self,startVal):
        self.startVal = startVal
        self.currVal  = startVal 
