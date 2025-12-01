#### Author of this code: Eva-Maria Ahrer, adapted from Tiberius TransitGPPM model (author: J. Kirk)

import numpy as np
import pandas as pd

from scipy import optimize,stats
import matplotlib.pyplot as plt

from fitting_utils import parametric_fitting_functions as pf
from fitting_utils import plotting_utils as pu
from fitting_utils import priors

class Param(object):
    '''A Param (parameter) needs a starting value and a current value. However, when first defining the Param object, it takes the starting value as the current value.

    Inputs:
    startVal: the starting value for the parameter

    Returns:
    Param object'''
    def __init__(self,startVal):
        self.startVal = startVal
        self.currVal  = startVal 



class LightcurveModel(object):
    def __init__(self,flux,flux_error,time_array,prior_file,astrophysical_models, systematic_models,ld_law="quadratic"):

        """
        

        Inputs:
        flux              - the light curve flux data points. 
        flux_error        - the errors on the flux data points to be fitted. 
        time_array        - the array of time
        prior_file        - .txt file with priors
        ld_law            - the limb darkening law we want to use: linear/quadratic/nonlinear/squareroot, default = quadratic
        

        Can return:
        - model calculated on a certain time array
        - ... 
        """

        self.flux_array = flux
        self.flux_err = flux_error
        self.time_array = time_array
        self.ld_law = ld_law

        file = pd.read_csv(prior_file, sep='\s+')
        currVals = list(file['value'])
        param_names = list(file['Name'])
        fixed = list(file['fitting'])
        self.param_dict = {}
        self.param_list_free = []
        self.prior_dict = {}

        for i in range(len(param_names)):
            if self.fixed[i] == 'free':
                self.param_list_free.append(param_names[i])
                self.param_dict[param_names[i]] = Param(currVals[i])
                self.prior_dict[param_names[i]+'_1'] = file['prior_1'][i]
                self.prior_dict[param_names[i]+'_2'] = file['prior_2'][i]
                self.prior_dict[param_names[i]+'_prior'] = file['prior_type'][i]
            elif self.fixed[i] == 'fixed':
                self.param_dict[param_names[i]] = float(currVals[i])
            else:
                print('something is wrong with your prior file')
        

    def return_free_parameter_list(self):
        return self.param_dict_free   
    def return_parameter_dict(self):
        return self.param_dict    

    def calc(self,time=None):

        """Calculates and returns the evaluated Mandel & Agol transit model, using catwoman.

        Inputs:
        time - the array of times at which to evaluate the model. Can be left blank if this has not changed from the initial init call.

        Returns:
        model - the full light curve"""

        
        return model


    def update_model(self,theta):
        for i in range(len(theta)):
            self.param_dict[self.param_list_free[i]].currVal = theta[i]

        if 'infl_err' in param_list_free:
            return self.param_dict['infl_err'].currVal * self.flux_err
        else:
            return self.flux_err




