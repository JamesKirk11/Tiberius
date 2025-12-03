#### Author of this code: Eva-Maria Ahrer, adapted from Tiberius TransitGPPM model (author: J. Kirk)

import numpy as np
import pandas as pd


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
    def __init__(self,flux,flux_error,time_array,prior_file,fit_models,model_inputs):

        """
        

        Inputs:
        flux              - the light curve flux data points. 
        flux_error        - the errors on the flux data points to be fitted. 
        time_array        - the array of time
        prior_file        - .txt file with priors
        model_inputs      - input for models
        ld_law            - the limb darkening law we want to use: linear/quadratic/nonlinear/squareroot, default = quadratic
        

        Can return:
        - model calculated on a certain time array
        - ... 
        """

        self.flux_array = flux
        self.flux_err = flux_error
        self.time_array = time_array

        file = pd.read_csv(prior_file, sep='\s+', comment='#')
        currVals = list(file['value'])
        param_names = list(file['Name'])
        fixed = list(file['fitting'])
        self.param_dict = {}
        self.param_list_free = []
        self.prior_dict = {}

        for i in range(len(param_names)):
            if fixed[i] == 'free':
                self.param_list_free.append(param_names[i])
                self.param_dict[param_names[i]] = Param(currVals[i])
                self.prior_dict[param_names[i]+'_1'] = file['prior_1'][i]
                self.prior_dict[param_names[i]+'_2'] = file['prior_2'][i]
                self.prior_dict[param_names[i]+'_prior'] = file['prior_type'][i]
            elif fixed[i] == 'fixed':
                self.param_dict[param_names[i]] = float(currVals[i])
            else:
                print('something is wrong with your prior file')

        # initialise models
        self.transit_model_package = fit_models['transit_model']
        self.transit_model_inputs = model_inputs['transit_model']
        self.systematics_model_methods = fit_models['systematics_model']
        self.systematic_model_inputs = model_inputs['systematic_model']
        self.gp_model_inputs = model_inputs['gp_model']
        self.spot_model_package = model_inputs['spot_model']
        

        if transit_model_package == 'batman':
            from fitting_utils import BatmanModel
            self.transit_model = BatmanModel(self.param_dict, self.param_list_free, self.transit_model_inputs)
        elif transit_model_package == 'catwoman':
            from fitting_utils import CatwomanModel
            self.transit_model = CatwomanModel(self.param_dict, self.param_list_free, self.transit_model_inputs)
        
        from fitting_utils import systematics_model as sm
        self.systematic_model = sm.SystematicsModel(self.param_dict, self.systematics_model_inputs,
                                                        self.systematics_model_methods, self.time_array)
        
        if  self.gp_model_inputs['kernel_classes'] is not None:
            from fitting_utils import GPModel as gpm
            self.GP_used = True
            self.GP_model = gpm.GPModel(self.param_dict,self.gp_model_inputs, self.time_array, self.flux, self.flux_error)
        else:
            self.GP_used = False
        
        self.spot_used = False # add spot model here
        

    def return_free_parameter_list(self):
        return self.param_dict_free   
    def return_parameter_dict(self):
        return self.param_dict    

    def calc(self,time=None,decompose=False):

        """Calculates and returns the evaluated Mandel & Agol transit model, using catwoman.

        Inputs:
        time - the array of times at which to evaluate the model. Can be left blank if this has not changed from the initial init call.

        Returns:
        model - the full light curve"""

        if time is None:
            time = self.time_array

        
        transit_calc = self.transit_model.calc(time)
        model_calc = np.array(transit_calc)

        sys_calc = self.systematic_model.calc(time, decompose=decompose)
        model_calc *= sys_calc

        # if self.spot_used:
            # spot_model_calc = # Evie add spot model
            # model_calc += spot_model_calc
        

        if self.GP_used:
            GP_calc = self.GP_model.calc(time, model_calc, decompose=decompose)
            model_calc *= GP_calc


        return model


    def update_model(self,theta):
        for i in range(len(theta)):
            self.param_dict[self.param_list_free[i]].currVal = theta[i]
        return 
        

    def return_flux_err(self):
        if 'infl_err' in param_list_free:
            return self.param_dict['infl_err'].currVal * self.flux_err
        else:
            return self.flux_err


    def calc_residuals(self):
        curr_model = self.calc(self.time_array)
        return flux_array - curr_model

