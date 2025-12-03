import numpy as np
from fitting_utils import parametric_fitting_functions as pf
from fitting_utils.LightcurveModel import Param


class SystematicsModel:
    """
    Encapsulates exponential ramps, polynomial red-noise trends,
    and step functions for systematics modeling.
    """

    def __init__(self,param_dict,
                 systematics_model_inputs,
                 systematics_model_methods, time_array):

        """
        The SystematicsModel model class.

        Inputs:
        param_dict - the dictionary of the planetary and systematics parameters
        systematics_model_inputs - the dictionary of inputs for each systematics model
        systematics_model_methods - a list of the systematics models used

        Returns:
        SystematicsModel object
        """

        self.param_dict = param_dict
        self.systematics_model_inputs = systematics_model_inputs
        self.polynomial_model_inputs = self.systematics_model_inputs['model_inputs']
        self.systematics_model_methods = systematics_model_methods
        self.time = time_array

        if 'polynomial' in self.systematics_model_methods:
            self.polynomial_orders = self.systematics_model_inputs['polynomial_orders']
            self.poly_used = True
            if type(param_dict["c1"]) is Param:
                self.poly_fixed = False
            else:
                self.poly_fixed = True
        else:
            self.poly_used = False
            

        if 'exponential_ramp' in self.systematics_model_methods:
            self.exp_ramp_used = True
        else:
            self.exp_ramp_used = False
        
        if 'step_function' in self.systematics_model_methods:
            self.step_func_used = True
        else:
            self.step_func_used = False


    def red_noise_poly(self,time=None,sys_model_inputs=None,deconstruct_polys=False):

        """The function that calculates the time polynomial to fit the red noise. This uses the pf.systematics_model() function.

        Inputs:
        time - the array of times at which to evaluate the polynomial. Can be blank if these haven't changed from the initial init call.
        sys_model_inputs - the array of inputs to feed into the polynomial. Can be blank if these haven't changed from the initial init call.
        deconstruct_polys - do you want to additionally return the polynomial components as individual arrays for plotting? True/False

        Returns:
        red_noise_trend - the evaluated polynomial
        if deconstruct_polys == True: also returns the poly_components
        """

        if sys_model_inputs is not None:
            poly_inputs = sys_model_inputs
        else:
            poly_inputs = self.polynomial_model_inputs


        # extract the relevant parameters from the dictionary
        if self.poly_fixed:
            red_noise_pars = np.array([self.param_dict['c%d'%i] for i in range(1,self.polynomial_orders.sum()+2)])
        else:
            red_noise_pars = np.array([self.param_dict['c%d'%i].currVal for i in range(1,self.polynomial_orders.sum()+2)])

        
        # generate the model
        if deconstruct_polys:
            red_noise_trend,poly_components = pf.systematics_model(red_noise_pars,poly_inputs,self.polynomial_orders,False,deconstruct_polys)
            return red_noise_trend,poly_components
        else:
            red_noise_trend = pf.systematics_model(red_noise_pars,poly_inputs,self.polynomial_orders,False,deconstruct_polys)
            return red_noise_trend


    def exponential_ramp(self,time=None):

            """The function that calculates a two component ramp model to fit trends in light curves. This only operates over the time axis.

            Inputs:
            time - the array of times at which to evaluate the polynomial. Can be blank if these haven't changed from the initial init call.

            Returns:
            exp_ramp_model - the evaluated exponential ramp
            """

            if type(pars_dict["r1"]) is Param:
                r1 = self.param_dict['r1'].currVal
            else:
                r1 = self.param_dict['r1']
            
            if type(pars_dict["r2"]) is Param:
                r2 = self.param_dict['r2'].currVal
            else:
                r2 = self.param_dict['r2']

            if self.exp_ramp_fixed:
                exp_ramp_model = r1*np.exp(r2*self.time)
            else:
                exp_ramp_model = r1*np.exp(r2*self.time)

            return exp_ramp_model


    def step_function(self,time=None):

        """The function that calculates a step function to help fit out mirror tilt events in JWST data

        Inputs:
        time - the array of times at which to evaluate the step function

        Returns:
        step_model - the evaluated step function
        """

        step_model = np.ones_like(self.time)

        if self.param_dict["breakpoint"] is Param:
            step_model[:int(self.param_dict["breakpoint"].currVal)] *= self.param_dict["step1"].currVal
        else:
            step_model[:int(self.param_dict["breakpoint"])] *= self.param_dict["step1"].currVal

        return step_model


    def calc(self,time=None,poly_inputs=None,decompose=False):

        if time is None:
            combined_model = np.ones_like(self.time)
        else:
            combined_model = np.ones_like(time)

        if poly_inputs is None:
            sys_model_inputs = self.polynomial_model_inputs
        else:
            sys_model_inputs = poly_inputs

        model_components = {}

        if self.poly_used:
            poly_model = self.red_noise_poly(self,time,sys_model_inputs,deconstruct_polys=decompose)
            combined_model *= poly_model
            model_components['poly_model'] = poly_model

        if self.exp_ramp_used:
            ramp_model = self.exponential_ramp(self,time)
            combined_model *= ramp_model
            model_components['ramp_model'] = ramp_model

        if self.step_func_used:
            step_model = self.step_function(self,time)
            combined_model *= step_model
            model_components['step_model'] = step_model

        if decompose:
            return combined_model, model_components

        return combined_model
