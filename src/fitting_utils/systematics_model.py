import numpy as np
from fitting_utils import parametric_fitting_functions as pf


class SystematicsModel:
    """
    Encapsulates exponential ramps, polynomial red-noise trends,
    and step functions for systematics modeling.
    """

    def __init__(param_dict,
                 systematics_model_inputs,
                 polynomial_orders,
                 exp_ramp,
                 exp_ramp_components,
                 step_func):

        """
        The SystematicsModel model class.

        Inputs:
        param_dict - the dictionary of the planet's transit parameters, as defined in gppm_fit.py:
        systematics_model_inputs - the ndarray of ancillary data parsed to the GP/polynomials, e.g. np.array([airmass,sky,time])
        polynomial_orders - if wanting to use polynomial detrending, use this to define the corders of each polynomial to be used. This must be the same length as the systematic_model_inputs.
        ld_law - the limb darkening law we want to use: linear/quadratic/nonlinear/squareroot
        exp_ramp - True/False. Do you want to additionally fit a 2 component expoential ramp model? Default = False
        exp_ramp_components (int) - The number of exponential ramp components to fit. Default=0, no ramp.
        step_func - True/False. Do you want to additionally fit a step function model with arbitrary breakpoint? Default = False

        Returns:
        SystematicsModel object
        """

        self.param_dict = param_dict
        self.systematics_model_inputs = systematics_model_inputs
        self.polynomial_orders = polynomial_orders
        self.exp_ramp_used = exp_ramp
        self.exp_ramp_components = exp_ramp_components
        self.step_func_used = step_func

        # Acknowledge the fact that we're using a polynomial here
        if polynomial_orders is None:
            self.poly_used = False
        else:
            self.poly_used = True
            if type(pars_dict["c1"]) is Param:
                self.poly_fixed = False
            else:
                self.poly_fixed = True

        if exp_ramp:
            if type(pars_dict["r1"]) is Param:
                self.exp_ramp_fixed = False
            else:
                self.exp_ramp_fixed = True


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

        # extract the relevant parameters from the dictionary
        if self.poly_fixed:
            red_noise_pars = np.array([self.param_dict['c%d'%i] for i in range(1,self.polynomial_orders.sum()+2)])
        else:
            red_noise_pars = np.array([self.param_dict['c%d'%i].currVal for i in range(1,self.polynomial_orders.sum()+2)])

        if sys_model_inputs is not None:
            poly_inputs = sys_model_inputs
        else:
            poly_inputs = self.systematics_model_inputs

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

            exp_ramp_model = 1

            for i in range(0,2*self.exp_ramp_components,2):

                if self.exp_ramp_fixed:
                    exp_ramp_model += self.param_dict['r%d'%(i+1)]*np.exp(self.param_dict['r%d'%(i+2)]*time)
                else:
                    exp_ramp_model += self.param_dict['r%d'%(i+1)].currVal*np.exp(self.param_dict['r%d'%(i+2)].currVal*time)

            return exp_ramp_model


    def step_function(self,time=None):

        """The function that calculates a step function to help fit out mirror tilt events in JWST data

        Inputs:
        time - the array of times at which to evaluate the step function

        Returns:
        step_model - the evaluated step function
        """

        step_model = np.ones_like(time)

        if self.white_light_fit:
            step_model[:int(self.param_dict["breakpoint"].currVal)] *= self.param_dict["step1"].currVal
        else:
            step_model[:int(self.param_dict["breakpoint"])] *= self.param_dict["step1"].currVal

        return step_model


    def calc(self,time,sys_model_inputs=None,decompose=False):

        combined_model = np.ones_like(time)
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
