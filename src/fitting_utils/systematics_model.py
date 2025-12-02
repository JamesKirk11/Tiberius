import numpy as np
from fitting_utils import parametric_fitting_functions as pf


class SystematicsModel:
    """
    Encapsulates exponential ramps, polynomial red-noise trends,
    and step functions for systematics modeling.
    """

    def __init__(self,
                 input_dict,
                 param_dict,
                 param_list_free
                 systematics_model_methods,
                 systematics_model_inputs,
                 wb):
        """
        Parameters
        ----------

        """
        # self.pars = pars
        # self.poly_orders = np.array(poly_orders) if poly_orders is not None else None
        # self.systematics_inputs = systematics_inputs
        # self.exp_ramp_components = exp_ramp_components
        # self.exp_ramp_fixed = exp_ramp_fixed
        # self.poly_fixed = poly_fixed
        # self.step_func_used = step_func_used
        # self.white_light_fit = white_light_fit

        self.param_dict = param_dict
        self.param_list_free = param_list_free
        self.methods = systematics_model_methods
        self.inputs = systematics_model_inputs

        # store inputs
        self.model_inputs = model_inputs
        self.wb = wb

        # ---------------------------------------------------------
        # Parse exponential ramp usage
        # ---------------------------------------------------------
        self.exp_ramp_used = bool(int(input_dict.get("exponential_ramp", 0)))

        # ---------------------------------------------------------
        # Parse step function usage
        # ---------------------------------------------------------
        self.step_func_used = bool(int(input_dict.get("step_function", 0)))

        # ---------------------------------------------------------
        # Parse polynomial orders
        # ---------------------------------------------------------
        poly_orders_str = input_dict.get("polynomial_orders")
        if poly_orders_str is not None:
            self.polynomial_orders = np.array([int(i) for i in poly_orders_str.split(",")])
            if self.polynomial_orders.sum() == 0:
                self.poly_used = False
                self.polynomial_orders = None
            else:
                self.poly_used = True
        else:
            self.poly_used = False
            self.polynomial_orders = None

        # ---------------------------------------------------------
        # Load starting coefficients if provided
        # ---------------------------------------------------------
        coeff_file = input_dict.get("polynomial_coefficients")
        if self.poly_used or self.exp_ramp_used:
            if coeff_file is not None:
                keys = np.loadtxt(coeff_file, usecols=0, dtype=str)
                values = np.loadtxt(coeff_file, usecols=2)

                self.polynomial_coefficients = []
                self.ramp_coefficients = []

                for k, v in zip(keys, values):
                    # Only load coefficients corresponding to this white-light bin
                    if int(k.split("_")[1]) == wb + 1:
                        if k.startswith("c"):
                            self.polynomial_coefficients.append(v)
                        elif k.startswith("r"):
                            self.ramp_coefficients.append(v)
            else:
                self.polynomial_coefficients = None
                self.ramp_coefficients = None
        else:
            self.polynomial_coefficients = None
            self.ramp_coefficients = None

        # ---------------------------------------------------------
        # Normalize inputs if requested
        # ---------------------------------------------------------
        norm_inputs = bool(int(input_dict.get("normalise_inputs", 0)))
        if norm_inputs and self.model_inputs is not None:
            self.systematics_model_inputs = np.array(
                [(i - i.mean()) / i.std() for i in self.model_inputs]
            )
        else:
            self.systematics_model_inputs = np.array(self.model_inputs)



    # ---------------------------------------------------------
    # Exponential ramp
    # ---------------------------------------------------------
    def exponential_ramp(self, time):
        """
        Compute exponential ramp model.
        """
        if self.exp_ramp_components == 0:
            return np.ones_like(time)

        model = np.ones_like(time)

        for i in range(0, 2 * self.exp_ramp_components, 2):
            amp = self.pars[f"r{i+1}"]
            tau = self.pars[f"r{i+2}"]

            amp_val = amp if self.exp_ramp_fixed else amp.currVal
            tau_val = tau if self.exp_ramp_fixed else tau.currVal

            model += amp_val * np.exp(tau_val * time)

        return model


    # ---------------------------------------------------------
    # Polynomial red-noise trend
    # ---------------------------------------------------------
    # def red_noise_poly(self, time=None, sys_inputs=None, deconstruct=False):
    #     """
    #     Polynomial systematics model using pf.systematics_model().
    #     """
    #     if self.poly_orders is None:
    #         return np.ones_like(time)
    #
    #     # Extract polynomial coefficients
    #     if self.poly_fixed:
    #         coeffs = np.array([self.pars[f"c{i}"] for i in range(1, self.poly_orders.sum() + 2)])
    #     else:
    #         coeffs = np.array([self.pars[f"c{i}"].currVal for i in range(1, self.poly_orders.sum() + 2)])
    #
    #     inputs = sys_inputs if sys_inputs is not None else self.systematics_inputs
    #
    #     if deconstruct:
    #         trend, components = pf.systematics_model(coeffs, inputs, self.poly_orders,
    #                                                  GP=False, deconstruct_polys=True)
    #         return trend, components
    #
    #     trend = pf.systematics_model(coeffs, inputs, self.poly_orders,
    #                                  GP=False, deconstruct_polys=False)
    #     return trend


    def polynomials(self,time,deconstruct_polys=False):

        # Ancillary data and poly_orders are ALWAYS in the order:

        offset = self.param_dict["f_norm"]

        offset = p0[0] # offset added to model, which is at the start of p0
        systematics_model = 0 # initiate systematics offset as being zero so that susbequent models can be added together
        current_index = 0
        individual_models = []

        for i in range(len(model_inputs)):
            if poly_orders[i] > 0:
                poly_coefficients = np.hstack((p0[1+current_index:1+current_index+poly_orders[i]],[0]))
                poly = np.poly1d(poly_coefficients)

                if normalise_inputs:
                    input_norm = (model_inputs[i]-model_inputs[i].mean())/model_inputs[i].std()
                else:
                    input_norm = model_inputs[i]

                poly_eval = poly(input_norm)

                # add to current systematics model
                systematics_model += poly_eval
                individual_models.append(poly_eval)

                current_index += poly_orders[i]

        if deconstruct_polys:
            return systematics_model + offset,individual_models
        else:
            return systematics_model + offset


    # ---------------------------------------------------------
    # Step function (JWST mirror tilt correction)
    # ---------------------------------------------------------
    def step_function(self, time):
        """
        Simple step function for JWST tilt events.
        """
        if not self.step_func_used:
            return np.ones_like(time)

        model = np.ones_like(time)

        bp_par = self.pars["breakpoint"]
        step_par = self.pars["step1"]

        bp = int(bp_par.currVal if self.white_light_fit else bp_par)
        step_val = step_par.currVal

        model[:bp] *= step_val

        return model


    # ---------------------------------------------------------
    # Combine all systematics components
    # ---------------------------------------------------------
    def calc(self, time, sys_inputs=None):
        """
        Returns the TOTAL systematics model evaluated at 'time'.

        This is the product:
            exponential_ramp
            × red_noise_poly
            × step_function

        Only includes components that are enabled.
        """
        model = np.ones_like(time)

        if "ramp" in self.methods:
            # Multiply ramp
            model *= self.exponential_ramp(time)

        # Multiply polynomial trend
        if "polynomial" in self.methods:

            f_norm = self.param_dict['f_norm']

            poly_model = 0 # initiate systematics offset as being zero so that susbequent models can be added together

            for poly_cfg in self.methods["polynomials"]:
                model += self.evaluate_single_polynomial(poly_cfg)

        if self.poly_orders is not None:
            model *= self.red_noise_poly(time=time, sys_inputs=sys_inputs)

        # Multiply step function
        if self.step_func_used:
            model *= self.step_function(time)

        return model

    # ---------------------------------------------------------
    # Return each component separately (for plotting)
    # ---------------------------------------------------------
    def decompose(self, time, sys_inputs=None):
        """
        Returns a dictionary containing:
            - each individual systematics component
            - the total combined systematics model

        Useful for plotting contributions of:
            ramp, polynomial, step function, etc.
        """
        components = {}

        # Exponential ramp
        if self.exp_ramp_components != 0:
            ramp = self.exponential_ramp(time)
            components["ramp"] = ramp

        # Polynomial trend
        if self.poly_orders is not None:
            _, poly = self.red_noise_poly(time=time, sys_inputs=sys_inputs, deconstruct=True)
            components["poly"] = poly

        # Step function
        if self.step_func_used:
            step = self.step_function(time)
            components["step"] = step

        # Total model (product of components)
        # total = ramp * poly * step
        # components["total"] = total

        return components
