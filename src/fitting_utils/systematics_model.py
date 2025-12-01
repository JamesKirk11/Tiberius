import numpy as np
from fitting_utils import parametric_fitting_functions as pf


class SystematicsModel:
    """
    Encapsulates exponential ramps, polynomial red-noise trends,
    and step functions for systematics modeling.
    """

    def __init__(self,
                 pars,
                 poly_orders=None,
                 systematics_inputs=None,
                 exp_ramp_components=0,
                 exp_ramp_fixed=False,
                 poly_fixed=False,
                 step_func_used=False,
                 white_light_fit=False):
        """
        Parameters
        ----------
        pars : dict
            Dictionary of parameter objects or floats.
        poly_orders : np.array or list
            Orders of the polynomial basis.
        systematics_inputs : list/array
            Inputs to feed into polynomial systematics model.
        exp_ramp_components : int
            Number of exponential ramp components.
        exp_ramp_fixed : bool
        poly_fixed : bool
        step_func_used : bool
        white_light_fit : bool
        """
        self.pars = pars
        self.poly_orders = np.array(poly_orders) if poly_orders is not None else None
        self.systematics_inputs = systematics_inputs
        self.exp_ramp_components = exp_ramp_components
        self.exp_ramp_fixed = exp_ramp_fixed
        self.poly_fixed = poly_fixed
        self.step_func_used = step_func_used
        self.white_light_fit = white_light_fit


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
    def red_noise_poly(self, time=None, sys_inputs=None, deconstruct=False):
        """
        Polynomial systematics model using pf.systematics_model().
        """
        if self.poly_orders is None:
            return np.ones_like(time)

        # Extract polynomial coefficients
        if self.poly_fixed:
            coeffs = np.array([self.pars[f"c{i}"] for i in range(1, self.poly_orders.sum() + 2)])
        else:
            coeffs = np.array([self.pars[f"c{i}"].currVal for i in range(1, self.poly_orders.sum() + 2)])

        inputs = sys_inputs if sys_inputs is not None else self.systematics_inputs

        if deconstruct:
            trend, components = pf.systematics_model(coeffs, inputs, self.poly_orders,
                                                     GP=False, deconstruct_polys=True)
            return trend, components

        trend = pf.systematics_model(coeffs, inputs, self.poly_orders,
                                     GP=False, deconstruct_polys=False)
        return trend


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

        # Multiply ramp
        model *= self.exponential_ramp(time)

        # Multiply polynomial trend
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
