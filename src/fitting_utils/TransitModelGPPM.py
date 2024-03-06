#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import batman
import george
from george import kernels
from scipy import optimize,stats
import matplotlib.pyplot as plt
from fitting_utils import parametric_fitting_functions as pf
from fitting_utils import plotting_utils as pu

class TransitModelGPPM(object):
    def __init__(self,pars_dict,systematics_model_inputs,kernel_classes,flux_error,time_array,kernel_priors=None,wn_kernel=True,use_kipping=False,ld_std_priors=None,polynomial_orders=None,ld_law="quadratic",exp_ramp=False,exp_ramp_components=0,step_func=False):

        """
        The GPPM transit model class, which uses batman to generate the analytic, quadratically limb-darkened transit light curves, and george to generate the GP red noise models.
        However, this has the added option of fitting the time dependence with a polynomial, removing it as a parameter given to the GP. The thought behind this is that the GP has less to do, leading to smaller uncertainties in Rp/Rs.

        Inputs:
        pars_dict - the dictionary of the planet's transit parameters, as defined in gppm_fit.py:
        systematics_model_inputs - the ndarray of ancillary data parsed to the GP/polynomials, e.g. np.array([airmass,sky,time])
        kernel_classes - the names of the kernels to be used (same length as systematics_model_inputs). If not using a GP, keep this as 'None'
        flux_error - the errors on the flux data points to be fitted. It is not very satisfactory to parse these here but they are needed so that the can be added to the covariance matrix computed by the GP
        time_array - the array of time, defined here so that the batman model can be init upon the first call of the model
        kernel_priors - use this to define the upper and lower bounds on the kernel hyperparameters. Default=None, in which case the GP just places bounds such that the hyperparameters do not get computationally too large or small
        wn_kernel - True/False - use this to define whether a white noise kernel is to be used or not, but only if we are using a GP. Default=True
        use_kipping - True/False - use this to use David Kipping's parameterisation of the limb darkening coefficients for efficient sampling. This is not fully tested yet. Default=False
        ld_std_priors - A dictionary that can be used to place Gaussian priors on the limb darkening coefficients. Defined as {'u1_prior':standard deviation of Gaussian,'u2_prior':standard deviation of Gaussian}
        time_poly - the order of the polynomial used to fit the time dependent-noise. Default=0 (no poly)
        polynomial_orders - if wanting to use polynomial detrending, use this to define the corders of each polynomial to be used. This must be the same length as the systematic_model_inputs.
        ld_law - the limb darkening law we want to use: linear/quadratic/nonlinear/squareroot
        exp_ramp - True/False. Do you want to additionally fit a 2 component expoential ramp model? Default = False
        exp_ramp_components (int) - The number of exponential ramp components to fit. Default=0, no ramp.
        step_func - True/False. Do you want to additionally fit a step function model with arbitrary breakpoint? Default = False

        Returns:
        TransitModelGPPM object
        """

        self.pars = pars_dict
        self.systematics_model_inputs = systematics_model_inputs
        self.time_array = time_array
        self.kernel_classes = kernel_classes
        self.kernel_priors = kernel_priors
        self.wn_kernel = wn_kernel
        self.use_kipping = use_kipping
        self.ld_std_priors = ld_std_priors
        self.ld_law = ld_law
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

        # Acknowledge the fact that we're using a GP model here, useful for mcmc_utils.py
        if kernel_classes is None:
            self.GP_used = False
        else:
            self.GP_used = True

        # determine whether this is a white or spectroscopic light curve by checking whether t0 is a fit parameter or not
        if isinstance(pars_dict['t0'],float):
            self.white_light_fit = False

        else:
            # Now we are fitting the system parameters
            self.white_light_fit = True

        self.namelist = [k for k in self.pars.keys() if self.pars[k] is not None and not isinstance(self.pars[k],float)]
        self.data = [v for v in self.pars.values() if v is not None and not isinstance(v,float)]

        if self.GP_used:
            self.gp_ndim = len([c for c in kernel_classes if c is not None])
            self.starting_gp_object = self.construct_gp(split=False,compute=True,flux_err=flux_error)
            self.gp_npars = self.gp_ndim+1 # capture the GP amplitude parameter
            if wn_kernel:
                self.gp_npars += 1

        if not self.use_kipping:
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
        else:
            self.fix_u1 = self.fix_u2 = False

        ##### Batman initialisation - note this is first outside of model calculation as it is the fastest way

        self.batman_params = batman.TransitParams()
        if self.white_light_fit:
            self.batman_params.t0 = self.pars['t0'].currVal                       #time of inferior conjunction
            try:
                self.batman_params.per = self.pars['period'].currVal       #orbital period
                self.period_fixed = False
            except:
                self.batman_params.per = self.pars['period']                     #orbital inclination (in degrees)
                self.period_fixed = True
            try:
                self.batman_params.a = self.pars['aRs'].currVal                       #semi-major axis (in units of stellar radii)
                self.ars_fixed = False
            except:
                self.batman_params.a = self.pars['aRs']
                self.ars_fixed = True
            try:
                self.batman_params.inc = self.pars['inc'].currVal                     #orbital inclination (in degrees)
                self.inc_fixed = False
            except:
                self.batman_params.inc = self.pars['inc']                     #orbital inclination (in degrees)
                self.inc_fixed = True
            try:
                self.batman_params.ecc = self.pars['ecc'].currVal                     #orbital inclination (in degrees)
                self.ecc_fixed = False
            except:
                self.batman_params.ecc = self.pars['ecc']                     #orbital inclination (in degrees)
                self.ecc_fixed = True
            try:
                self.batman_params.w = self.pars['omega'].currVal                     #longitude of periastron (in degrees), fix to 90 if ecc==0
                self.omega_fixed = False
            except:
                self.batman_params.w = self.pars['omega']                     #longitude of periastron (in degrees), fix to 90 if ecc==0
                self.omega_fixed = True
        else:
            self.batman_params.t0 = self.pars['t0']                       #time of inferior conjunction
            self.batman_params.per = self.pars['period']                     #orbital period
            self.batman_params.a = self.pars['aRs']                       #semi-major axis (in units of stellar radii)
            self.batman_params.inc = self.pars['inc']                     #orbital inclination (in degrees)
            self.batman_params.ecc = self.pars['ecc']                      #eccentricity
            self.batman_params.w = self.pars['omega']                       #longitude of periastron (in degrees), fix to 90 if ecc==0

        self.batman_params.rp = self.pars['k'].currVal                      #planet radius (in units of stellar radii)

        if not self.use_kipping:
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

        else:
            # Need to convert from q1 and q2 to u1 and u2
            u1 = 2*np.sqrt(self.pars['u1'].currVal)*self.pars['u2'].currVal
            u2 = np.sqrt(self.pars['u1'].currVal)*(1-2*self.pars['u2'].currVal)
            gamma = [u1,u2]

        self.batman_params.u = gamma                #limb darkening coefficients [u1, u2, ..,]

        if self.ld_law == "squareroot": # change to match Batman's naming
            self.batman_params.limb_dark = "square-root"       #limb darkening model
        else:
            self.batman_params.limb_dark = self.ld_law

        self.batman_model = batman.TransitModel(self.batman_params, time_array, nthreads=1)    #initializes model

    def calc(self,time=None,sys_model_inputs=None):

        """Calculates and returns the evaluated Mandel & Agol transit model, using batman.

        Inputs:
        time - the array of times at which to evaluate the model. Can be left blank if this has not changed from the initial init call.
        sys_model_inputs - the array of inputs to give the polynomials if fitting with polys. Can be left blank if this has not changed from the initial init call or you're not using polynomials.

        Returns:
        transitShape - the modelled transit light curve"""

        if self.white_light_fit:
            self.batman_params.t0 = self.pars['t0'].currVal                       #time of inferior conjunction
            if not self.period_fixed:
                self.batman_params.per = self.pars['period'].currVal                      #orbital period
            if not self.ars_fixed:
                self.batman_params.a = self.pars['aRs'].currVal                       #semi-major axis (in units of stellar radii)
            if not self.inc_fixed:
                self.batman_params.inc = self.pars['inc'].currVal                     #orbital inclination (in degrees)
            if not self.ecc_fixed:   # eccentricity
                self.batman_params.ecc = self.pars['ecc'].currVal
            if not self.omega_fixed:   # longitude of periastron
                self.batman_params.w = self.pars['omega'].currVal

        self.batman_params.rp = self.pars['k'].currVal                      #planet radius (in units of stellar radii)

        # set up limb darkening coefficients
        if not self.use_kipping:
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

        else:
            # Need to convert from q1 and q2 to u1 and u2
            u1 = 2*np.sqrt(self.pars['u1'].currVal)*self.pars['u2'].currVal
            u2 = np.sqrt(self.pars['u1'].currVal)*(1-2*self.pars['u2'].currVal)
            gamma = [u1,u2]

        self.batman_params.u = gamma                #limb darkening coefficients [u1, u2]

        if time is not None:
            if np.any(time != self.time_array): # optionally recalculating batman model if the time array has changed
                self.batman_model = batman.TransitModel(self.batman_params, time, nthreads=1)

        transitShape = self.batman_model.light_curve(self.batman_params)
        model = transitShape

        if self.poly_used: # then we're using a polynomial to fit systematics
            red_noise_poly_model = self.red_noise_poly(time,sys_model_inputs)
            model *= red_noise_poly_model

        if self.exp_ramp_used:
            exponential_ramp_model = self.exponential_ramp(time)
            model *= exponential_ramp_model

        if self.step_func_used:
            step_model = self.step_function(time)
            model *= step_model

        if not self.poly_used and not self.exp_ramp_used and not self.step_func_used: # we're using a normalization constant to offset the transit depth
            model *= self.pars['f'].currVal

        return model


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
                exp_ramp_model += self.pars['r%d'%(i+1)]*np.exp(self.pars['r%d'%(i+2)]*time)
            else:
                exp_ramp_model += self.pars['r%d'%(i+1)].currVal*np.exp(self.pars['r%d'%(i+2)].currVal*time)

        return exp_ramp_model


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
            red_noise_pars = np.array([self.pars['c%d'%i] for i in range(1,self.polynomial_orders.sum()+2)])
        else:
            red_noise_pars = np.array([self.pars['c%d'%i].currVal for i in range(1,self.polynomial_orders.sum()+2)])

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


    def step_function(self,time=None):

        """The function that calculates a step function to help fit out mirror tilt events in JWST data

        Inputs:
        time - the array of times at which to evaluate the step function

        Returns:
        step_model - the evaluated step function
        """

        step_model = np.ones_like(time)

        step_model[:int(self.pars["breakpoint"].currVal)] *= self.pars["step1"].currVal
        step_model[int(self.pars["breakpoint"].currVal):] *= self.pars["step2"].currVal

        ## If wanting to use two break points, use the below
        # step_model[:int(self.pars["breakpoint1"].currVal)] *= self.pars["step1"].currVal
        # step_model[int(self.pars["breakpoint2"].currVal):] *= self.pars["step2"].currVal
        #
        # x_break = np.array([int(self.pars["breakpoint1"].currVal),int(self.pars["breakpoint2"].currVal)])
        # y_break = np.array([step_model[int(self.pars["breakpoint1"].currVal)],step_model[int(self.pars["breakpoint2"].currVal)]])
        # break_poly = np.poly1d(np.polyfit(x_break,y_break,1))
        #
        # step_model[int(self.pars["breakpoint1"].currVal):int(self.pars["breakpoint2"].currVal)] = break_poly(np.arange(int(self.pars["breakpoint1"].currVal),int(self.pars["breakpoint2"].currVal)))

        return step_model


    def lnprior(self,sys_priors=None):
        """The priors for the MCMC are handled here.

        Inputs:
        sys_priors - use this if wanting to place Gaussian priors on the system parameters (k,inc,aRs) for a white light curve fit.
                     If so, this is a list of the literature standard deviations on these parameters and 3x these values are used. Default=None (no priors)

        Returns:
        the evaluated ln(prior) as a float
        """

        # define a variable to track the prior value
        retVal = 0

        if sys_priors is not None or self.white_light_fit:
            # Rp/Rs prior
            if sys_priors["k_prior"] is not None:
                if self.pars['k'].currVal < self.pars['k'].startVal-10*sys_priors["k_prior"] or self.pars['k'].currVal > self.pars['k'].startVal+10*sys_priors["k_prior"]:
                    return -np.inf
                retVal += stats.norm(scale=sys_priors["k_prior"],loc=self.pars['k'].startVal).pdf(self.pars['k'].currVal)

            if self.white_light_fit:
                # period prior
                if not self.period_fixed:
                    if self.pars['period'].currVal < 0:
                        return -np.inf
                    if sys_priors["period_prior"] is not None:
                        retVal += stats.norm(scale=sys_priors["period_prior"],loc=self.pars['period'].startVal).pdf(self.pars['period'].currVal)

                # inclination prior
                if not self.inc_fixed:
                    if self.pars['inc'].currVal > 90  or self.pars['inc'].currVal < self.pars['inc'].startVal - 5:
                        return -np.inf
                    if sys_priors["inc_prior"] is not None:
                        retVal += stats.norm(scale=sys_priors["inc_prior"],loc=self.pars['inc'].startVal).pdf(self.pars['inc'].currVal)

                # a/Rs prior
                if not self.ars_fixed:
                    if self.pars['aRs'].currVal <= 1:
                        return -np.inf
                    if sys_priors["aRs_prior"] is not None:
                        retVal += stats.norm(scale=sys_priors["aRs_prior"],loc=self.pars['aRs'].startVal).pdf(self.pars['aRs'].currVal)

                # ecc prior
                if not self.ecc_fixed:
                    if self.pars['ecc'].currVal > 1 or self.pars['ecc'].currVal < 0:
                        return -np.inf
                    if sys_priors["ecc_prior"] is not None:
                        retVal += stats.norm(scale=sys_priors["ecc_prior"],loc=self.pars['ecc'].startVal).pdf(self.pars['ecc'].currVal)

                # omega / longitude of periastron prior
                if not self.omega_fixed:
                    if self.pars['omega'].currVal > 360 or self.pars['ecc'].currVal < 0:
                        return -np.inf
                    if sys_priors["omega_prior"] is not None:
                        retVal += stats.norm(scale=sys_priors["omega_prior"],loc=self.pars['omega'].startVal).pdf(self.pars['omega'].currVal)

                # t0 prior
                if self.pars['t0'].currVal < self.pars['t0'].startVal-0.1 or self.pars['t0'].currVal > self.pars['t0'].startVal+0.1:
                    return -np.inf
                if sys_priors["t0_prior"] is not None:
                    retVal += stats.norm(scale=sys_priors["t0_prior"],loc=self.pars['t0'].startVal).pdf(self.pars['t0'].currVal)


        if self.pars['k'].currVal < 0. or self.pars['k'].currVal > 0.5: # reject non-physical and non-sensical values
            return -np.inf

        if self.GP_used:

            # white noise kernel priors
            if self.wn_kernel and self.pars['s'].currVal > np.log((self.kernel_priors['max_WN_sigma'])**2) or self.wn_kernel and self.pars['s'].currVal < np.log((self.kernel_priors['min_WN_sigma'])**2): # lower bound is np.log(1ppm**2):
                return -np.inf

            # Priors on GP kernel inputs, defined from the data
            for i in range(1,self.gp_ndim+1):
                if self.kernel_priors['min_A'] <= self.pars['A'].currVal <= self.kernel_priors['max_A'] and \
                   self.kernel_priors['min_lniL_%s'%i] <= self.pars['lniL_%s'%i].currVal <= self.kernel_priors['max_lniL_%s'%i]:
                        retVal += 0

                else:
                    return -np.inf

        if self.poly_used: # priors on polynomial coefficients
            if not self.poly_fixed:
                for i in range(0,self.polynomial_orders.sum()+1):
                    if self.pars['c%d'%(i+1)].currVal > 10 or self.pars['c%d'%(i+1)].currVal < -10:
                        return -np.inf

        if self.exp_ramp_used: # priors on polynomial coefficients
            if not self.exp_ramp_fixed:
                # for i in range(0,self.exp_ramp_components*3):
                for i in range(0,self.exp_ramp_components*2):
                    if self.pars['r%d'%(i+1)].currVal > 1e2 or self.pars['r%d'%(i+1)].currVal < -1e2:
                        return -np.inf

        if not self.poly_used and not self.exp_ramp_used: # if not using a polynomial, this is the prior on the normalization constant
            if self.pars['f'].currVal > 1.5 or self.pars['f'].currVal < 0.5:
                return -np.inf

        # now deal with the limb darkening coefficients
        if not self.use_kipping:
            ld_prior_value = 0
            if self.fix_u1:
                u1 = self.pars['u1']
            else:
                u1 = self.pars['u1'].currVal
                if self.ld_std_priors is not None:
                    ld_prior_value += stats.norm(scale=self.ld_std_priors['u1_prior'],loc=self.pars['u1'].startVal).pdf(u1)

            if self.ld_law != "linear":
                if self.fix_u2:
                    u2 = self.pars['u2']
                else:
                    u2 = self.pars['u2'].currVal
                    if self.ld_std_priors is not None:
                        ld_prior_value += stats.norm(scale=self.ld_std_priors['u2_prior'],loc=self.pars['u2'].startVal).pdf(u2)

            if self.ld_law == "nonlinear":
                if self.fix_u3:
                    u3 = self.pars['u3']
                else:
                    u3 = self.pars['u3'].currVal
                    if self.ld_std_priors is not None:
                        ld_prior_value += stats.norm(scale=self.ld_std_priors['u3_prior'],loc=self.pars['u3'].startVal).pdf(u3)

                if self.fix_u4:
                    u4 = self.pars['u4']
                else:
                    u4 = self.pars['u4'].currVal
                    if self.ld_std_priors is not None:
                        ld_prior_value += stats.norm(scale=self.ld_std_priors['u4_prior'],loc=self.pars['u4'].startVal).pdf(u4)

            if self.ld_law == "quadratic":
                if u1 + u2 < 1 and u1 > 0 and u1 + 2*u2 > 0:
                    retVal += ld_prior_value
                else:
                    return -np.inf

        else:
            if self.pars['u1'].currVal < 0 or self.pars['u1'].currVal > 1 or self.pars['u2'].currVal < 0 or self.pars['u2'].currVal > 1: # priors on q1 and q2 from Kipping paper
                return -np.inf
            else:
                if self.ld_std_priors is not None:
                    retVal += stats.norm(scale=self.ld_std_priors['u1_prior'],loc=self.pars['u1'].startVal).pdf(self.pars['u1'].currVal)
                    retVal += stats.norm(scale=self.ld_std_priors['u2_prior'],loc=self.pars['u2'].startVal).pdf(self.pars['u2'].currVal)

        return retVal

    def construct_gp(self,split=False,compute=False,flux_err=None,sys_model_inputs=None):
        """The function that constructs the GP kernel and GP object.

        Inputs:
        split - True/False - determine whether to split the GP into its component kernels. Useful for plotting. Default=False
        compute - True/False - determine whether to compute the GP. Default=False
        flux_err - the errors on the flux data points, to be added in quadrature to the covariance matrix. Default=None
        sys_model_inputs - the inputs to feed to the GP. Can be left blank if these haven't changed from the init call.

        Returns:
        gp - george.GP object
        gp_split - george.GP objects for each kernel (if split=True)

        """

        gp_split = []

        A2 = self.pars['A'].currVal # log of the amplitude

        for i in range(self.gp_ndim):

            # lniL = log-inverse-length-scale
            lniL = self.pars['lniL_%d'%(i+1)].currVal
            L2 = ( 1./np.exp( lniL ) )**2

            if self.kernel_classes[i] == 'Matern32':
                KERNEL = kernels.Matern32Kernel(L2,ndim=self.gp_ndim,axes=i)
            if self.kernel_classes[i] == 'ExpSquared':
                KERNEL = kernels.ExpSquaredKernel(L2,ndim=self.gp_ndim,axes=i)
            if self.kernel_classes[i] == 'RationalQuadratic':
                KERNEL = kernels.RationalQuadraticKernel(log_alpha=1,metric=L2,ndim=self.gp_ndim,axes=i)
            if self.kernel_classes[i] == 'Exp':
                KERNEL = kernels.ExpKernel(L2,ndim=self.gp_ndim,axes=i)


            if split:
                gp_split.append(kernels.ConstantKernel(A2,ndim=self.gp_ndim,axes=i)*KERNEL)

            if i == 0:
                kernel = KERNEL
            else:
                kernel += KERNEL

        if self.wn_kernel:
            WN = self.pars['s'].currVal
            fit_WN = True
        else:
            WN = None
            fit_WN = False

        if self.gp_ndim > 1:
            gp = george.GP(kernels.ConstantKernel(A2,ndim=self.gp_ndim,axes=np.arange(self.gp_ndim))*kernel,white_noise=WN,fit_white_noise=fit_WN,mean=0,fit_mean=False)

        else: # use george's HODLRSolver, which provides faster computation. My tests show this doesn't always seem to perform well for GPs with > 1 kernel.
            gp = george.GP(kernels.ConstantKernel(A2,ndim=self.gp_ndim,axes=np.arange(self.gp_ndim))*kernel,white_noise=WN,fit_white_noise=fit_WN,mean=0,fit_mean=False,solver=george.solvers.HODLRSolver)

        if sys_model_inputs is None:
            gp_model_inputs = self.systematics_model_inputs
        else:
            gp_model_inputs = sys_model_inputs

        if compute:
            if self.gp_ndim > 1:
                gp.compute(gp_model_inputs.T,yerr=flux_err)
            else:
                gp.compute(gp_model_inputs[0],yerr=flux_err)

        if split:
            return gp,gp_split
        else:
            return gp


    def lnlike(self,time,flux,flux_err,sys_model_inputs=None,typeII=False):
        """The log likelihood

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the GP. Can be left blank if these have not changed since the initial init call.
        typeII - True/False - define whether we're using a typeII maximum likelihood estimation. Default=False

        Returns:
        gp.lnlikelihood - the log likelihood of the GP evaulated by george
        """

        if self.GP_used:
            if typeII:
                gp = self.starting_gp_object
            else:
                gp = self.construct_gp()

            if sys_model_inputs is None:
                gp_model_inputs = self.systematics_model_inputs
            else:
                gp_model_inputs = sys_model_inputs

            if self.gp_ndim > 1:
                gp.compute(gp_model_inputs.T,flux_err)
            else:
                gp.compute(gp_model_inputs[0],flux_err)
        else:
            n = len(flux)
            return -0.5*(n*np.log(2*np.pi) + np.sum(np.log(flux_err**2)) + np.sum(((flux-self.calc(time))**2)/(flux_err**2)))

        return gp.lnlikelihood(flux-self.calc(time),quiet=True)


    def lnprob(self,time,flux,flux_err,sys_model_inputs=None,sys_priors=None,typeII=False):
        """A self evaluation of the log probability, given as the prior ln likelihood + the GP ln likelihood.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the GP. Can be left blank if these have not changed since the initial init call.
        sys_priors - define the priors on [k,aRs,inc] if using them and fitting a white light curve. Default=None (no prior)
        typeII - True/False - define whether we're using a typeII maximum likelihood estimation. Default=False

        Returns:
        the evaulated ln probability
        """
        lnp = self.lnprior(sys_priors)
        if np.isfinite(lnp):
            return lnp + self.lnlike(time,flux,flux_err,sys_model_inputs,typeII)
        else:
            return lnp


    def calc_gp_component(self,time,flux,flux_err,sys_model_inputs=None,deconstruct_gp=False):
        """The function that generates the systematics (red) noise model using the GP.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the GP. Can be left blank if these have not changed since the initial init call.
        deconstruct_gp - True/False - use this if wanting to return the systematics model for each component (kernel) of the GP

        Returns:
        mu - the mean systematics model
        std - the standard deviation of the systematics model
        mu_components - list of [mu_i,std_i] for i in range(1,nkernels+1)
        """

        mean_function = self.calc(time)

        if sys_model_inputs is None:
            gp_model_inputs = self.systematics_model_inputs
        else:
            gp_model_inputs = sys_model_inputs

        if deconstruct_gp:
            gp,kernels = self.construct_gp(split=True,compute=True,flux_err=flux_err)
            mu_components = [gp.predict(flux-mean_function,gp_model_inputs.T, return_cov=False, kernel=k) for k in kernels]

        else:
            gp = self.construct_gp(compute=True,flux_err=flux_err)

        predictions = gp.predict(flux-mean_function,gp_model_inputs.T)

        mu = predictions[0]
        cov = predictions[1]
        std = np.sqrt(np.diag(cov))

        if deconstruct_gp:
            return mu,std,mu_components
        else:
            return mu,std

    def chisq(self,time,flux,flux_err,sys_model_inputs=None):

        """Evaluate the chi2 of the object.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the polynomial/GP. Can be left blank if these have not changed since the initial init call or you're not using a polynomial.

        Returns:
        evaluated chi squared
        """
        if self.GP_used:
            mu, std = self.calc_gp_component(time,flux,flux_err,sys_model_inputs)
            resids = (flux - self.calc(time,sys_model_inputs) - mu)/flux_err
        else:
            resids = (flux - self.calc(time,sys_model_inputs))/flux_err

        return np.sum(resids*resids)

    def reducedChisq(self,time,flux,flux_err,sys_model_inputs=None):
        """Evaluate the reduced chi2 of the object.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the polynomial/GP. Can be left blank if these have not changed since the initial init call or you're not using a polynomial.

        Returns:
        the evaluated reduced chi squared"""
        return self.chisq(time,flux,flux_err,sys_model_inputs) / (len(flux) - self.npars)


    def rms(self,time,flux,flux_err=None,sys_model_inputs=None):
        """Evaluate the RMS of the residuals

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the polynomial/GP. Can be left blank if these have not changed since the initial init call or you're not using a polynomial.

        Returns:
        the evaluated RMS of the residuals"""

        if self.GP_used:
            mu, std = self.calc_gp_component(time,flux,flux_err,sys_model_inputs)
            resids = (flux - self.calc(time,sys_model_inputs) - mu)
        else:
            resids = (flux - self.calc(time,sys_model_inputs))

        rms = np.sqrt(np.square(resids).mean())

        return rms

    def BIC(self,time,flux,flux_err,sys_model_inputs=None):
        """Evaluate the Bayesian Information Criterion of the object.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the polynomial/GP. Can be left blank if these have not changed since the initial init call or you're not using a polynomial.

        Returns:
        the evaluated BIC"""

        return  self.npars * np.log(len(flux)) - 2 * self.lnlike(time,flux,flux_err,sys_model_inputs)


    def AIC(self,time,flux,flux_err,sys_model_inputs=None):
        """Evaluate the Akaike Information Criterion of the object.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the polynomial/GP. Can be left blank if these have not changed since the initial init call or you're not using a polynomial.

        Returns:
        the evaluated AIC"""

        return  2*self.npars - 2 * self.lnlike(time,flux,flux_err,sys_model_inputs)


    def red_noise_beta(self,time,flux,flux_err,sys_model_inputs=None):
        """Evaluate the red noise beta factor of the residuals

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        sys_model_inputs - the array of values/inputs to feed to the polynomial/GP. Can be left blank if these have not changed since the initial init call or you're not using a polynomial.

        Returns:
        the evaluated RMS of the residuals"""

        if self.GP_used:
            mu, std = self.calc_gp_component(time,flux,flux_err,sys_model_inputs)
            resids = (flux - self.calc(time,sys_model_inputs) - mu)
        else:
            resids = (flux - self.calc(time,sys_model_inputs))

        rms = np.sqrt(np.square(resids).mean())

        time_diff = np.diff(time).min()*24*60 # in mins
        max_points = int(np.round(30/time_diff)) # go up to maximum of 30 minute bins - note this can be too long, so trust the beta factors in plot_output.py over these ones here
        bins = np.linspace(time[0],time[-1],int(len(time)/max_points))
        binned_x,binned_y,binned_e = pu.rebin(bins,time,resids,flux_err,weighted=False)
        max_rms = np.sqrt(np.nanmean(np.square(binned_y)))

        gaussian_white_noise = np.array(1/np.sqrt([1,max_points]))
        offset = max(gaussian_white_noise)/rms
        gaussian_white_noise = gaussian_white_noise/offset
        beta_factor = max(max_rms/gaussian_white_noise) # maximum ratio of measured dispersion to theortical value, https://iopscience.iop.org/article/10.1086/589737/pdf

        return beta_factor


    def optimise_params(self,time,flux,flux_err,reset_starting_gp=False,contact1=None,contact4=None,full_model=False,sys_priors=None,verbose=True,LM_fit=False):
        """Function to optimise the parameters of the model. Either just for the GP hyperparams (default) using the out of transit data or the full transit model.

        Inputs:
        time - the array of times at which to evaluate the model
        flux - the flux data points
        flux_err - the error in the flux data points
        reset_starting_gp - True/False - use this to overwrite the GP hyperparameter starting values that the object was initialised with, with the optimised hyperparameters.
                            This is not strictly necessary as the object's current values will be equal to the optimised values. Default=False.
        contact1 - location of the transit's first contact point, in data points. Necessary to determine where the out of transit data are. Default=None
        contact4 - location of the transit's fourth contact point, in data points. Necessary to determine where the out of transit data are. Default=None
        full_model - True/False - are we optimising the full model (marginalising over *all* parameters)? Default=False
        verbose - True/False - print output of Nelder-Mead to screen? Default = True
        LM_fit - True/False - if using a Levenberg-Marquardt algorithm, we don't use a Nelder-Mead and we instead use this to optimize and estimate our uncertainties!

        Returns:
        gp.get_parameter_vector() - the optimised values for the GP hyperparameters

        """

        if not full_model: # then we are only optimising the GP hyperparams
            if contact1 is not None:
                evaluated_model = self.calc(time)
                evaluated_model = np.hstack((evaluated_model[:contact1],evaluated_model[contact4:]))

                time = np.hstack((time[:contact1],time[contact4:]))
                flux = np.hstack((flux[:contact1],flux[contact4:]))
                error = np.hstack((flux_err[:contact1],flux_err[contact4:]))

                kern_inputs = np.array([np.hstack((ki[:contact1],ki[contact4:])) for ki in self.systematics_model_inputs])

            else:
                error = flux_err
                kern_inputs = self.systematics_model_inputs
                evaluated_model = self.calc(time)

            y = flux - evaluated_model

        else:
            y = flux
            error = flux_err
            kern_inputs = self.systematics_model_inputs
            evaluated_model = self.calc(time)

        if self.GP_used:
            gp = self.construct_gp()

            if self.gp_ndim > 1:
                gp.compute(kern_inputs.T,error)
            else:
                gp.compute(kern_inputs[0],error)

            # Print the initial ln-likelihood.
            if verbose:
                print('Initial ln-likelihood = ',gp.lnlikelihood(y))
                print('Initial GP hyperparams = ',gp.get_parameter_vector())

        if full_model or self.GP_used is False:
            p0 = extract_model_values(self)
        else:
            p0 = gp.get_parameter_vector()

        bnds = []

        if full_model or self.GP_used is False:
            if self.white_light_fit:
                bnds += [(self.pars['t0'].currVal-0.1,self.pars['t0'].currVal+0.1)]
                if not self.inc_fixed:
                    bnds += [(self.pars['inc'].currVal-5,90)]
                if not self.ars_fixed:
                    bnds += [(self.pars['aRs'].currVal-10,self.pars['aRs'].currVal+10)]
                if not self.period_fixed:
                    bnds += [(self.pars['period'].currVal-0.1,self.pars['period'].currVal+0.1)]
                if not self.ecc_fixed:
                    bnds += [(0,1)]
                if not self.omega_fixed:
                    bnds += [(0,180)]

            bnds += [(0,self.pars['k'].currVal+0.5)]

            if not self.fix_u1:
                bnds += [(self.pars['u1'].currVal-0.2,self.pars['u1'].currVal+0.2)] # uncomment this line when not running on test PRISM data!
                # bnds += [(-2,2)] # comment this line when not running on test PRISM data!
            if not self.fix_u2:
                bnds += [(self.pars['u2'].currVal-0.2,self.pars['u2'].currVal+0.2)] # uncomment this line when not running on test PRISM data!
                # bnds += [(-2,2)] # comment this line when not running on test PRISM data!
            if self.ld_law == 'nonlinear':
                if not self.fix_u3:
                    bnds += [(self.pars['u3'].currVal-0.2,self.pars['u3'].currVal+0.2)]
                if not self.fix_u4:
                    bnds += [(self.pars['u4'].currVal-0.2,self.pars['u4'].currVal+0.2)]

        if self.GP_used:
            if self.wn_kernel:
                bnds += [(np.log((self.kernel_priors['min_WN_sigma'])**2),np.log((self.kernel_priors['max_WN_sigma'])**2))]

            # extract the bounds on the GP red noise hyperparams from the kernel_priors dict
            names = ['min_A','max_A']
            for i in range(self.gp_ndim):
                 names += ['min_lniL_%d'%(i+1)]
                 names += ['max_lniL_%d'%(i+1)]

            bnds += [(self.kernel_priors[names[i]],self.kernel_priors[names[i+1]]) for i in range(0,len(names),2) if names[i] in self.kernel_priors]

        if full_model and self.poly_used:
            if not self.poly_fixed:
                # put priors on the polynomical parameters
                bnds += [(0,10)] # this is the y intercept
                bnds += [(-5,5)]*self.polynomial_orders.sum() # these are the coefficients of the polynomial

        if full_model and self.exp_ramp_used:
            if not self.exp_ramp_fixed:
                # put priors on the polynomical parameters
                bnds += [(-100,100)]*self.exp_ramp_components*2 # these are the coefficients of the expoential ramp

        if full_model and self.step_func_used:
            bnds += [(0.9,1.1)]*2 # these are the normalisation constants of the step function
            bnds += [(0,len(time))] # this is the breakpoint of the step function

        if full_model and not self.poly_used and not self.exp_ramp_used and not self.step_func_used:
            bnds += [(0.5,2)] # this is the bound on the normalization constant that we use if we don't have a polynomial

        # Now if we're fitting the full model (all parameters) or we're not using a GP we perform this step
        if full_model or self.GP_used is False:
            if verbose:
                if LM_fit:
                    print("\n ...running Levenberg-Marquardt \n")
                else:
                    print("\n Running Nelder-Mead")
                disp = True
            else:
                disp = False

            if not LM_fit:
                results = optimize.minimize(nll, p0,args=(self,y,True,time,flux_err,sys_priors,False),method='Nelder-Mead',bounds=tuple(bnds),options=dict(maxiter=1e4,disp=disp))
            else:
                results = optimize.least_squares(nll, p0,args=(self,y,True,time,flux_err,sys_priors,False,True),method='lm')

            update_model(self,results.x)

            if LM_fit:
                J = results.jac
                try:
                    cov = np.linalg.inv(J.T.dot(J))*self.reducedChisq(time,flux,flux_err)
                    uncertainties = np.sqrt(np.diagonal(cov))
                except:
                    print("Unable to estimate uncertainties from covariance matrix")
                    uncertainties = np.zeros_like(results.x)

                return self,results.x,uncertainties

            else:
                return self,results.x

        # Otherwise, we're fitting only the GP hyperparameters
        else:
            results = optimize.minimize(nll, p0, jac=grad_nll,args=(gp,y),method='L-BFGS-B',bounds=tuple(bnds))

        if verbose:
            print('Final ln-likelihood = ',gp.lnlikelihood(y))
            print("kernel = ",gp.kernel)

        # update the values of the GP hyperparameters in the GP model
        if reset_starting_gp:
            if self.wn_kernel:
                self.pars['s'].currVal = results.x[0]
                self.pars['A'].currVal = results.x[1]
                for i in range(self.gp_ndim):
                   self.pars['lniL_%d'%(i+1)].currVal = results.x[2+i]
            else:
                self.pars['A'].currVal = results.x[0]
                for i in range(self.gp_ndim):
                   self.pars['lniL_%d'%(i+1)].currVal = results.x[1+i]

            self.starting_gp_object = self.construct_gp(compute=True,flux_err=flux_err)

        if verbose:
            print('Kernel params = ',self.starting_gp_object.get_parameter_vector())

        return gp.get_parameter_vector()


    # Parameters to set and update values within the object
    def __getitem__(self,ind):
            return self.data[ind].currVal
    def __setitem__(self,ind,val):
            self.data[ind].currVal = val
    def __delitem__(self,ind):
            self.data.remove(ind)
    def __len__(self):
            return len(self.data)

    @property
    def npars(self):
        return len(self.data)

class Param(object):
    '''A Param (parameter) needs a starting value and a current value. However, when first defining the Param object, it takes the starting value as the current value.

    Inputs:
    startVal: the starting value for the parameter

    Returns:
    Param object'''
    def __init__(self,startVal):
        self.startVal = startVal
        self.currVal  = startVal # this is only the starting value when the parameter is first initialised - it gets updated within the TransitModel class

def update_model(model,fit_results):
    """Update the model parameters with the result from a model fit.

    Inputs:
    model - the model (TransitModel class object) which needs to be updated
    fit_results - the new parameter values

    Returns:
    model - the inputted model with updated parameter values"""

    for i,j in enumerate(fit_results):
        model[i] = j
    return model

def extract_model_values(model,typeII=False):
    """Extract the free parameters of a TransitModel object.

    Inputs:
    model - the TransitModel object
    typeII - True/False - is this a typeII maximum likelihood fit

    Returns:
    values - the extracted free parameters' values"""

    values = np.array([v for v in model])

    if typeII: # don't return GP hyperparameters as these are fixed
        return values[:-self.gp_npars]

    return values



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
