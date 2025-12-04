import numpy as np
from fitting_utils import parametric_fitting_functions as pf
from fitting_utils.LightcurveModel import Param

import george
from george import kernels

class GPModel(object):
    """
    Encapsulates exponential ramps, polynomial red-noise trends,
    and step functions for systematics modeling.
    """

    def __init__(self,param_dict,
                 GP_model_inputs, time_array, flux, flux_error):

        """
        The SystematicsModel model class.

        Inputs:
        param_dict - the dictionary of the planetary and systematics parameters
        GP_model_inputs - the dictionary of inputs for each GP model
        time_array - time array

        Returns:
        SystematicsModel object
        """

        self.param_dict = param_dict
        self.GP_model_inputs = GP_model_inputs
        self.GP_kernel_inputs = GP_model_inputs['model_inputs']
        self.flux_error = flux_error
        self.flux = flux
        self.GP_used = True
        
        self.time = time_array
        self.kernel_classes = self.GP_model_inputs['kernel_classes']
        self.gp_ndim = len([c for c in self.kernel_classes if c is not None])


        self.wn_kernel = self.GP_model_inputs['white_noise_kernel']
            

        self.gp = self.construct_gp()


    def construct_gp(self,split=False,compute=False,flux_err=None):
        """The function that constructs the GP kernel and GP object.

        Inputs:
        split - True/False - determine whether to split the GP into its component kernels. Useful for plotting. Default=False
        compute - True/False - determine whether to compute the GP. Default=False
        flux_err - the errors on the flux data points, to be added in quadrature to the covariance matrix. Default=None

        Returns:
        gp - george.GP object
        gp_split - george.GP objects for each kernel (if split=True)

        """

        gp_split = []

        if type(self.param_dict['A']) is Param:
            A2 = self.param_dict['A'].currVal # log of the amplitude
        else:
            A2 = self.param_dict['A'] # log of the amplitude


        for i in range(self.gp_ndim):

            # lniL = log-inverse-length-scale
            if type(self.param_dict['lniL_%d'%(i+1)]) is Param:
                lniL = self.param_dict['lniL_%d'%(i+1)].currVal
            else:
                lniL = self.param_dict['lniL_%d'%(i+1)]
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
            WN = self.param_dict['s'].currVal
        else:
            WN = None

        if self.gp_ndim > 1:
            gp = george.GP(kernels.ConstantKernel(A2,ndim=self.gp_ndim,axes=np.arange(self.gp_ndim))*kernel,white_noise=WN,fit_white_noise=self.wn_kernel,mean=0,fit_mean=False)

        else: # use george's HODLRSolver, which provides faster computation. My tests show this doesn't always seem to perform well for GPs with > 1 kernel.
            gp = george.GP(kernels.ConstantKernel(A2,ndim=self.gp_ndim,axes=np.arange(self.gp_ndim))*kernel,white_noise=WN,fit_white_noise=self.wn_kernel,mean=0,fit_mean=False,solver=george.solvers.HODLRSolver)

        if compute:
            if flux_err is None:
                err = self.flux_error
            else:
                err = flux_err

            if self.gp_ndim > 1:
                gp.compute(self.GP_kernel_inputs.T,yerr=err)
            else:
                gp.compute(self.GP_kernel_inputs[0],yerr=err)

        if split:
            return gp,gp_split
        else:
            return gp

    def update_model(self, new_param_dict):
        self.param_dict = new_param_dict
        if self.param_dict['infl_err'] is Param:
            self.gp = self.construct_gp(flux_err=self.flux_err*self.param_dict['infl_err'].currVal)
        else:
            self.gp = self.construct_gp()
        return


    def calc(self,model_calc,time=None,flux=None,flux_err=None,kernel_inputs=None,deconstruct_gp=False):
        """The function that generates the systematics (red) noise model using the GP.

        Inputs:
        time - the array of times at which to evaluate the model, Can be left blank if these have not changed since the initial init call.
        flux - the flux data points, Can be left blank if these have not changed since the initial init call.
        flux_err - the error in the flux data points, Can be left blank if these have not changed since the initial init call.
        kernel_inputs - the array of values/inputs to feed to the GP. Can be left blank if these have not changed since the initial init call.
        deconstruct_gp - True/False - use this if wanting to return the systematics model for each component (kernel) of the GP

        Returns:
        mu - the mean systematics model
        std - the standard deviation of the systematics model
        mu_components - list of [mu_i,std_i] for i in range(1,nkernels+1)
        """

        if time is None:
            time = self.time
        if flux is None:
            flux = self.flux
        if flux_err is None:
            flux_err = self.flux_error

        mean_function = model_calc

        if kernel_inputs is None:
            gp_model_inputs = self.GP_kernel_inputs
        else:
            gp_model_inputs = kernel_inputs

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

    def lnlike(self, model_calc,flux_err):
        """The log likelihood
    
        Returns:
        gp.lnlikelihood - the log likelihood of the GP evaulated by george
        """
        if self.gp_ndim > 1:
            self.gp.compute(self.gp_model_inputs.T,flux_err)
        else:
            self.gp.compute(self.gp_model_inputs[0],flux_err)

        return self.gp.lnlikelihood(self.flux-model_calc,quiet=True)
