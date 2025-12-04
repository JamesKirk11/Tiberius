#### Author of this code: James Kirk & Eva-Maria Ahrer
import numpy as np
import batman


class BatmanModel(object):
    def __init__(self,param_dict,param_list_free,transit_model_inputs,time_array):

        """
        The transit model class, which uses batman to generate the analytic, quadratically limb-darkened transit light curves, and george to generate the GP red noise models.
        However, this has the added option of fitting the time dependence with a polynomial, removing it as a parameter given to the GP. The thought behind this is that the GP has less to do, leading to smaller uncertainties in Rp/Rs.

        Inputs:
        aram_dict                   - the dictionary of the planet's transit parameters
        param_list_free              - list of free parameters
        transit_model_inputs         - inputs to build the batman model

        Returns:
        TransitModel object
        """

        self.param_dict = param_dict
        self.param_list_free = param_list_free
        self.transit_model_inputs = transit_model_inputs
        self.time_array = time_array
        self.use_kipping = transit_model_inputs['use_kipping']
        ld_list = ['u1', 'u2', 'u3', 'u4']

        ##### Batman initialisation - note this is first outside of model calculation as it is the fastest way

        self.batman_params = batman.TransitParams()

        all_params = list(param_dict.keys())
        u = []
        for i in range(len(all_params)):
            if all_params[i] in self.param_list_free and all_params[i] not in ld_list:
                setattr(self.batman_params, all_params[i], self.param_dict[all_params[i]].currVal)
            if all_params[i] not in self.param_list_free and all_params[i] not in ld_list:
                setattr(self.batman_params, all_params[i], self.param_dict[all_params[i]])
        
        # for getting the limb-darkening as one array
        if not self.use_kipping:
            for i in range(len(ld_list)):
                if ld_list[i] in self.param_list_free:
                    u.append(self.param_dict[ld_list[i]].currVal)
                if ld_list[i] not in self.param_list_free and ld_list[i] in all_params:
                    u.append(self.param_dict[ld_list[i]])
            self.batman_params.limb_dark = transit_model_inputs['ld_law'] 
            self.batman_params.u = u 
        else:
            u1 = 2*np.sqrt(self.param_dict['u1'].currVal)*self.param_dict['u2'].currVal
            u2 = np.sqrt(self.param_dict['u1'].currVal)*(1-2*self.param_dict['u2'].currVal)
            self.batman_params.limb_dark = 'quadratic'
            self.batman_params.u = [u1, u2]

        self.batman_model = batman.TransitModel(self.batman_params, self.time_array, nthreads=1)    #initializes model
    
    def update_model(self, new_param_dict):
        self.param_dict = new_param_dict
        ld_list = ['u1', 'u2', 'u3', 'u4']

        u = []
        for i in self.param_list_free:
            if i not in ld_list:
                setattr(self.batman_params, i, self.param_dict[i].currVal)
            else:
                u.append(self.param_dict[i].currVal)
        
        # for getting the limb-darkening as one array
        if not self.use_kipping:
            self.batman_params.u = u 
        else:
            u1 = 2*np.sqrt(self.param_dict['u1'].currVal)*self.param_dict['u2'].currVal
            u2 = np.sqrt(self.param_dict['u1'].currVal)*(1-2*self.param_dict['u2'].currVal)
            self.batman_params.u = [u1, u2]

        self.batman_model = batman.TransitModel(self.batman_params, self.time_array, nthreads=1)    #initializes model
        return

    def calc(self,time_array=None, overwrite=False):

        """Calculates and returns the evaluated Mandel & Agol transit model, using batman.

        Inputs:
        time_array - the array of times at which to evaluate the model. Can be left blank if this has not changed from the initial init call.
        overwrite  - if the batman model should be overwritten using the new time array

        Returns:
        transitShape - the modelled transit light curve"""

        if time_array is not None:
            new_model = batman.TransitModel(self.batman_params, time_array)    # model
        if time_array is not None and overwrite == True:
            self.batman_model = batman.TransitModel(self.batman_params, time_array)    #if we want to continue using that time for the model
            self.time_array = time_array
        model = new_model.light_curve(self.batman_params)
        return model


