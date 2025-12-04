#### Author of this code: Eva-Maria Ahrer, adapted from Tiberius TransitGPPM model (author: J. Kirk)

import numpy as np
import catwoman


class CatwomanModel(object):
    def __init__(self,param_dict,param_list_free,transit_model_inputs,time_array,cw_fac=None):

        """
        

        Inputs:
        param_dict                   - the dictionary of the planet's transit parameters
        param_list_free              - list of free parameters
        transit_model_inputs         - inputs to build the model, e.g., whether to use kipping parameterisation
        cw_fac            - scaling factor for catwoman to make it run faster
        
        

        Can return:
        - model calculated on a certain time array
        - ... 
        """

        self.param_dict = param_dict
        self.param_list_free = param_list_free
        self.transit_model_inputs = transit_model_inputs
        self.time_array = time_array
        self.use_kipping = transit_model_inputs['use_kipping']
        self.catwoman_fac = cw_fac

        ##### Catwoman initialisation - note this is first outside of model calculation as it is the fastest way

        self.catwoman_params = catwoman.TransitParams()

        all_params = list(param_dict.keys())
        ld_list = ['u1', 'u2', 'u3', 'u4']

        u = []
        for i in range(len(all_params)):
            if all_params[i] in param_list_free and all_params[i] not in ld_list:
                setattr(self.catwoman_params, all_params[i], self.param_dict[all_params[i]].currVal)
            if all_params[i] not in param_list_free and all_params[i] not in ld_list:
                setattr(self.catwoman_params, all_params[i], self.param_dict[all_params[i]])
        
        if not self.use_kipping:
            # for getting the limb-darkening as one array
            for i in range(len(ld_list)):
                if ld_list[i] in param_list_free:
                    u.append(self.param_dict[ld_list[i]].currVal)
                if ld_list[i] not in param_list_free and ld_list[i] in all_params:
                    u.append(self.param_dict[ld_list[i]])
            self.catwoman_params.limb_dark = transit_model_inputs['ld_law'] 
            self.catwoman_params.u = u 
        else:
            u1 = 2*np.sqrt(self.param_dict['u1'].currVal)*self.param_dict['u2'].currVal
            u2 = np.sqrt(self.param_dict['u1'].currVal)*(1-2*self.param_dict['u2'].currVal)
            self.catwoman_params.limb_dark = 'quadratic'
            self.catwoman_params.u = [u1, u2]

        if 'rp2' not in all_params:
            self.catwoman_params.rp2 = None
            self.catwoman_params.phi = 0.
    
        # initalise
        self.catwoman_model = catwoman.TransitModel(self.catwoman_params, self.time_array,fac=self.catwoman_fac)    #initializes model

    def update_model(self, new_param_dict):
        self.param_dict = new_param_dict
        all_params = new_param_dict.keys()

        u = []
        for i in range(len(all_params)):
            if all_params[i] in param_list_free and all_params[i] not in ld_list:
                setattr(self.catwoman_params, all_params[i], self.param_dict[all_params[i]].currVal)
            if all_params[i] not in param_list_free and all_params[i] not in ld_list:
                setattr(self.catwoman_params, all_params[i], self.param_dict[all_params[i]])
        
        # for getting the limb-darkening as one array
        if not self.use_kipping:
            for i in range(len(ld_list)):
                if ld_list[i] in param_list_free:
                    u.append(self.param_dict[ld_list[i]].currVal)
                if ld_list[i] not in param_list_free and ld_list[i] in all_params:
                    u.append(self.param_dict[ld_list[i]])
            self.catwoman_params.limb_dark = transit_model_inputs['ld_law'] 
            self.catwoman_params.u = u 
        else:
            u1 = 2*np.sqrt(self.param_dict['u1'].currVal)*self.param_dict['u2'].currVal
            u2 = np.sqrt(self.param_dict['u1'].currVal)*(1-2*self.param_dict['u2'].currVal)
            self.catwoman_params.u = [u1, u2]

        self.catwoman_model = catwoman.TransitModel(self.catwoman_params, self.time_array,fac=self.catwoman_fac)    #initializes model
        return
        


    def calc(self,time_array=None, overwrite=False):

        """Calculates and returns the evaluated Mandel & Agol transit model, using catwoman.

        Inputs:
        time - the array of times at which to evaluate the model. Can be left blank if this has not changed from the initial init call.
        overwrite  - if the catwoman model should be overwritten using the new time array

        Returns:
        model - the modelled catwoman light curve"""
        if time_array is not None:
            new_model = catwoman.TransitModel(self.catwoman_params, time_array,fac=self.catwoman_fac)    # model
        if time_array is not None and overwrite == True:
            self.catwoman_model = catwoman.TransitModel(self.catwoman_params, time_array,fac=self.catwoman_fac)    #if we want to continue using that time for the model
            self.time_array = time_array
        model = new_model.light_curve(self.catwoman_params)
        return model

