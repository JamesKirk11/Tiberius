import numpy as np
from fitting_utils import parametric_fitting_functions as pf
from fitting_utils.LightcurveModel import Param
from fleck import Star

class FleckModel:
    """
    Encapsulates exponential ramps, polynomial red-noise trends,
    and step functions for systematics modeling.
    """

    def __init__(param_dict, time_array, param_list_free, transit_model_inputs):

        """
        The SystematicsModel model class.

        Inputs:
        param_dict - the dictionary of the planetary and systematics parameters
        time_array - time array
        param_list_free  - list of free parameters
        transit_model_inputs - transit model inputs, e.g., whether to use Kipping parameterisation, limb-darkening law, etc.


        Returns:
        FleckModel object, is a transit light curve
        """

        self.param_dict = param_dict
        self.time = time_array
        self.param_list_free = param_list_free
        self.transit_model_inputs = transit_model_inputs
        self.use_kipping = transit_model_inputs['use_kipping']
        self.inc_stellar = transit_model_inputs['inc_stellar']
        ld_list = ['u1', 'u2', 'u3', 'u4']

        # read in fleck parameters
        self.fleck_parameters = ['stellar_rotation_period', 'spot_contrast', 'spot_lon', 'spot_lat', 'spot_radius']

        for i in self.fleck_parameters:
            if self.param_dict[i] is Param:
                setattr(self, i, self.param_dict[i].currVal)
            else:
                setattr(self, i, self.param_dict[i])
        

        ##### Batman initialisation - note this is first outside of model calculation as it is the fastest way

        self.batman_params = batman.TransitParams()

        # get planet parameters
        all_params = param_dict.keys()
        u = []
        for i in range(len(all_params)):
            if all_params[i] in param_list_free and all_params[i] not in ld_list and all_params[i] not in fleck_parameters:
                setattr(self.batman_params, all_params[i], self.param_dict[all_params[i]].currVal)
            if all_params[i] not in param_list_free and all_params[i] not in ld_list and all_params[i] not in fleck_parameters:
                setattr(self.batman_params, all_params[i], self.param_dict[all_params[i]])
        
        # for getting the limb-darkening as one array
        self.fit_ld = False
        if not self.use_kipping:
            for i in range(len(ld_list)):
                if ld_list[i] in param_list_free:
                    u.append(self.param_dict[ld_list[i]].currVal)
                    self.fit_ld = True
                if ld_list[i] not in param_list_free and ld_list[i] in all_params:
                    u.append(self.param_dict[ld_list[i]])
            self.batman_params.limb_dark = transit_model_inputs['ld_law'] 
            self.batman_params.u = u 
        else:
            u1 = 2*np.sqrt(self.param_dict['u1'].currVal)*self.param_dict['u2'].currVal
            u2 = np.sqrt(self.param_dict['u1'].currVal)*(1-2*self.param_dict['u2'].currVal)
            self.fit_ld = True
            self.batman_params.limb_dark = 'quadratic'
            self.batman_params.u = [u1, u2]

        # initialise star model
        self.star = Star(spot_contrast=self.spot_contrast, u_ld=self.batman_params.u, rotation_period=self.rotation_period)
        self.transit_with_spot_lightcurve = self.star.light_curve(self.spot_lon, self.spot_lat, self.spot_radius,\
                                                                    self.inc_stellar,planet=self.batman_params,\
                                                                    times=self.time_array)


    def update_model(self, new_param_dict):
        self.param_dict = new_param_dict
        all_params = new_param_dict.keys()

        for i in self.fleck_parameters:
            if self.param_dict[i] is Param:
                setattr(self, i, self.param_dict[i].currVal)
            else:
                setattr(self, i, self.param_dict[i])

        u = []
        for i in range(len(all_params)):
            if all_params[i] in param_list_free and all_params[i] not in ld_list and all_params[i] not in fleck_parameters:
                setattr(self.batman_params, all_params[i], self.param_dict[all_params[i]].currVal)
            if all_params[i] not in param_list_free and all_params[i] not in ld_list and all_params[i] not in fleck_parameters:
                setattr(self.batman_params, all_params[i], self.param_dict[all_params[i]])
        
        # for getting the limb-darkening as one array
        if not self.use_kipping:
            for i in range(len(ld_list)):
                if ld_list[i] in param_list_free:
                    u.append(self.param_dict[ld_list[i]].currVal)
                if ld_list[i] not in param_list_free and ld_list[i] in all_params:
                    u.append(self.param_dict[ld_list[i]])
            self.batman_params.limb_dark = transit_model_inputs['ld_law'] 
            self.batman_params.u = u 
        else:
            u1 = 2*np.sqrt(self.param_dict['u1'].currVal)*self.param_dict['u2'].currVal
            u2 = np.sqrt(self.param_dict['u1'].currVal)*(1-2*self.param_dict['u2'].currVal)
            self.batman_params.u = [u1, u2]

        if self.param_dict['spot_contrast'] is Param or self.param_dict['rotation_period'] is Param or self.fit_ld == True:
            self.star = Star(spot_contrast=self.spot_contrast, u_ld=self.batman_params.u, rotation_period=self.rotation_period)

        self.transit_with_spot_lightcurve = self.star.light_curve(self.spot_lon, self.spot_lat, self.spot_radius,\
                                                                    self.inc_stellar,planet=self.batman_params,\
                                                                    times=self.time_array)
        return



    def calc(self,time_array=None, overwrite=False):

        """Calculates and returns the evaluated Mandel & Agol transit model, using batman.

        Inputs:
        time_array - the array of times at which to evaluate the model. Can be left blank if this has not changed from the initial init call.
        overwrite  - if the batman model should be overwritten using the new time array

        Returns:
        transitShape - the modelled transit light curve"""

        if time_array is not None:
            transit_with_spot_lightcurve = self.star.light_curve(self.spot_lon, self.spot_lat, self.spot_radius,\
                                                 self.inc_stellar,planet=self.batman_params,times=time_array)
            return transit_with_spot_lightcurve
        if time_array is not None and overwrite == True:
            self.transit_with_spot_lightcurve = self.star.light_curve(self.spot_lon, self.spot_lat, self.spot_radius,\
                                                 self.inc_stellar,planet=self.batman_params,times=time_array)
            self.time_array = time_array

            return self.transit_with_spot_lightcurve

