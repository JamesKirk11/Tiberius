#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from scipy.interpolate import UnivariateSpline as US

from global_utils import parseInput

from fitting_utils import LightcurveModel as lc
from fitting_utils import sampling as s
# from fitting_utils import plotting_utils as pu


parser = argparse.ArgumentParser(description='Run fit to a single light curve that is either a wavelength-binned or white light curve. This makes use of the TransitModelGPPM class, which fits the red noise as a GP + parametric model.')
parser.add_argument('wavelength_bin', help="which wavelength bin are we running the fit to? This is indexed from 0. If running fit to the white light curve, this must be given as '0'",type=int)
parser.add_argument('-dbp',"--determine_best_polynomials", help="Use this option to loop over all combination of polynomial input vectors and orders to determine the best fitting polynomials via a Nelder-Mead. This prevents an MCMC from running. Set this number to the maximum polynomial order you want to consider. e.g. 3 = cubic polys",default=0,type=int)
args = parser.parse_args()


### Load in parameter file

input_dict = parseInput('fitting_input.txt')

try:
    wavelength_centres = float(input_dict['wvl_centres'])
    wvl_bin_full_width = float(input_dict['wvl_bin_full_width'])
    white_light_fit = True
except:
    wavelength_centres = pickle.load(open(input_dict['wvl_centres'],'rb'))
    wvl_bin_full_width = pickle.load(open(input_dict['wvl_bin_full_width'],'rb'))

    nbins = len(wavelength_centres)

wb = args.wavelength_bin

if white_light_fit and wb > 1:
    raise ValueError('if fitting wavelength bins, need to have a wavelength array in fitting_input.txt')

### Plotting controls

rebin_data = input_dict['rebin_data']
if rebin_data is not None:
    rebin_data = int(rebin_data)

show_plots = bool(int(input_dict['show_plots']))
save_plots = bool(int(input_dict['save_plots']))

cwd = os.getcwd()
output_foldername = cwd + '/' + str(input_dict['output_foldername']) + '/'

os.makedirs(output_foldername, exist_ok=True) 
if save_plots:
    os.makedirs(output_foldername + '/Figures', exist_ok=True) 
os.makedirs(output_foldername + '/pickled_objects', exist_ok=True) 

### Load in various input arrays
time = pickle.load(open(input_dict['time_file'],'rb'))

try:
    first_integration = int(input_dict["first_integration"])
    print("\n...Clipping first %d integrations (%d minutes)"%(first_integration,24*60*(time[first_integration]-time[0])))
except:
    first_integration = 0
try:
    last_integration = int(input_dict["last_integration"])
    print("\n...Clipping beyond integration %d (%d minutes)"%(last_integration,24*60*(time[-1]-time[last_integration])))
except:
    last_integration = len(time)

time = time[first_integration:last_integration]

if white_light_fit:
    flux = pickle.load(open(input_dict['flux_file'],'rb'))[first_integration:last_integration]
    flux_error = pickle.load(open(input_dict['error_file'],'rb'))[first_integration:last_integration]
    wb = 0
    print('\n\n## RUNNING FIT TO WHITE LIGHT CURVE')
    single_fit = True

else:

    nfiles = pickle.load(open(input_dict['flux_file'],'rb')).shape[0]

    flux = np.atleast_2d(pickle.load(open(input_dict['flux_file'],'rb')))[wb].astype(float)[first_integration:last_integration]
    flux_error = np.atleast_2d(pickle.load(open(input_dict['error_file'],'rb')))[wb].astype(float)[first_integration:last_integration]

    print('\n\n## RUNNING FIT TO WAVELENGTH BIN %d'%(wb+1))


### Common noise correction using a fit to a white light curve

if input_dict['common_noise_model'] is not None:
    print("applying common mode correction...")
    common_noise_model = pickle.load(open(input_dict['common_noise_model'],'rb'))

    if show_plots:
        plt.figure()
        plt.errorbar(time,flux,yerr=flux_error,fmt='o',alpha=0.5,ecolor='r',color='r',capsize=2,label='Before correction')
        plt.errorbar(time,flux/common_noise_model,yerr=flux_error,fmt='o',ecolor='k',color='k',capsize=2,alpha=0.5,label='After correction')
        plt.xlabel('Time (MJD)')
        plt.ylabel('Normalised flux')
        plt.title('Common mode correction')
        plt.legend(loc='upper left')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    
    if save_plots:
        plt.figure()
        plt.errorbar(time,flux,yerr=flux_error,fmt='o',alpha=0.5,ecolor='r',color='r',capsize=2,label='Before correction',rasterized=True)
        plt.errorbar(time,flux/common_noise_model,yerr=flux_error,fmt='o',ecolor='k',color='k',capsize=2,alpha=0.5,label='After correction',rasterized=True)
        plt.xlabel('Time (MJD)')
        plt.ylabel('Normalised flux')
        plt.title('Common mode correction')
        plt.legend(loc='upper left')
        plt.savefig(output_foldername +'/Figures/Common_mode_correction.png', bbox_inches=True)
        plt.close()

    y = flux

    # Divide by the common noise model
    flux = flux/common_noise_model
    flux_error = (flux_error/y)*flux


fit_models = {}
fit_models['transit_model'] = str(input_dict['transit_model'])
fit_models['systematics_model'] = []

model_inputs = {}
model_inputs['systematic_model'] = {}

### Red noise polynomial model parameters

# define the order of each polynomial fitted to each ancillary data set
if input_dict['polynomial_orders'] is not None:
    fit_models['systematics_model'].append('polynomial')
    model_inputs['systematic_model']['polynomial_orders'] = np.array([int(i) for i in input_dict['polynomial_orders'].split(',')])
    model_input_files = [i.strip() for i in input_dict['model_input_files'].split(',')]

# determine whether we're using an exponential ramp model or not
if bool(int(input_dict['exponential_ramp'])):
   fit_models['systematics_model'].append('exponential_ramp')

# determine whether we're using a step function or not
if bool(int(input_dict['step_function'])):
    fit_models['systematics_model'].append('step_function')


systematics_model_inputs = []
for i in model_input_files:
    model_in = np.atleast_2d(pickle.load(open(i,'rb')))[:,first_integration:last_integration]
    if model_in.shape[0] == 1:
        vector = model_in[0]
        # replace any nans
        vector[~np.isfinite(vector)] = 1e-10
        systematics_model_inputs.append(vector)
    if model_in.shape[0] > 1:
        vector = model_in[wb]
        # replace any nans
        vector[~np.isfinite(vector)] = 1e-10
        systematics_model_inputs.append(model_in[wb])

# Do we want to normalise inputs? Defined as (input - mean(input))/std(input)
norm_inputs = bool(int(input_dict['normalise_inputs']))

if norm_inputs:
    print('standardising model inputs...')
    systematics_model_inputs = np.array([(i-i.mean())/i.std() for i in systematics_model_inputs])
else:
    systematics_model_inputs = np.array(systematics_model_inputs)



### GP controls
if input_dict['kernel_classes'] is not None:
    try:
        kernel_classes = [i.strip() for i in input_dict['kernel_classes'].split(',')]
    except:
        GP_used = False

    model_inputs['GP_model']['kernel_classes'] = kernel_classes
    # are we using a white noise kernel?
    model_inputs['GP_model']['white_noise_kernel'] = bool(int(input_dict['white_noise_kernel']))
    GP_used = True
else:
    GP_used = False


if GP_used:
    GP_model_input_files = [i.strip() for i in input_dict['GP_model_input_files'].split(',')]
    GP_model_inputs = []
    for i in GP_model_input_files:
        model_in = np.atleast_2d(pickle.load(open(i,'rb')))[:,first_integration:last_integration]
        if model_in.shape[0] == 1:
            vector = model_in[0]
            # replace any nans
            vector[~np.isfinite(vector)] = 1e-10
            GP_model_inputs.append(vector)
        if model_in.shape[0] > 1:
            vector = model_in[wb]
            # replace any nans
            vector[~np.isfinite(vector)] = 1e-10
            GP_model_inputs.append(model_in[wb])
    
    norm_GP_inputs = bool(int(input_dict['normalise_GP_inputs']))
    if norm_GP_inputs:
        print('standardising GP model inputs...')
        GP_model_inputs = np.array([(i-i.mean())/i.std() for i in GP_model_inputs])
    else:
        GP_model_inputs = np.array(GP_model_inputs)



## Remove any nans and zeroes from the error array
not_nans = np.isfinite(flux)*np.isfinite(flux_error)
time = time[not_nans]
flux = flux[not_nans]
flux_error = flux_error[not_nans]
zero_errors = flux_error == 0
if np.any(zero_errors):
    flux_error[zero_errors] = np.mean(flux_error)
systematics_model_inputs = systematics_model_inputs[:,not_nans]
if GP_used:
    GP_model_inputs = GP_model_inputs[:,not_nans]



### Optionally clip outliers using running median

clip_outliers = bool(int(input_dict['clip_outliers']))
median_clip = bool(int(input_dict['median_clip']))
sigma_clip = float(input_dict['sigma_cut'])

if clip_outliers and median_clip:
    flux, flux_error, time, keep_idx = clipping_outliers_with_median_clip(flux, flux_error, time, sigma_clip, show_plots, save_plots, output_foldername)
    
    systematics_model_inputs = np.array(systematics_model_inputs)[:,keep_idx].reshape(len(systematics_model_inputs),len(np.where(keep_idx == True)[0]))
    if GP_used:
        GP_model_inputs = np.array(GP_model_inputs)[:,keep_idx].reshape(len(GP_model_inputs),len(np.where(keep_idx == True)[0]))
    pickle.dump(keep_idx,open(output_foldername + '/pickled_objects/' + 'data_quality_flags_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))




### for GP optimisation and variance limits
contact1 = int(input_dict['contact1']) - first_integration
contact4 = int(input_dict['contact4']) - first_integration


## renormalise flux to out-of-transit median?
if bool(int(input_dict['renorm_flux'])):
    print("re-normalising flux array...")
    oot_median = np.nanmedian(np.hstack((flux[:contact1],flux[contact4:])))
    flux /= oot_median
    flux_error /= oot_median


### Save clipped arrays for ease of future plotting
pickle.dump(flux,open(output_foldername + '/pickled_objects/' + 'Used_flux_wb%s.pickle'%(str(wb+1).zfill(4)),'wb')) # add '0' in front of single digit wavelength bin numbers so that linux sorts them properly
pickle.dump(time,open(output_foldername + '/pickled_objects/' + 'Used_time_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))
pickle.dump(flux_error,open(output_foldername + '/pickled_objects/' + 'Used_error_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))


model_inputs['systematic_model']['model_inputs'] = systematics_model_inputs
pickle.dump(systematics_model_inputs,open(output_foldername + '/pickled_objects/' + 'Used_model_inputs_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))

if GP_used:
    model_inputs['GP_model']['model_inputs'] = GP_model_inputs
    pickle.dump(GP_model_inputs,open(output_foldername + '/pickled_objects/' + 'Used_GP_model_inputs_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))

   


prior_file = str(input_dict['prior_filename'])

# initalise light curve model
lc_class = lc.LightcurveModel(flux,flux_error,time,prior_file,fit_models,model_inputs)
param_dict = lc_class.return_parameter_dict()
param_list_free = lc_class.return_free_parameter_list()

# sampling controls
sampling_method = str(input_dict['sampling_method'])
sampling_arguments = {}

if sampling_method == 'emcee':

    sampling_arguments['nwalk'] = int(input_dict['nwalkers'])
    sampling_arguments['nstep'] = input_dict['nsteps']
    if sampling_arguments['nstep'] != "auto": # use the autocorrelation time to determine when the chains have converged
        sampling_arguments['nstep'] = int(nstep)

    sampling_arguments['nthreads'] = int(input_dict['nthreads'])
    sampling_arguments['use_typeII'] = bool(int(input_dict['typeII_maximum_likelihood']))
    sampling_arguments['optimise_model'] = bool(int(input_dict['optimise_model']))

    sampling_arguments['save_chain'] = bool(int(input_dict['save_chain']))
    sampling_arguments['prod_only'] = bool(int(input_dict['prod_only']))

elif sampling_method == 'dynesty':
    sampling_arguments['nlive_pdim'] = int(input_dict['nlive_points_pdim'])
    sampling_arguments['precision_crit'] = input_dict['precision_crit']

else:
    raise SystemExit


sampling = s.Sampling(lc_class,param_dict,param_list_free,prior_dict,sampling_arguments,sampling_method)
if sampling_method == 'emcee':
    sampling.run_emcee()
elif sampling_method == 'dynesty':
    sampling.run_dynesty()