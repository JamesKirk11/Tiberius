#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
import argparse
from scipy.interpolate import UnivariateSpline as US

from global_utils import parseInput
from Tiberius.src.fitting_utils import mcmc_utils as mc
from Tiberius.src.fitting_utils import TransitModelGPPM as tmgp
from Tiberius.src.fitting_utils import parametric_fitting_functions as pf
from Tiberius.src.fitting_utils import plotting_utils as pu


parser = argparse.ArgumentParser(description='Run fit to a single light curve that is either a wavelength-binned or white light curve. This makes use of the TransitModelGPPM class, which fits the red noise as a GP + parametric model.')
parser.add_argument('wavelength_bin', help="which wavelength bin are we running the fit to? This is indexed from 0. If running fit to the white light curve, this must be given as '0'",type=int)
parser.add_argument('-dbp',"--determine_best_polynomials", help="Use this option to loop over all combination of polynomial input vectors and orders to determine the best fitting polynomials via a Nelder-Mead. This prevents an MCMC from running. Set this number to the maximum polynomial order you want to consider. e.g. 3 = cubic polys",default=0,type=int)
args = parser.parse_args()


### Load in parameter file

input_dict = parseInput('fitting_input.txt')

white_light_fit = bool(int(input_dict['white_light_fit']))

wb = args.wavelength_bin

if white_light_fit and wb > 1:
    raise ValueError('if fitting wavelength bins, need to set white_light_fit = 0 in fitting_input.txt')

### Plotting controls

rebin_data = input_dict['rebin_data']
if rebin_data is not None:
    rebin_data = int(rebin_data)
show_plots = bool(int(input_dict['show_plots']))

### Load in various input arrays
time = pickle.load(open(input_dict['time_file'],'rb'))
# time -= int(time[0]) # offset the time array to help with minimization

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


model_input_files = [i.strip() for i in input_dict['model_input_files'].split(',')]

model_inputs = []
for i in model_input_files:
    model_in = np.atleast_2d(pickle.load(open(i,'rb')))[:,first_integration:last_integration]
    if model_in.shape[0] == 1:
        vector = model_in[0]
        # replace any nans
        vector[~np.isfinite(vector)] = 1e-10
        model_inputs.append(vector)
    if model_in.shape[0] > 1:
        vector = model_in[wb]
        # replace any nans
        vector[~np.isfinite(vector)] = 1e-10
        model_inputs.append(model_in[wb])


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

    y = flux

    # Divide by the common noise model
    flux = flux/common_noise_model
    flux_error = (flux_error/y)*flux


### Red noise polynomial model parameters

# define the order of each polynomial fitted to each ancillary data set
if input_dict['polynomial_orders'] is not None:
    polynomial_orders = np.array([int(i) for i in input_dict['polynomial_orders'].split(',')])

    if polynomial_orders.sum() == 0:
        polynomial_orders = None
        polynomial_coefficients = None
        poly_used = False
    else:
        poly_used = True
else:
    polynomial_orders = None
    polynomial_coefficients = None
    poly_used = False

# determine whether we're using an exponential ramp model or not
if bool(int(input_dict['exponential_ramp'])):
    exp_ramp_used = True
else:
    exp_ramp_used = False

# determine whether we're using a step function or not
if bool(int(input_dict['step_function'])):
    step_func_used = True
else:
    step_func_used = False


# check whether the starting locations for the coefficients are given in fitting_input.txt, otherwise define these here
if poly_used or exp_ramp_used:
    if input_dict['polynomial_coefficients'] is not None:
        # polynomial_coefficients = np.array([float(i) for i in input_dict['polynomial_coefficients'].split(',')]) # in case the white light fit has already been performed, we can define the actual values here
        polynomial_coefficients_keys = np.loadtxt(input_dict['polynomial_coefficients'],usecols=0,dtype=str)
        polynomial_coefficients_values = np.loadtxt(input_dict['polynomial_coefficients'],usecols=2)
        polynomial_coefficients = []
        ramp_coefficients = []
        for k,v in zip(polynomial_coefficients_keys,polynomial_coefficients_values):
            if int(k.split("_")[1]) == wb + 1:
                if "c" in k.split("_")[0]:
                    polynomial_coefficients.append(v)
                if "r" in k.split("_")[0]:
                    ramp_coefficients.append(v)
    else:
        polynomial_coefficients = None
        ramp_coefficients = None


# Do we want to normalise inputs? Defined as (input - mean(input))/std(input)
norm_inputs = bool(int(input_dict['normalise_inputs']))

if norm_inputs:
    print('standardising model inputs...')
    systematics_model_inputs = np.array([(i-i.mean())/i.std() for i in model_inputs])
else:
    systematics_model_inputs = np.array(model_inputs)


### for GP optimisation and variance limits
contact1 = int(input_dict['contact1']) - first_integration
contact4 = int(input_dict['contact4']) - first_integration

## renormalise flux to out-of-transit median?
if bool(int(input_dict['renorm_flux'])):
    print("re-normalising flux array...")
    oot_median = np.nanmedian(np.hstack((flux[:contact1],flux[contact4:])))
    flux /= oot_median
    flux_error /= oot_median

## Remove any nans and zeroes from the error array
not_nans = np.isfinite(flux)*np.isfinite(flux_error)
time = time[not_nans]
flux = flux[not_nans]
flux_error = flux_error[not_nans]
zero_errors = flux_error == 0
if np.any(zero_errors):
    flux_error[zero_errors] = np.mean(flux_error)
systematics_model_inputs = systematics_model_inputs[:,not_nans]

### GP controls
kernel_classes = input_dict['kernel_classes']
if kernel_classes is not None:

    try:
        kernel_classes = [i.strip() for i in input_dict['kernel_classes'].split(',')]
    except:
        pass

    nkernels = len(kernel_classes)
    GP_used = True

    # are we using a white noise kernel?
    white_noise_kernel = bool(int(input_dict['white_noise_kernel']))

    # Define kernel priors
    kernel_priors_dict = {}

    if white_noise_kernel:
        kernel_priors_dict['min_WN_sigma'] = float(input_dict['min_WN_sigma'])
        kernel_priors_dict['max_WN_sigma'] = float(input_dict['max_WN_sigma'])

    max_A_multiplier = float(input_dict['max_A_multiplier']) # 100 is good for this
    min_A_multiplier = float(input_dict['min_A_multiplier']) # 0.01 is good for this

    dy_max_multiplier =  float(input_dict['dy_max_multiplier']) # 2 is good for this
    dy_min_multiplier =  float(input_dict['dy_min_multiplier']) # 1 is good for this

    max_A = np.log(max_A_multiplier*np.var(np.hstack((flux[:contact1],flux[contact4:]))))#/nkernels # max is ? times the variation of out of transit flux
    min_A = np.log(min_A_multiplier*np.var(np.hstack((flux[:contact1],flux[contact4:]))))#/nkernels #  min is ? times the variation of out of transit flux

    kernel_priors_dict['min_A'] = min_A
    kernel_priors_dict['max_A'] = max_A

    kernel_priors_dict['min_A'] = min_A
    kernel_priors_dict['max_A'] = max_A

    for i in range(nkernels):

        dy_min = dy_min_multiplier*abs(np.median(np.diff(systematics_model_inputs[i]))) # minimum distance is median distance between data points
        dy_max = dy_max_multiplier*(systematics_model_inputs[i].max()-systematics_model_inputs[i].min()) # maximum distance is ?x the range.

        # define max and min log inverse length scales
        max_lniL = np.log(1/dy_min)
        min_lniL = np.log(1/dy_max)

        print("Kernel %d: min amp = % .2f ; max amp = %.2f ; min lniL = %.2f ; max lniL = %.2f"%((i+1),min_A,max_A,min_lniL,max_lniL))

        kernel_priors_dict['min_lniL_%d'%(i+1)] = min_lniL
        kernel_priors_dict['max_lniL_%d'%(i+1)] = max_lniL

else:
    nkernels = 0
    GP_used = False
    kernel_priors_dict = sigma = None
    white_noise_kernel = False

### System params, fit if white light, fix if not

k = tmgp.Param(float(input_dict['k']))

if not white_light_fit:
    aRs = float(input_dict['aRs'])
    inclination = float(input_dict['inclination'])
    t0 = float(input_dict['t0'])
    # t0 -= int(t0)
    ecc = float(input_dict['ecc'])
    omega = float(input_dict['omega'])
    period = float(input_dict['period'])
    k_prior = input_dict['k_prior']

    if k_prior is not None:
        sys_priors = {"period_prior":None,"ecc_prior":None,"k_prior":float(k_prior),"aRs_prior":None,"inc_prior":None,"omega_prior":None}

    else:
        sys_priors = None


else:

    sys_priors = {}

    if input_dict['k_prior'] is not None:
        sys_priors["k_prior"] = float(input_dict['k_prior'])
    else:
        sys_priors["k_prior"] = None

    if input_dict['period_prior'] == 'fixed':
        period = float(input_dict['period'])
        sys_priors["period_prior"] = None
    else:
        period = tmgp.Param(float(input_dict['period']))
        if input_dict['period_prior'] is not None:
            sys_priors["period_prior"] = float(input_dict['period_prior'])
        else:
            sys_priors["period_prior"] = None

    if input_dict['ecc_prior'] == 'fixed':
        ecc = float(input_dict['ecc'])
        sys_priors["ecc_prior"] = None
    else:
        ecc = tmgp.Param(float(input_dict['ecc']))
        if input_dict['ecc_prior'] is not None:
            sys_priors["ecc_prior"] = float(input_dict['ecc_prior'])
        else:
            sys_priors["ecc_prior"] = None

    if input_dict['aRs_prior'] == 'fixed':
        aRs = float(input_dict['aRs'])
        sys_priors["aRs_prior"] = None
    else:
        aRs = tmgp.Param(float(input_dict['aRs']))
        if input_dict['aRs_prior'] is not None:
            sys_priors["aRs_prior"] = float(input_dict['aRs_prior'])
        else:
            sys_priors["aRs_prior"] = None


    if input_dict['inclination_prior'] == 'fixed':
        inclination = float(input_dict['inclination'])
        sys_priors["inc_prior"] = None
    else:
        inclination = tmgp.Param(float(input_dict['inclination']))
        if input_dict['inclination_prior'] is not None:
            sys_priors["inc_prior"] = float(input_dict['inclination_prior'])
        else:
            sys_priors["inc_prior"] = None

    if input_dict['omega_prior'] == 'fixed':
        omega = float(input_dict['omega'])
        sys_priors["omega_prior"] = None
    else:
        omega = tmgp.Param(float(input_dict['omega']))
        if input_dict['omega_prior'] is not None:
            sys_priors["omega_prior"] = float(input_dict['omega_prior'])
        else:
            sys_priors["omega_prior"] = None

    t0_guess = float(input_dict['t0'])
    # t0_guess -= int(t0_guess)
    t0 = tmgp.Param(t0_guess)

    if input_dict['t0_prior'] is not None and input_dict['t0_prior'] != "fixed":
        sys_priors["t0_prior"] = float(input_dict['t0_prior'])
    else:
        sys_priors["t0_prior"] = None


### Limb darkening
ld_law = input_dict["ld_law"]
FIX_U1 = bool(int(input_dict['fix_u1']))
FIX_U2 = bool(int(input_dict['fix_u2']))
FIX_U3 = bool(int(input_dict['fix_u3']))
FIX_U4 = bool(int(input_dict['fix_u4']))
use_ld_prior = bool(int(input_dict['ld_prior']))
if use_ld_prior:
    ld_prior = OrderedDict()
else:
    ld_prior = None

# Do we want to use Kipping's change of variable for efficient sampling? Note: this is not fully tested yet!
use_kipping = bool(int(input_dict['use_kipping_parameterisation']))
if use_kipping and FIX_U1 or use_kipping and FIX_U2:
    raise ValueError('Not advisable to use Kipping parameterisation and fix a coefficient (sort of defeats the object...)')

# Load in coefficients generated through generate_LDCS.py
try:
    wc,we,u1,u1_err,u2,u2_err,u3,u3_err,u4,u4_err = np.loadtxt('LD_coefficients.txt',unpack=True)
except:
    raise SystemError('Need to first generate limb darkening values before running this fitting.')

# if not white_light_fit and not single_fit:
u1 = np.atleast_1d(u1)[wb]
u1_err = np.atleast_1d(u1_err)[wb]
u2 = np.atleast_1d(u2)[wb]
u2_err = np.atleast_1d(u2_err)[wb]
u3 = np.atleast_1d(u3)[wb]
u3_err = np.atleast_1d(u3_err)[wb]
u4 = np.atleast_1d(u4)[wb]
u4_err = np.atleast_1d(u4_err)[wb]

if use_ld_prior and np.all(u1_err) == 0:
    raise ValueError("can't set ld_prior = 1 as all your LD_coefficients.dat uncertainties = 0")


### Fitting controls

nwalk = int(input_dict['nwalkers'])
nstep = input_dict['nsteps']
if nstep != "auto": # use the autocorrelation time to determine when the chains have converged
    nstep = int(nstep)

nthreads = int(input_dict['nthreads'])
use_typeII = bool(int(input_dict['typeII_maximum_likelihood']))
optimise_model = bool(int(input_dict['optimise_model']))

clip_outliers = bool(int(input_dict['clip_outliers']))
sigma_clip = float(input_dict['sigma_cut'])
if clip_outliers:
    median_clip = bool(int(input_dict['median_clip']))
else:
    median_clip = False

save_chain = bool(int(input_dict['save_chain']))
prod_only = bool(int(input_dict['prod_only']))


### Initiate dictionaries with starting parameters

d = OrderedDict()

d['t0'] = t0
d['inc'] = inclination
d['aRs'] = aRs
d['period'] = period
d['ecc'] = ecc
d['omega'] = omega
d['k'] = k

if not use_kipping:

    if FIX_U1:
        d['u1'] = u1
    else:
        d['u1'] = tmgp.Param(u1)
        if use_ld_prior:
            ld_prior['u1_prior'] = u1_err

    if ld_law != "linear":
        if FIX_U2:
            d['u2'] = u2
        else:
            d['u2'] = tmgp.Param(u2)
            if use_ld_prior:
                ld_prior['u2_prior'] = u2_err

    if ld_law == "nonlinear":
        if FIX_U3:
            d['u3'] = u3
        else:
            d['u3'] = tmgp.Param(u3)
            if use_ld_prior:
                ld_prior['u3_prior'] = u3_err

        if FIX_U4:
            d['u4'] = u4
        else:
            d['u4'] = tmgp.Param(u4)
            if use_ld_prior:
                ld_prior['u4_prior'] = u4_err

else:
    print("\n - Using Kipping's parameterisation of quadratic limb darkening coefficients")
    # convert from u1, u2 into q1, q2 if using Kipping parameterisation

    q1 = (u1+u2)**2
    q2 = u1/(2*(u1+u2))

    # Note: I am not transforming the uncertainties here on purpose as I want the uncertainties to be read from LD_coefficients.dat to be == the q1 and q2 standard deviations.
    if use_ld_prior:
        ld_prior['u1_prior'] = u1_err
        ld_prior['u2_prior'] = u2_err

    d['u1'] = tmgp.Param(q1)
    d['u2'] = tmgp.Param(q2)

### Now define the GP and polycoefficient parameters.
if GP_used:
    if white_noise_kernel:
        d['s'] = tmgp.Param(np.log(100e-6**2)) # start white noise kernel with 100ppm noise. Note: this is likely much too large but MCMC will sort this out.

    # starting values for the GP hyperparameters
    d['A'] = tmgp.Param(np.log(np.var(np.hstack((flux[:contact1],flux[contact4:]))))) # this is the amplitude of the GP in natural logs..
    for j in range(nkernels):
        d['lniL_%d'%(j+1)] = tmgp.Param(np.log((1/(systematics_model_inputs[j].max()-systematics_model_inputs[j].min()))))

if poly_used:
    if polynomial_coefficients is not None:
        for i,c in enumerate(polynomial_coefficients):
            # d['c%d'%(i+1)] = tmgp.Param(c)
            d['c%d'%(i+1)] = c
    else:
        d['c1'] = tmgp.Param(1.0)
        for i in range(1,polynomial_orders.sum()+1):
            d['c%d'%(i+1)] = tmgp.Param(-1e-5)

if exp_ramp_used:
    exp_ramp_components = int(input_dict["exponential_ramp"])
    if ramp_coefficients is not None:
        for i,c in enumerate(ramp_coefficients):
            # d['r%d'%(i+1)] = tmgp.Param(c)
            d['r%d'%(i+1)] = c
    else:
        for i in range(0,exp_ramp_components*2):
            if i%2 == 0:
                d["r%d"%(i+1)] = tmgp.Param(0) # the r1 parameter
            if i%2 == 1:
                d["r%d"%(i+1)] = tmgp.Param(-5) # the r2 parameter

else:
    exp_ramp_components = 0

if step_func_used:
    d["step1"] = tmgp.Param(1)
    # d["step2"] = tmgp.Param(1)
    if white_light_fit:
        d["breakpoint"] = tmgp.Param(float(input_dict["step_breakpoint"]))
    else:
        d["breakpoint"] = float(input_dict["step_breakpoint"])
    # d["breakpoint2"] = tmgp.Param(int(input_dict["step_breakpoint"])+1)

if not poly_used and not exp_ramp_used and not step_func_used:
    # if we're not using a polynomial, we're including a normalization factor to multiply the transit light curve by to account for imperfect normalisation of out-of-transit data
    d['f'] = tmgp.Param(1)


### Optionally clip outliers using running median

if clip_outliers and median_clip:

    from scipy.signal import medfilt

    print('Clipping outliers...')

    # Running median
    box_width = int(len(flux)/10)
    if box_width % 2 == 0:
        box_width += 1

    MF = medfilt(flux,box_width)

    # Use polynomial to remove edge effects of running median
    x = np.arange(len(flux))

    poly_left = np.poly1d(np.polyfit(x[:box_width*2],MF[:box_width*2],1))
    poly_right = np.poly1d(np.polyfit(x[-box_width*2:],MF[-box_width*2:],1))

    MF[:box_width] = poly_left(x[:box_width])
    MF[-box_width:] = poly_right(x[-box_width:])

    filtered_residuals = flux - MF
    standard_residuals = np.std(filtered_residuals)

    keep_idx = ((filtered_residuals <= sigma_clip*standard_residuals) & (filtered_residuals >= -sigma_clip*standard_residuals))

    clipped_flux = flux[keep_idx]
    clipped_flux_error = flux_error[keep_idx]
    clipped_time = time[keep_idx]
    clipped_model_input = np.array(systematics_model_inputs)[:,keep_idx].reshape(len(systematics_model_inputs),len(np.where(keep_idx == True)[0]))

    if show_plots:
        plt.figure()
        plt.subplot(211)
        plt.errorbar(time[~keep_idx],flux[~keep_idx],yerr=flux_error[~keep_idx],fmt='o',ecolor='r',color='r',label='Clipped outliers')
        plt.errorbar(clipped_time,clipped_flux,yerr=clipped_flux_error,fmt='o',ecolor='k',color='k',alpha=0.5)
        plt.plot(time,MF,'r')
        plt.ylabel('Normalised flux')
        plt.title('Outlier clipping')
        plt.legend(loc='upper left')

        # residuals
        plt.subplot(212)
        plt.errorbar(time[~keep_idx],flux[~keep_idx]-MF[~keep_idx],yerr=flux_error[~keep_idx],fmt='o',ecolor='r',color='r',label='Clipped outliers')
        plt.errorbar(clipped_time,clipped_flux-MF[keep_idx],yerr=clipped_flux_error,fmt='o',ecolor='k',color='k',alpha=0.5)
        plt.axhline(sigma_clip*standard_residuals,ls='--',color='gray')
        plt.axhline(sigma_clip*-standard_residuals,ls='--',color='gray')
        plt.ylim(-10*standard_residuals,10*standard_residuals)

        plt.xlabel('Time (MJD)')
        plt.ylabel('Residuals')

        plt.show(block=False) # only show for 5 seconds. This is necessary when running fits to multiple bins so that the code doesn't have to wait for user to manually close windows before continuing.
        plt.pause(5)
        plt.close()


### Optionally optimise the transit model parameters using a cubic-in-time polynomial here to handle systematic noise here. We can also optionally use this fit to clip outliers instead of through the median clip
# raise SystemExit

if optimise_model or clip_outliers and not median_clip:

    # Make a new dictionary and toy model where we don't include any GP parameters, these are optimised later
    d_clip = OrderedDict()
    for k,v in zip(d.keys(),d.values()):
        if k != "s" and k != "A" and "lniL" not in k and 'f' not in k:
            d_clip[k] = v

    if poly_used: # we use the polynomial orders and inputs as defined in fitting_input.txt
        polynomial_orders_toy = polynomial_orders
        if median_clip:
            red_noise_model_inputs = clipped_model_input
        else:
            red_noise_model_inputs = systematics_model_inputs
    else: # in this case we want to use a polynomial to clip outliers but for these purposes we're only going to use time (quadratic) to detrend
        if not exp_ramp_used:
            polynomial_orders_toy = np.array([2])
            d_clip['c1'] = tmgp.Param(1.0)
            d_clip['c2'] = tmgp.Param(-1e-5)
            d_clip['c3'] = tmgp.Param(-1e-5)
        else:
            polynomial_orders_toy = None

        if median_clip:
            red_noise_model_inputs = [clipped_time]
        else:
            red_noise_model_inputs = [time]

    if exp_ramp_used:
        for i in range(0,exp_ramp_components*2):
            if i%2 == 0:
                d_clip["r%d"%(i+1)] = tmgp.Param(0) # the r1 parameter
            if i%2 == 1:
                d_clip["r%d"%(i+1)] = tmgp.Param(-5) # the r2 parameter

    if step_func_used:
        d_clip["step1"] = tmgp.Param(1)
        # d_clip["step2"] = tmgp.Param(1)
        if white_light_fit:
            d_clip["breakpoint"] = tmgp.Param(float(input_dict["step_breakpoint"]))
        else:
            d_clip["breakpoint"] = float(input_dict["step_breakpoint"])
        # d_clip["breakpoint2"] = tmgp.Param(int(input_dict["step_breakpoint"])+1)



    ### Generate starting model
    if median_clip:
        clip_model = tmgp.TransitModelGPPM(d_clip,red_noise_model_inputs,None,clipped_flux_error,clipped_time,kernel_priors_dict,white_noise_kernel,use_kipping,ld_prior,polynomial_orders_toy,ld_law,exp_ramp_used,exp_ramp_components,step_func_used)
    else:
        clip_model = tmgp.TransitModelGPPM(d_clip,red_noise_model_inputs,None,flux_error,time,kernel_priors_dict,white_noise_kernel,use_kipping,ld_prior,polynomial_orders_toy,ld_law,exp_ramp_used,exp_ramp_components,step_func_used)

    if not np.any(flux_error) > 0: # the errors are all zeroes, need to be replaced for sake of fit only
        print("using dummy flux errors for sigma clipping only")
        dummy_error = 1e-3*flux
    else:
        dummy_error = flux_error

    # Now fit the model to get the clipped model
    if median_clip:
        try:
            fitted_clip_model,_,_ = clip_model.optimise_params(clipped_time,clipped_flux,clipped_flux_error,reset_starting_gp=False,contact1=contact1,contact4=contact4,full_model=True,sys_priors=sys_priors,LM_fit=True)
        except:
            fitted_clip_model,_ = clip_model.optimise_params(clipped_time,clipped_flux,clipped_flux_error,reset_starting_gp=False,contact1=contact1,contact4=contact4,full_model=True,sys_priors=sys_priors)

    else:
        try:
            fitted_clip_model,_,_ = clip_model.optimise_params(time,flux,flux_error,reset_starting_gp=False,contact1=contact1,contact4=contact4,full_model=True,sys_priors=sys_priors,LM_fit=True)
        except:
            fitted_clip_model,_ = clip_model.optimise_params(time,flux,flux_error,reset_starting_gp=False,contact1=contact1,contact4=contact4,full_model=True,sys_priors=sys_priors)


        # check contact points
        initial_red_noise = 1
        if poly_used:
            initial_red_noise *= fitted_clip_model.red_noise_poly(time)
        if exp_ramp_used:
            initial_red_noise *= fitted_clip_model.exponential_ramp(time)
        if step_func_used:
            initial_red_noise *= fitted_clip_model.step_function(time)


        initial_transit_model = fitted_clip_model.calc(time)/initial_red_noise

    if white_light_fit:
        try:
            # we can now use the transit light curve fit as another check of where the first and fourth contact points are in terms of frame numbers
            new_contact1 = max(np.where(initial_transit_model[:contact1+10]==1)[0]) # we assume a 10 frame uncertainty on the user-defined guess
            new_contact4 = min(np.where(initial_transit_model[contact4-10:]==1)[0]+contact4-10)
            print("\n## Contact 1 from fit = %d, contact 4 from fit = %d"%(new_contact1,new_contact4))
        except:
            pass

    ### Plot the results
    if clip_outliers and not median_clip:
        print('\nPlotting lsq fit for clipping using polynomial....')
    else:
        print("\nPlotting initial optimised model....")

    if show_plots:
        if median_clip:
            fig = pu.plot_single_model(fitted_clip_model,clipped_time,clipped_flux,clipped_flux_error,save_fig=False,plot_residual_std=sigma_clip)
        else:
            fig = pu.plot_single_model(fitted_clip_model,time,flux,flux_error,save_fig=False,plot_residual_std=sigma_clip)

    if optimise_model:

        ### update starting transit model parameters with these optimised parameters
        print("...updating transit and (optionally) polynomial parameters with optimised values")
        for k,v in zip(d_clip.keys(),d_clip.values()):
            if k in d:
                d[k] = v

        ### update photometric uncertainties given best-fit model
        if median_clip:
            print("\nRescaling photometric uncertainties by %.3f to give rChi2 = 1"%np.sqrt(fitted_clip_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error)))
            clipped_flux_error = clipped_flux_error*np.sqrt(fitted_clip_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error))
            pickle.dump(clipped_flux_error,open('rescaled_errors_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))
        else:
            print("\nRescaling photometric uncertainties by %.3f to give rChi2 = 1"%np.sqrt(fitted_clip_model.reducedChisq(time,flux,flux_error)))
            flux_error = flux_error*np.sqrt(fitted_clip_model.reducedChisq(time,flux,flux_error))
            pickle.dump(flux_error,open('rescaled_errors_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))


    if clip_outliers and not median_clip: # use the above to clip outliers, if we've not already clipped them with the median clipping above
        residuals_1 = flux - fitted_clip_model.calc(time)
        rms_1 = np.sqrt(np.mean(residuals_1**2))
        keep_idx = ((residuals_1 >= -sigma_clip*rms_1) & (residuals_1 <= sigma_clip*rms_1))

        clipped_flux = flux[keep_idx]
        clipped_flux_error = flux_error[keep_idx]
        clipped_time = time[keep_idx]
        clipped_model_input = np.array(systematics_model_inputs)[:,keep_idx].reshape(len(systematics_model_inputs),len(np.where(keep_idx == True)[0]))

        if show_plots:
            print("Plotting light curve after outlier clipping...")
            fig = pu.plot_single_model(fitted_clip_model,clipped_time,clipped_flux,clipped_flux_error,save_fig=False,plot_residual_std=sigma_clip,systematics_model_inputs=clipped_model_input)


if not clip_outliers:
    keep_idx = np.ones_like(time).astype(bool)
    clipped_flux = flux
    clipped_flux_error = flux_error
    clipped_time = time
    clipped_model_input = np.array(systematics_model_inputs)


### Save clipped arrays for ease of future plotting
pickle.dump(clipped_flux,open('sigma_clipped_flux_wb%s.pickle'%(str(wb+1).zfill(4)),'wb')) # add '0' in front of single digit wavelength bin numbers so that linux sorts them properly
pickle.dump(clipped_time,open('sigma_clipped_time_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))
pickle.dump(clipped_flux_error,open('sigma_clipped_error_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))
pickle.dump(clipped_model_input,open('sigma_clipped_model_inputs_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))
pickle.dump(keep_idx,open('data_quality_flags_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))

if clip_outliers:
    print("\n %d data points (%.2f%%) clipped from fit"%(len(time)-len(clipped_time),100*(len(time)-len(clipped_time))/len(time)))

starting_model = tmgp.TransitModelGPPM(d,clipped_model_input,kernel_classes,clipped_flux_error,clipped_time,kernel_priors_dict,white_noise_kernel,use_kipping,ld_prior,polynomial_orders,ld_law,exp_ramp_used,exp_ramp_components,step_func_used)

if not optimise_model and show_plots:
    print("plotting starting model")
    fig = pu.plot_single_model(starting_model,clipped_time,clipped_flux,clipped_flux_error,save_fig=False)


### Optionally fit all combinations of polynomials with a Nelder-Mead to determine the best orders and inputs to use
if args.determine_best_polynomials > 0 and not GP_used:
    if not optimise_model:
        print(optimise_model)
        raise ValueError("If using -dbp, we must first optimise the starting model! Set optimise_model = 1 in fitting_input.txt")
    print("\n### Determining best combination of polynomials and input vectors (this could take a while)...")
    cont = input("...Is time given as the first path given to model_input_files in fitting_input.txt? [Y/n]: ")
    if cont == "Y":
        print("continuing...\n")
    else:
        print("Make sure time is first input given to model_input_files in fitting_input.txt\n")
        raise SystemExit

    # Rescale uncertainties to give rChi2 = 1 for reference/starting model
    clipped_flux_error = clipped_flux_error*np.sqrt(starting_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error))

    # Now feed in the rescaled errors to all cominations fitting
    pf.fit_all_polynomial_combinations(starting_model,clipped_time,clipped_flux,clipped_flux_error,clipped_model_input,max_order=args.determine_best_polynomials,sys_priors=sys_priors)

    print("All combinations tested, exiting.")
    raise SystemExit

if args.determine_best_polynomials and GP_used:
    raise ValueError("determine_best_polynomials and GPs cannot be used at the same time -> turn one off")


if show_plots and GP_used: # if we're not using a GP, we've already plotted the starting model from the optimise output above
    print('\nPlotting starting model....')
    ### Plot starting model
    fig = pu.plot_single_model(starting_model,clipped_time,clipped_flux,clipped_flux_error,rebin_data=rebin_data,save_fig=False)

### optimise model parameters prior to MCMC?
if optimise_model and GP_used:
    print('\nOptimising hyperparameters....')
    optimised_kernel_params = starting_model.optimise_params(clipped_time,clipped_flux,clipped_flux_error,reset_starting_gp=starting_model.GP_used,contact1=contact1,contact4=contact4)

    print('\nPlotting optimised GP model....')
    if show_plots:
        fig = pu.plot_single_model(starting_model,clipped_time,clipped_flux,clipped_flux_error,rebin_data=rebin_data,save_fig=False)

    if bool(int(input_dict["reset_kernel_priors"])):
        print("\nUpdating kernel priors with optimised values...\n")
        kp_counter = 0

        if white_noise_kernel:
            new_bounds_wn = (np.sqrt(np.exp(optimised_kernel_params[0]))*0.9,np.sqrt(np.exp(optimised_kernel_params[0]))*1.1)
            starting_model.kernel_priors["min_WN_sigma"] = min(new_bounds_wn)
            starting_model.kernel_priors["max_WN_sigma"] = max(new_bounds_wn)
            kp_counter += 1

        new_bounds_A = (0.9*optimised_kernel_params[kp_counter],1.1*optimised_kernel_params[kp_counter])
        starting_model.kernel_priors["min_A"] = min(new_bounds_A)
        starting_model.kernel_priors["max_A"] = max(new_bounds_A)
        kp_counter += 1

        for i in range(nkernels):
            new_bounds_lniL = (0.9*optimised_kernel_params[kp_counter],1.1*optimised_kernel_params[kp_counter])
            starting_model.kernel_priors["min_lniL_%d"%(i+1)] = min(new_bounds_lniL)
            starting_model.kernel_priors["max_lniL_%d"%(i+1)] = max(new_bounds_lniL)
            kp_counter += 1


### check that the prior likelihood is not minus infinity, otherwise the code will fail with no warning
if not np.isfinite(starting_model.lnprior(sys_priors)) :
    raise ValueError('Model lnprior is negative infinity')

### Run the MCMC

if use_typeII:
    raise Warning("TypeII has not been tested in a long time, may not be accurate")
    # Using Type II maximum likelihood estimation as used by Gibson+ and Rajpaul+
    print('Running Type II maximum likelihood...')
    median_typeII,upper_typeII,lower_typeII,typeII_model = mc.run_emcee(starting_model,clipped_time,clipped_flux,clipped_flux_error,nwalk,nstep,nthreads,burn=False,wavelength_bin=wb,sys_priors=sys_priors,typeII=True,save_chain=False)
    print('...Type II maximum likelihood complete')
    starting_model = typeII_model
    raise SystemExit
    continue_mcmc = input('Type II worked? (check corner plot) [Y/n] : ')
    if continue_mcmc == 'Y':
        pass
    else:
        raise SystemExit

if nstep != 0:
    if not prod_only:
        # Run burn-in
        if nstep != "auto":
            if nstep > 5000 and GP_used: # only run a maximum of 5000 steps for burn in if we're using a GP since we don't perform an uncertainty rescaling
                nstep_burn = 5000
            else:
                nstep_burn = nstep
        else:
            if poly_used and not GP_used: # then we're rescaling the uncertainties so we need to make sure the chains are long enough to accurately rescale the uncertainties
                nstep_burn = nstep
            else:
                nstep_burn = 2000 # short burn in before we perform the auto correlation testing

        median_burn,upper_burn,lower_burn,burn_model = mc.run_emcee(starting_model,clipped_time,clipped_flux,clipped_flux_error,nwalk,nstep_burn,nthreads,burn=True,wavelength_bin=wb,sys_priors=sys_priors,typeII=False)

        # Update burn_model starting params with current params
        for i in burn_model.pars.keys():
            try:
                burn_model.pars[i].startVal = burn_model.pars[i].currVal
            except:
                pass

        if not GP_used:
            # we need to rescale the photometric uncertainties to give reduced chi2 = 1
            print("\nRescaling photometric uncertainties to give rChi2 = 1")
            clipped_flux_error = clipped_flux_error*np.sqrt(burn_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error))
            if show_plots:
                fig = pu.plot_single_model(burn_model,clipped_time,clipped_flux,clipped_flux_error,rebin_data=rebin_data,save_fig=False)
            rchi2_rescaled = burn_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error)
            print("reduced Chi2 following error rescaling = %.2f"%(rchi2_rescaled))
            pickle.dump(clipped_flux_error,open('rescaled_errors_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))

        # Run production
        median,upper,lower,prod_model = mc.run_emcee(burn_model,clipped_time,clipped_flux,clipped_flux_error,nwalk,nstep,nthreads,burn=False,wavelength_bin=wb,sys_priors=sys_priors,typeII=False,save_chain=save_chain)

    else: # we're running a single chain, all the way through.
        # Run production
        median,upper,lower,prod_model = mc.run_emcee(starting_model,clipped_time,clipped_flux,clipped_flux_error,nwalk,nstep,nthreads,burn=False,wavelength_bin=wb,sys_priors=sys_priors,typeII=False,save_chain=save_chain)

else: # we're not running an MCMC at all here, we're just using a Levenberg-Marquadt to estimate the parameters and uncertainties!

    # the "burn" model is our first iteration before rescaling the uncertainties
    burn_model,median_burn,uncertainties_burn = starting_model.optimise_params(clipped_time,clipped_flux,clipped_flux_error,reset_starting_gp=False,contact1=contact1,contact4=contact4,full_model=True,sys_priors=sys_priors,LM_fit=True)

    # we need to rescale the photometric uncertainties to give reduced chi2 = 1
    print("\nRescaling photometric uncertainties by %.3f to give rChi2 = 1"%np.sqrt(burn_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error)))
    clipped_flux_error = clipped_flux_error*np.sqrt(burn_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error))
    if show_plots:
        fig = pu.plot_single_model(burn_model,clipped_time,clipped_flux,clipped_flux_error,rebin_data=rebin_data,save_fig=False)
    rchi2_rescaled = burn_model.reducedChisq(clipped_time,clipped_flux,clipped_flux_error)
    print("reduced Chi2 following error rescaling = %.2f"%(rchi2_rescaled))
    pickle.dump(clipped_flux_error,open('rescaled_errors_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))

    # the 'prod' model is the second fit after rescaling the uncertainties
    prod_model,median_prod,uncertainties_prod = burn_model.optimise_params(clipped_time,clipped_flux,clipped_flux_error,reset_starting_gp=False,contact1=contact1,contact4=contact4,full_model=True,sys_priors=sys_priors,LM_fit=True)

    # save the results
    mc.save_LM_results(median_prod,uncertainties_prod,prod_model.namelist,wb+1,prod_model,clipped_time,clipped_flux,clipped_flux_error,verbose=True)
    pickle.dump(prod_model,open('prod_model_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))

#### Plot and save final models

if GP_used:
    if white_noise_kernel:
        # if using white noise kernel, we need to take this into account in our error bars
        GP = prod_model.construct_gp(compute=True,flux_err=clipped_flux_error)
        wn_var = np.exp(GP.white_noise.get_value(clipped_time))
        wn_std = np.sqrt(wn_var)
        e = np.sqrt(wn_std**2+clipped_flux_error**2)
        pickle.dump(e,open('rescaled_errors_wb%s.pickle'%(str(wb+1).zfill(4)),'wb'))

    else:
        e = clipped_flux_error
else:
    e = clipped_flux_error

if not GP_used:
    fig = pu.plot_single_model(prod_model,clipped_time,clipped_flux,e,rebin_data=rebin_data,save_fig=True,wavelength_bin=wb,deconstruct=True)

    if white_light_fit:
        # generate red noise model for possible common mode correction. Note this is performed on the full time array, not the clipped time array, so that the same number of data points are used in the WLC and wb fits

        # first get the red noise model
        red_noise_model = 1

        if poly_used:
            red_noise_model *= prod_model.red_noise_poly(time,systematics_model_inputs)

        if exp_ramp_used:
            red_noise_model *= prod_model.exponential_ramp(time)

        if step_func_used:
            red_noise_model *= prod_model.step_function(time)

        pickle.dump(red_noise_model,open('red_noise_model.pickle','wb'))

        # now get the common noise model, which is the flux array minus the best fitting transit model
        full_model = prod_model.calc(time,systematics_model_inputs)

        transit_model = full_model/red_noise_model

        common_noise_model = flux/transit_model
        pickle.dump(common_noise_model,open('common_noise_model.pickle','wb'))

else:
    fig = pu.plot_single_model(prod_model,clipped_time,clipped_flux,e,rebin_data=rebin_data,save_fig=True,wavelength_bin=wb,deconstruct=True)

    # Save white light GP model as common mode
    if white_light_fit:
        mu,std,mu_components = prod_model.calc_gp_component(time,flux,flux_error,systematics_model_inputs,True)

        pickle.dump(mu,open('gp_model_all.pickle','wb'))

        for i,m in enumerate(mu_components):
            pickle.dump(m,open('gp_model_kernel%d.pickle'%(i+1),'wb'))

        # the GP component combined with the residuals is given by subtracting the best-fitting transit model. This can be calculated for original arrays
        tm = prod_model.calc(time,systematics_model_inputs)
        common_noise_model = flux/tm
        pickle.dump(common_noise_model,open("common_noise_model.pickle","wb"))


if white_light_fit: # make plot of rms vs bins and expected vs calculated LDCs for white light curve. If wb fits, this is handled elsewhere
    print("\nNow running plot_output.py for expected_vs_calculated_LDCS.pdf and rms_vs_bins.pdf...")
    import os
    os.system("python %s/plot_output.py -wlc -cp -st -s"%os.path.dirname(tmgp.__file__))
    print("Making model table")
    os.system("python %s/model_table_generator.py"%os.path.dirname(tmgp.__file__))
