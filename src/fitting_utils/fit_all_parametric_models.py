#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from global_utils import parseInput

try:
    import fitting_utils.parametric_fitting_functions as pf
    # import fitting_utils.mcmc_utils as mc
except:
    from . import parametric_fitting_functions as pf
    # from . import mcmc_utils as mc

parser = argparse.ArgumentParser(description='Run fits of all possible parametric models to a white light curve. This currently takes time, airmass, FWHM, x positions, y positions and sky background as possible to polynomials. It then fits all combinations of polynomials up to cubic and spits out the results. Note: a directory called white_light_parametric_model_fits must exist within cwd.')

### Load in parameter file

input_dict = parseInput('parametric_input.dat')


### Load in various input arrays

time = pickle.load(open(input_dict['time_file'],'rb'))

flux = pickle.load(open(input_dict['flux_file'],'rb'))
flux_error = pickle.load(open(input_dict['error_file'],'rb'))

# The code is currently designed to take the following as possible inputs
am = pickle.load(open(input_dict['airmass_file'],'rb'))
fwhm = pickle.load(open(input_dict['fwhm_file'],'rb'))
xpos = pickle.load(open(input_dict['xpos_file'],'rb'))
ypos = pickle.load(open(input_dict['ypos_file'],'rb'))
sky = pickle.load(open(input_dict['sky_file'],'rb'))

# If wanting to normalise inputs (input - mean(input))/std(input)
normalise_inputs = bool(int(input_dict['normalise_inputs']))

# The chi2 cut above which we don't bother saving the plots and the results go into the failed results table
chi2_cut = float(input_dict['chi2_cut'])

# Do we want to clip outlying points?
clip_outliers = bool(int(input_dict['clip_outliers']))


### System params

period = float(input_dict['period'])

aRs = float(input_dict['aRs'])
inclination = float(input_dict['inclination'])
t0 = time.mean()
rp_rs = float(input_dict['rp_rs'])
u1 = 0.5 # This is default start point. Not too important as it is a fitted parameter
u2 = 0.2 # as above

# The contact points in frame numbers. Needed so that we can extract the out of transit data
contact1 = int(input_dict['contact1'])
contact4 = int(input_dict['contact4'])

# Transit (Mandel & Agol) parameters
transit_pars = np.array([t0,rp_rs,aRs,inclination,u1,u2])


### Clip outliers using running median

if clip_outliers:
    from scipy.signal import medfilt

    plt.figure()

    MF = medfilt(flux,7) # 7 is a good default over which to perform this filter
    filtered_residuals = flux - MF
    standard_residuals = np.std(filtered_residuals)

    # now we remove points that lay more than 4 standard deviations from the mean
    keep_idx = ((filtered_residuals <= 4*standard_residuals) & (filtered_residuals >= -4*standard_residuals))

    clipped_flux = flux[keep_idx]
    clipped_flux_error = flux_error[keep_idx]
    clipped_time = time[keep_idx]

    if am is not None:
        clipped_am = am[keep_idx]
    else:
        clipped_am = None

    if fwhm is not None:
        clipped_fwhm = fwhm[keep_idx]
    else:
        clipped_fwhm = None

    if xpos is not None:
        clipped_xpos = xpos[keep_idx]
    else:
        clipped_xpos = None

    if ypos is not None:
        clipped_ypos = ypos[keep_idx]
    else:
        clipped_ypos = None

    if sky is not None:
        clipped_sky = sky[keep_idx]
    else:
        clipped_sky = None

    # plot the clipped and kept data
    plt.errorbar(time[~keep_idx],flux[~keep_idx],yerr=flux_error[~keep_idx],fmt='o',ecolor='r',color='r')
    plt.errorbar(clipped_time,clipped_flux,yerr=clipped_flux_error,fmt='o',ecolor='k',color='k',alpha=0.5)
    plt.xlabel('Time (MJD)')
    plt.ylabel('Normalised flux')
    plt.show()

else:
    clipped_flux = flux
    clipped_flux_error = flux_error
    clipped_time = time

    clipped_am = am
    clipped_fwhm = fwhm
    clipped_xpos = xpos
    clipped_ypos = ypos
    clipped_sky = sky



### Run fits

all_functions = pf.polynomial_fitting(clipped_time,clipped_flux,clipped_flux_error,clipped_am,clipped_fwhm,clipped_xpos,clipped_ypos,clipped_sky,chi2_cut,transit=True,period=period,transit_pars=transit_pars,contact1 = contact1,contact4 = contact4)
