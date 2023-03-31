#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk 

### Fitting a simple white light model to determine the scatter in the white light curve

import numpy as np
import matplotlib.pyplot as plt
import fitting_utils.mcmc_utils as mc
from fitting_utils.TransitModelPM import *
from scipy import optimize
import fitting_utils.plotting_utils as pu
import pickle
import argparse
from scipy.signal import medfilt
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Fit white light curves as produced by long slit extraction pipeline and print the RMS of the residuals')
parser.add_argument("-am","--am_limit",help="Define airmass limit",type=float,default=3)
parser.add_argument("-n","--norm_factor",help="Use noddy factor for normalisation by",type=float)
parser.add_argument("-c1","--contact1",help="Frame number of first contact point for normalisation",type=int)
parser.add_argument("-c4","--contact4",help="Frame number of fourth contact point for normalisation",type=int)
parser.add_argument("-clip","--sigma_clip",help='Clip outliers using a running median, define here the level of clipping',type=float,default=0)
parser.add_argument("-refit","--refit",help='Refit light curve model following clipping after an initial fit. Argument is number of sigma to clip at.',type=float,default=0)
parser.add_argument("-order","--poly_order",help='Order of polynomial used for fitting, default = 3',type=int,default=3)
parser.add_argument("-m","--fit_method",help='Which fitting method to use? As specified in scipy.optimize. Default is "TNC" but "Nelder-Mead" is good alternative if first fit is poor',default='TNC')
parser.add_argument("-cut","--cut",help='Cut the end of the light curve at an arbitrary point, given in frame number.',type=int)
# parser.add_argument('-save_kept_indices',"--save_kept_indices",help="use this to save the kept indices frames resulting from the residual clipping - useful for subsequent transiting fitting. The number given is appended to the file name.",type=int)
args = parser.parse_args()

x,y,e = np.loadtxt('white_light.dat',unpack=True)

if args.cut:
    x = x[:args.cut]
    y = y[:args.cut]
    e = e[:args.cut]

if args.contact1 is None or args.contact4 is None:
    if args.norm_factor is None:
        print("Enter frame number of 1st and 4th contact point")
        plt.figure()
        plt.plot(y,'r')
        plt.xlabel('Frame number')
        plt.show()

        contact1 = int(input("Contact 1:" ))
        contact4 = int(input("Contact 4:" ))
else:
    contact1 = args.contact1
    contact4 = args.contact4

if args.norm_factor is None:
    normalisation_factor = np.median(np.hstack((y[:contact1],y[contact4:])))
else:
    normalisation_factor = args.norm_factor

y = y/normalisation_factor
e = e/normalisation_factor

try:
    am = pickle.load(open('airmass.pickle','rb'))
except:
    try:
        am = np.loadtxt('airmass.dat')
    except:
        am = pickle.load(open('pickled_objects/airmass.pickle','rb'))

if args.am_limit > 0 and len(am) == len(x):
    x = x[am<=args.am_limit]
    y = y[am<=args.am_limit]
    e = e[am<=args.am_limit]

if args.sigma_clip > 0:

     MF = medfilt(y,7)
     filtered_residuals = y - MF
     standard_residuals = np.std(filtered_residuals)

     keep_idx = ((filtered_residuals <= args.sigma_clip*standard_residuals) & (filtered_residuals >= -args.sigma_clip*standard_residuals))

     # pickle.dump(keep_idx,open('sigma_clipped_keep_idx.pickle','wb'))

     plt.figure()
     plt.plot(x,y,'ko',label='Kept points')
     plt.plot(x[keep_idx],y[keep_idx],'ro',label='clipped points')
     plt.plot(x,MF)
     plt.xlabel('Frame number')
     plt.ylabel('Normalised flux')
     plt.legend(loc='upper right')
     plt.show()

     x = x[keep_idx]
     y = y[keep_idx]
     e = e[keep_idx]

x = x - int(x[0])

input_dict = mc.parseInput('system_params.dat')

# Note system_params.dat is a text file which must include the following: e.g. period = 1.056

period = float(input_dict['period'])
rp_rs = float(input_dict['rp_rs'])
rp_rs_prior = float(input_dict['rp_rs_prior'])
aRs = float(input_dict['aRs'])
aRs_prior = float(input_dict['aRs_prior'])
inclination = float(input_dict['inclination'])
inclination_prior = float(input_dict['inclination_prior'])

system_priors = [(x.mean()-0.5,x.mean()+0.5),(inclination-3*inclination_prior,90),(aRs-3*aRs_prior,aRs+3*aRs_prior)]

### Generate dictionary for TransitModel class

d = OrderedDict()

d['t0'] = Param(x.mean()) # assume t0 is near middle of data
d['inc'] = Param(inclination)
d['ars'] = Param(aRs)
d['k'] = Param(rp_rs)
d['u1'] = Param(0.5) # assumed starting value
d['u2'] = Param(0.2) # assumed starting value

d['c1'] = Param(1.0)

for i in range(1,args.poly_order+1):
    d['c%d'%(i+1)] = Param(0)


### Define the red noise model. Only using time here but polynomial order can be defined
red_noise_model_inputs = [x,None,None,None,None,None]
polynomial_orders = np.array([args.poly_order,0,0,0,0,0])


### Generate starting model
model = TransitModelPM(period,d,red_noise_model_inputs,polynomial_orders,x)

### Optimise the starting model
fitted_model = optimise_model(model,x,y,e,reset_starting_vals=True,sys_bounds=system_priors,fit_method=args.fit_method)

### Print the results
print("\nFitted Rp/R* = %.4f; fitted inc = %.3f; fitted a/R* = %.3f; fitted t0 = %.5f \n"%(fitted_model.pars['k'].currVal,fitted_model.pars['inc'].currVal,fitted_model.pars['ars'].currVal,fitted_model.pars['t0'].currVal))

### Plot the results
print('\nPlotting lsq fit....')
fig = pu.plot_single_model(fitted_model,x,y,e,save_fig=False)

rms = fitted_model.rms(x,y)*1e6
label_location = fig.axes[1].get_xlim()[1] - 1
fig.axes[1].set_title("Residual scatter = %d ppm"%rms)
fig.savefig('wl_fitted_model.png',bbox_inches='tight')
plt.show()

chi2 = fitted_model.reducedChisq(x,y,e)

print('RMS of residuals (ppm) = %f'%rms)
if args.contact1 is not None:
    print('Std out-of-transit (ppm) = %.1f'%(np.std(np.hstack((y[:contact1],y[contact4:])))*1e6))
print('Chi2 of fit = %.1f'%chi2)



### Now perform second fit following the clipping of outliers from the first fit
if args.refit > 0:

    residuals_1 = y - fitted_model.calc(x)
    rms_1 = np.sqrt(np.mean(residuals_1**2))

    keep_idx_2 = [abs(residuals_1) <= args.refit*rms_1]

    # if save_kept_indices is not None:
    #     pickle.dump(keep_idx_2,open('keep_idx_bin%d.pickle'%args.save_kept_indices))

    x = x[abs(residuals_1) <= args.refit*rms_1] # only keep values less than the sigma clipped standard deviation
    y = y[abs(residuals_1) <= args.refit*rms_1] # only keep values less than the sigma clipped standard deviation
    e = e[abs(residuals_1) <= args.refit*rms_1] # only keep values less than the sigma clipped standard deviation


    ### Generate second model, starting with literature values for system parameters but using fitted values for red noise model.
    # Otherwise second fit can fail by refitting with precisely the same values as those resulting from the first fit

    d2 = OrderedDict()

    d2['t0'] = Param(x.mean())
    d2['inc'] = Param(inclination)
    d2['ars'] = Param(aRs)
    d2['k'] = Param(rp_rs)
    d2['u1'] = Param(0.5)
    d2['u2'] = Param(0.2)

    d2['c1'] = Param(1.0)

    for i in range(1,args.poly_order+1):
        d2['c%d'%(i+1)] = Param(fitted_model.pars['c%d'%(i+1)].currVal)


    model2 = TransitModelPM(period,d2,red_noise_model_inputs,polynomial_orders,x)
    fitted_model_2 = optimise_model(model2,x,y,e,reset_starting_vals=True,sys_bounds=system_priors,fit_method=args.fit_method)

    chi2 = fitted_model_2.reducedChisq(x,y,e)

    ### rescale error bars according to reduced chi squared
    e = e*np.sqrt(chi2)

    ### plot second iteration
    print("\n#### Refitting following clipping of residuals ####")
    print('\nPlotting lsq fit....')

    fig = pu.plot_single_model(fitted_model_2,x,y,e,save_fig=False)
    rms = fitted_model_2.rms(x,y)*1e6
    label_location = fig.axes[1].get_xlim()[1] - 1
    fig.axes[1].set_title("Residual scatter = %d ppm"%rms)

    # if args.save_kept_indices is None:
    fig.savefig('wl_fitted_model_refit.png',bbox_inches='tight')
    plt.show()

    chi2 = fitted_model_2.reducedChisq(x,y,e)

    print('RMS of residuals (ppm) = %f'%rms)
    if args.contact1 is not None:
        print('Std out-of-transit (ppm) = %.1f'%(np.std(np.hstack((y[:contact1],y[contact4:])))*1e6))
    print('Chi2 of fit = %.1f'%chi2)

    # else:
    #     plt.show(block=False)
    #     plt.pause(5)
    #     plt.close()




else:
    fitted_model_2 = fitted_model

# if args.save_kept_indices is None:
    ### print final fitted values
print("\nt0 = %.6f; inclination = %.3f; aRs = %.3f; Rp/Rs = %.6f; u1 = %.3f ; u2 = %.3f"%(fitted_model_2.pars['t0'].currVal,fitted_model_2.pars['inc'].currVal,fitted_model_2.pars['ars'].currVal,fitted_model_2.pars['k'].currVal,fitted_model_2.pars['u1'].currVal,fitted_model_2.pars['u2'].currVal))
