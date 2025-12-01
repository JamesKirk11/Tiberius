#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

from ldtk import LDPSetCreator, BoxcarFilter
import pickle
import numpy as np
import argparse
from scipy.ndimage import median_filter as MF
import matplotlib.pyplot as plt
from global_utils import parseInput


parser = argparse.ArgumentParser(description='Generate the limb darkening coefficients for the star evaluated for the wavelength bins provided. This used Limb Darkening Toolkit. Note: The stellar effect temperature, stellar log(g), stellar [Fe/H] and associated errors must be in the fitting_input.txt file for this code to execute. Also, the arrays of wavelength bin centres and widths must be defined in fitting_input.txt. This code returns the quadratic limb darkening coefficients and errors evaluated for each wavelength bin in a file called LD_coefficients.dat.')
parser.add_argument('-seq','--seq',help="""use this argument if wanting to calculate multiple limb darkening coefficients sequentially (necessary for incrementally increasing Na & K bins)""",action='store_true')
parser.add_argument('-ld_law','--ld_law',help="""Use this to define the limb darkening law: quadratic (default), linear, nonlinear, squareroot""",default='quadratic')
parser.add_argument('-um','--microns',help="""Use this if the wavelengths are given in microns""",action='store_true')
parser.add_argument('-exotic','--use_exotic',help="""Use this if wanting to use Exo-TIC instead of LDTk""",action='store_true')
parser.add_argument('-ld_ndim','--ld_model_dimensionality',help="""If using Exo-TIC, define whether we're using 1D or 3D models""")
parser.add_argument('-instrument','--instrument',help="""If using Exo-TIC, define what instrument we're using. e.g. 'JWST_NIRSpec_prism', 'JWST_NIRSpec_G395H', 'JWST_MIRI_LRS', 'JWST_NIRCam_F444'""")
args = parser.parse_args()

if args.use_exotic:
    from exotic_ld import StellarLimbDarkening

    # Stellar models: 1D or 3D grid.
    ld_model_dimensionality = args.ld_model_dimensionality

    # Path to the installed data.
    ld_data_path = '/Users/james/python/ExoTiC-LD_data'

    # instrument, see https://exotic-ld.readthedocs.io/en/latest/views/supported_instruments.html for full list of supported instruments
    instrument_mode = args.instrument



# Load in conrolling parameter file
input_dict = parseInput('fitting_input.txt')

# Define whether the coefficients are for the white light curve, in which case we're dealing with a single passband.
white_light_fit = bool(int(input_dict['white_light_fit']))

if white_light_fit:
    wavelength_centres = float(input_dict['wvl_centres'])
    wvl_bin_full_width = float(input_dict['wvl_bin_full_width'])
else:
    wavelength_centres = pickle.load(open(input_dict['wvl_centres'],'rb'))
    wvl_bin_full_width = pickle.load(open(input_dict['wvl_bin_full_width'],'rb'))

    nbins = len(wavelength_centres)

if args.microns: # have to change to Angstroms
    wavelength_centres *= 1e4
    wvl_bin_full_width *= 1e4

# Check that no wavelengths are beyond LDTk's upper limit. If so (e.g. for MIRI), we will have to use a different ExoTIC-LD to estimate the LDCs
if not args.use_exotic:
	if white_light_fit:
	    if 1e-4*(wavelength_centres + wvl_bin_full_width/2) > 5.5:
	        raise ValueError("Desired maximum wavelength is beyond LDTk's limit of 5.5um. Try using https://exotic-ld.readthedocs.io/en/latest/views/installation.html instead")
	else:
	    if 1e-4*(wavelength_centres[-1]+wvl_bin_full_width[-1]/2) > 5.5:
	        raise ValueError("Desired maximum wavelength is beyond LDTk's limit of 5.5um. Try using https://exotic-ld.readthedocs.io/en/latest/views/installation.html instead")


### Load in stellar parameters from fitting_input.txt

Teff, Teff_err = float(input_dict['Teff']),float(input_dict['Teff_err'])
logg_star, logg_star_err = float(input_dict['logg_star']),float(input_dict['logg_star_err'])
FeH, FeH_err = float(input_dict['FeH']),float(input_dict['FeH_err'])

error_inflation = float(input_dict['ld_uncertainty_multiplier']) # Note: this number is used to inflate the errors in LDCs in case we think errors on stellar parameters are underestimated.


### Define functions used by LDTk to generate coefficients

def ld_initialise(Teff,Teff_err,logg,logg_err,Z,Z_err,wvl_centre,wvl_error,ld_uncertainty_multiplier=3):
    """
    Function to generate the LDTk model for a list of wavelength bins.

    Inputs:
    Teff - effective temperature of the star in K
    Teff_err - error in the star's effective temperature in K
    logg - the star's surface gravity in c.g.s.
    logg_err - the error in the star's surface gravity, in c.g.s.
    Z - the metallicity [Fe/H] of the star
    Z_err - the error in [Fe/H]
    wvl_centre - the central wavelengths of the wavelength bins under consideration, in Angstroms
    wvl_error - the full width of the wavelength bins under consideration, in Angstroms
    ld_uncertainty_multiplier - the factor by which to multiply the errors in the stellar parameters. This is a conservative approach to estimating the limb darkening coefficients. Default=3

    Returns:
    ps - the LDTk profile object
    """
    # Need to convert wavelengths from Angstroms to nm
    filters = [BoxcarFilter('%s'%i,c-e//2,c+e//2) for i,(c,e) in enumerate(zip(wvl_centre/10.,wvl_error/10.))]

    # find the maximum resolution (minimum wavelength spacing) in nm
    resolution = np.diff(wvl_centre).min()/10.

    # find the maximum wavelength considered in nm
    max_wvl = wvl_centre.max()/10.

    if max_wvl > 2600:
        model_set = "visir"
    else:
        model_set = "vis"

    if resolution > 5:
        model_set += "-lowres"

    sc = LDPSetCreator(teff=(Teff,Teff_err),logg=(logg,logg_err),z=(Z,Z_err),filters=filters,dataset=model_set)#,force_download=True)

    ps = sc.create_profiles()
    ps.set_uncertainty_multiplier(ld_uncertainty_multiplier)

    return ps


def single_ld_model(Teff,Teff_err,logg,logg_err,Z,Z_err,wvl_min,wvl_max,ld_uncertainty_multiplier=3):
    """
    Function to generate the LDTk model for a single wavelength bin (e.g. the white light curve).

    Inputs:
    Teff - effective temperature of the star in K
    Teff_err - error in the star's effective temperature in K
    logg - the star's surface gravity in c.g.s.
    logg_err - the error in the star's surface gravity, in c.g.s.
    Z - the metallicity [Fe/H] of the star
    Z_err - the error in [Fe/H]
    wvl_min - the blue edge of the wavelength bin
    wvl_max - the red edge of the wavelength bin
    ld_uncertainty_multiplier - the factor by which to multiply the errors in the stellar parameters. This is a conservative approach to estimating the limb darkening coefficients. Default=3

    Returns:
    ps - the LDTk profile object
    """

    # Need to convert wavelengths from Angstroms to nm
    filters = [BoxcarFilter('a',wvl_min/10.,wvl_max/10.)]

    sc = LDPSetCreator(teff=(Teff,Teff_err),logg=(logg,logg_err),z=(Z,Z_err),filters=filters)#,force_download=True)

    ps = sc.create_profiles()
    ps.set_uncertainty_multiplier(ld_uncertainty_multiplier)

    return ps


def return_ld_components(ld_mod,ld_law,MCMC=True):
    """The function that returns the quadratic limb darkening coefficients for the LDTk profile object.

    Inputs:
    ld_mod - the LDTk profile object ('ps')
    ld_law - str - either 'quadratic', 'linear', 'nonlinear', 'squareroot'
    MCMC - True/False - use MCMC to perform parameter estimation? Default=True

    Returns:
    coeffs - the quadratic (u1 & u2) limb darkening coefficients for all bins
    errors - the uncertainties in u1 & u2"""

    if ld_law == 'quadratic':
        coeffs,errors = ld_mod.coeffs_qd(do_mc=False)        # Estimate quadratic law LD_coefficients
    elif ld_law == "linear":
        coeffs,errors = ld_mod.coeffs_ln(do_mc=MCMC)
    elif ld_law == "nonlinear":
        coeffs,errors = ld_mod.coeffs_nl(do_mc=MCMC)
    elif ld_law == "squareroot":
        coeffs,errors = ld_mod.coeffs_sq(do_mc=MCMC)
    else:
        return NameError("args.ld_law must be one of quadratic/linear/nonlinear/squareroot")

    return coeffs,errors


def exotic_ldcs(stellar_params,instrument_mode,wvl_centre,wvl_error,ld_law,ld_model,ld_data_path):

    M_H, Teff, logg = stellar_params

    sld = StellarLimbDarkening(M_H, Teff, logg, ld_model, ld_data_path)

    wvl_centre = np.atleast_1d(wvl_centre)
    wvl_error = np.atleast_1d(wvl_error)

    nbins = len(wvl_centre)

    coeffs = []

    for i in range(nbins):
        # Start and end of wavelength interval [angstroms].
        wavelength_range = [wvl_centre[i]-wvl_error[i]/2,wvl_centre[i]+wvl_error[i]/2]

        if ld_law == "linear":
            c = sld.compute_linear_ld_coeffs(wavelength_range, instrument_mode)

        if ld_law == "quadratic":
            c = sld.compute_quadratic_ld_coeffs(wavelength_range, instrument_mode)

        if ld_law == "nonlinear":
            c = sld.compute_4_parameter_non_linear_ld_coeffs(wavelength_range, instrument_mode)

        coeffs.append(c)

    return np.array(coeffs),np.zeros_like(coeffs)



if not args.use_exotic:
    print('Generating LDTk model...')

if white_light_fit: # we're only calculating coefficients for a single wavelength bin
    if not args.use_exotic:
        ld_model = single_ld_model(Teff,Teff_err,logg_star,logg_star_err,FeH,FeH_err,wavelength_centres-wvl_bin_full_width//2,wavelength_centres+wvl_bin_full_width//2,error_inflation)

else: # we're considering multiple bins

    if args.seq: # calculate single model each time

        u1,u1e,u2,u2e,u3,u3e,u4,u4e = [],[],[],[],[],[],[],[]

        for i in range(nbins):

            if not args.use_exotic:
                ld_model = single_ld_model(Teff,Teff_err,logg_star,logg_star_err,FeH,FeH_err,wavelength_centres[i]-wvl_bin_full_width[i]//2,wavelength_centres[i]+wvl_bin_full_width[i]//2,error_inflation)
                print('....LDTk model loaded for bin %d/%d \n'%(i+1,nbins))

                coeffs,errors = return_ld_components(ld_model,args.ld_law,MCMC=True)
                print('....coefficients calculated for bin %d/%d \n'%(i+1,nbins))

            else:
                coeffs,errors = exotic_ldcs([FeH,Teff,logg_star],instrument_mode,wavelength_centres[i],wvl_bin_full_width[i],args.ld_law,ld_model_dimensionality,ld_data_path)
                print('....coefficients calculated for bin %d/%d \n'%(i+1,nbins))

            u1.append(coeffs[0][0])
            u1e.append(errors[0][0])

            if args.ld_law != "linear":
                u2.append(coeffs[0][1])
                u2e.append(errors[0][1])
            else:
                u2.append(-99) # use -99 as flag that this value is not used
                u2e.append(-99)

            if args.ld_law == "nonlinear":
                u3.append(coeffs[0][2])
                u3e.append(errors[0][2])

                u4.append(coeffs[0][3])
                u4e.append(errors[0][3])
            else:
                u3.append(-99)
                u3e.append(-99)

                u4.append(-99)
                u4e.append(-99)


    else:
        if not args.use_exotic:
            ld_model = ld_initialise(Teff,Teff_err,logg_star,logg_star_err,FeH,FeH_err,wavelength_centres,wvl_bin_full_width,error_inflation)


def replace_negatives_with_median(arr):
    arr = arr.copy()
    n = len(arr)
    nreplacements = 0

    for i in range(n):
        if arr[i] < 0:
            nreplacements += 1
            # Find nearest non-negative value to the left
            left = None
            for j in range(i - 1, -1, -1):
                if arr[j] >= 0:
                    left = arr[j]
                    break

            # Find nearest non-negative value to the right
            right = None
            for j in range(i + 1, n):
                if arr[j] >= 0:
                    right = arr[j]
                    break

            # Determine replacement
            if left is not None and right is not None:
                arr[i] = np.median([left, right])
            elif left is not None:
                arr[i] = left
            elif right is not None:
                arr[i] = right
            else:
                raise ValueError("No non-negative values found in array")

    print("%d negative LD coefficients replaced"%nreplacements)

    return arr

if not args.seq:
    if not args.use_exotic:
        print('....LDTk model loaded \n')

        print('Calculating coefficients...')
        coeffs,errors = return_ld_components(ld_model,args.ld_law)
    else:
        print('Calculating coefficients...')
        coeffs,errors = exotic_ldcs([FeH,Teff,logg_star],instrument_mode,wavelength_centres,wvl_bin_full_width,args.ld_law,ld_model_dimensionality,ld_data_path)

    u1,u1e = coeffs[:,0],errors[:,0]
    u1 = replace_negatives_with_median(u1)

    if args.ld_law != "linear":
        u2,u2e = coeffs[:,1],errors[:,1]
        u2 = replace_negatives_with_median(u2)
    else:
        u2 = [-99]*len(u1) # pad with blank space
        u2e = [-99]*len(u1)

    if args.ld_law == "nonlinear":
        u3,u3e = coeffs[:,2],errors[:,2]
        u4,u4e = coeffs[:,3],errors[:,3]
    else:
        u3 = [-99]*len(u1) # pad with blank space
        u3e = [-99]*len(u1)

        u4 = [-99]*len(u1) # pad with blank space
        u4e = [-99]*len(u1)



### Save results to table

if args.microns:
    wavelength_centres /= 1e4
    wvl_bin_full_width /= 1e4

def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def smooth_ld(w,u,ue):

    if np.all(np.array(u) == -99):
        return u,ue

    if len(u) < 100:
        box_width = 3
    else:
        box_width = round_up_to_odd(len(u)/100)
    if box_width < 3:
        box_width = 3

    running_median = MF(u,box_width)
    u_poly = np.poly1d(np.polyfit(w,running_median,2))

    running_median_up = MF(u+ue,box_width)
    ue_poly_up = np.poly1d(np.polyfit(w,running_median_up,2))

    running_median_lo = MF(u-ue,box_width)
    ue_poly_lo = np.poly1d(np.polyfit(w,running_median_lo,2))

    smoothed_u = u_poly(w)
    smoothed_ue_up = ue_poly_up(w)
    smoothed_ue_lo = ue_poly_lo(w)
    smoothed_ue = np.mean((smoothed_ue_up-smoothed_u,smoothed_u-smoothed_ue_lo),axis=0)

    return smoothed_u,smoothed_ue

tab = open('LD_coefficients.txt','w')
tab.write('# Teff = %d +/- %.2f K ; log(g) = %.2f +/- %.2f ; FeH = %.2f +/- %.2f ; u error inflation factor = %.1f \n'%(Teff,Teff_err,logg_star,logg_star_err,FeH,FeH_err,error_inflation))
tab.write('# %s law used \n'%(args.ld_law))
if args.use_exotic:
	tab.write("# Exo-TIC-LD used with a %s model for instrument %s \n"%(args.ld_model_dimensionality,args.instrument))
else:
	tab.write("# LDTk used")
tab.write('# Wavelength | Width | u1 | u1 error | u2 | u2 error | u3 | u3 error | u4 | u4 error |\n')

if white_light_fit:
    tab.write('%f %f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n'%(wavelength_centres,wvl_bin_full_width,u1[0],u1e[0],u2[0],u2e[0],u3[0],u3e[0],u4[0],u4e[0]))
else:
    tab.write("# %d wavelength bins \n"%len(wavelength_centres))
    smoothed_tab = open('LD_coefficients_smoothed.txt','w')
    smoothed_tab.write('# Teff = %d +/- %.2f K ; log(g) = %.2f +/- %.2f ; FeH = %.2f +/- %.2f ; u error inflation factor = %.1f \n'%(Teff,Teff_err,logg_star,logg_star_err,FeH,FeH_err,error_inflation))
    smoothed_tab.write('# %s law used \n'%(args.ld_law))
    if args.use_exotic:
    	smoothed_tab.write("# Exo-TIC-LD used with a %s model for instrument %s \n"%(args.ld_model_dimensionality,args.instrument))
    else:
    	smoothed_tab.write("# LDTk used")
    smoothed_tab.write("# Quadratic polynomial was used to smooth the limb darkening coefficients \n")
    smoothed_tab.write("# %d wavelength bins \n"%len(wavelength_centres))
    smoothed_tab.write('# Wavelength | Width | u1 | u1 error | u2 | u2 error | u3 | u3 error | u4 | u4 error |\n')

    u1_smoothed,u1e_smoothed = smooth_ld(wavelength_centres,u1,u1e)
    u2_smoothed,u2e_smoothed = smooth_ld(wavelength_centres,u2,u2e)
    u3_smoothed,u3e_smoothed = smooth_ld(wavelength_centres,u3,u3e)
    u4_smoothed,u4e_smoothed = smooth_ld(wavelength_centres,u4,u4e)

    for i in range(nbins):
        tab.write('%f %f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n'%(wavelength_centres[i],wvl_bin_full_width[i],u1[i],u1e[i],u2[i],u2e[i],u3[i],u3e[i],u4[i],u4e[i]))
        smoothed_tab.write('%f %f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n'%(wavelength_centres[i],wvl_bin_full_width[i],u1_smoothed[i],u1e_smoothed[i],u2_smoothed[i],u2e_smoothed[i],u3_smoothed[i],u3e_smoothed[i],u4_smoothed[i],u4e_smoothed[i]))

    tab.close()
    smoothed_tab.close()

    plt.figure()
    plt.plot(wavelength_centres,u1,label="u1")
    plt.plot(wavelength_centres,u1_smoothed,label="u1_smoothed")
    plt.fill_between(wavelength_centres,u1_smoothed+u1e_smoothed,u1_smoothed-u1e_smoothed,color="gray",alpha=0.5)
    plt.plot(wavelength_centres,u2,label="u2")
    plt.plot(wavelength_centres,u2_smoothed,label="u2_smoothed")
    plt.fill_between(wavelength_centres,u2_smoothed+u2e_smoothed,u2_smoothed-u2e_smoothed,color="gray",alpha=0.5)
    plt.legend()
    plt.xlabel("Wavelength")
    plt.ylabel("Coefficient value")
    plt.savefig("LD_model_values.png",bbox_inches="tight",dpi=360)
    plt.show()


### Pickle LDTk model in case we need it later
if not args.seq and not args.use_exotic:
    pickle.dump(ld_model,open('ldtk_model.pickle','wb'))
