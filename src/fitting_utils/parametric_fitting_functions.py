#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
import os
import copy
from collections import OrderedDict
from fitting_utils import TransitModelGPPM as tmgp
from fitting_utils import plotting_utils as pu

def systematics_model(p0,model_inputs,poly_orders,normalise_inputs=False,deconstruct_polys=False):

    """
    Generate a systematics model which is fed any combination of airmass, fwhm, x positions, y positions and sky background.

    Input:
    p0 -- the offset and coefficients of the model. The added offset must *always* be set at index of 0, i.e. p0[0].
    model_inputs -- the ndarray of model inputs, e.g. [time,sky,x,y,...]
    poly_orders -- array of polynomial orders. This must be the same length as model_inputs and is interpreted in the same order as model_inputs, i.e. the first value given to poly_orders operates on the first vector for model_inputs

    example of poly_orders use:
    a cubic airmass polynomial, quadratic fwhm and xpos: model_inputs = [airmass,fwhm,xpos], poly_orders = np.array([3,2,2])

    Returns: the evaluated, combined systematics model as a numpy array.
    if deconstruct_polys = True: it also returns an ndarray of each polynomial contribution to the overall model
    """

    # Ancillary data and poly_orders are ALWAYS in the order:

    offset = p0[0] # offset added to model, which is at the start of p0
    systematics_model = 0 # initiate systematics offset as being zero so that susbequent models can be added together
    current_index = 0
    individual_models = []

    for i in range(len(model_inputs)):
        if poly_orders[i] > 0:
            poly_coefficients = np.hstack((p0[1+current_index:1+current_index+poly_orders[i]],[0]))
            poly = np.poly1d(poly_coefficients)

            if normalise_inputs:
                input_norm = (model_inputs[i]-model_inputs[i].mean())/model_inputs[i].std()
            else:
                input_norm = model_inputs[i]

            poly_eval = poly(input_norm)

            # add to current systematics model
            systematics_model += poly_eval
            individual_models.append(poly_eval)

            current_index += poly_orders[i]

    if deconstruct_polys:
        return systematics_model + offset,individual_models
    else:
        return systematics_model + offset




def fit_all_polynomial_combinations(starting_model,time_input,flux_input,error_input,model_inputs,max_order=4,sys_priors=None):

    """A function that fits all possible combinations of polynomial models.

    This detrends against up to 8 basis vectors, each up to quartic in order.

    Takes as input:

    starting_model -- a TransitModelGPPM object
    time_input -- array of time
    flux_input -- array of flux
    error_input -- array of errors on flux measurements
    model_inputs -- the ndarray of basis vectors
    max_order -- the maximum order of each polynomial to consider. Default = 4
    sys_prios -- the priors on the system parameters. Can be set to None for no priors


    Returns:
    Nothing, but files of successful and unsuccessful fits are saved to a new directory, white_light_parametric_model_fits/.
    The best model is the tail of the successful fits file

    """

    # generate boolean arrays of all possible combinations of True and False for an array of length equal to the number of model inputs
    all_possible_combinations = itertools.product([True,False],repeat=len(model_inputs))

    try:
        os.mkdir("white_light_parametric_model_fits")
    except:
        pass


    # make a reference for the best chi squared value so that subsequent fits can be compared with this
    best_rChi2 = rChi2_cut =  3*starting_model.reducedChisq(time_input,flux_input,error_input)
    best_BIC = starting_model.BIC(time_input,flux_input,error_input)
    best_AIC = starting_model.AIC(time_input,flux_input,error_input)
    best_rms = rms_cut = 1.5*starting_model.rms(time_input,flux_input)
    best_beta = starting_model.red_noise_beta(time_input,flux_input,error_input)
    best_rChi2_model = best_BIC_model = best_AIC_model = best_rms_model = best_beta_model = starting_model.polynomial_orders
    best_rChi2_inputs = best_BIC_inputs = best_AIC_inputs = best_rms_inputs = best_beta_inputs = [True]*len(model_inputs)

    # create results table for successful fits
    results_tab = open('white_light_parametric_model_fits/successful_fits_results_tab.txt','w')
    results_tab.write("Using linear polynomials as reference model...\n")
    results_tab.write("Reference rChi2 = %.2f \n"%best_rChi2)
    results_tab.write("Reference BIC = %.2f \n"%best_BIC)
    results_tab.write("Reference AIC = %.2f \n"%best_AIC)
    results_tab.write("Reference RMS = %d \n"%(best_rms*1e6))
    results_tab.write("Reference red noise Beta = %.4f \n"%best_beta)
    results_tab.write('---\n')
    results_tab.write("Maximum polynomial order considered = %d \n"%max_order)
    results_tab.write('---\n')
    results_tab.close()

    # create results table for failed fits
    results_tab = open('white_light_parametric_model_fits/failed_fits_results_tab.txt','w')
    results_tab.write("Using linear polynomials as reference model...\n")
    results_tab.write("Reference rChi2 = %.2f \n"%best_rChi2)
    results_tab.write("Reference BIC = %.2f \n"%best_BIC)
    results_tab.write("Reference AIC = %.2f \n"%best_AIC)
    results_tab.write("Reference RMS = %d \n"%(best_rms*1e6))
    results_tab.write("Reference red noise Beta = %.4f \n"%best_beta)
    results_tab.write('---\n')
    results_tab.write("Maximum polynomial order considered = %d \n"%max_order)
    results_tab.write('---\n')
    results_tab.close()

    # Now loop over each model input up the maximum order being considered in each.
    # ~ print(all_possible_combinations)
    # ~ width = len(all_possible_combinations)
    ncombinations = 2**len(model_inputs)
    npolys = len(model_inputs-1)*max_order+1
    niterations = ncombinations*npolys
    width = 100

    progress_count = 0
    print('Progress:') # for progress bar

    # now loop over all possible combinations

    for combination in all_possible_combinations:

        n = int((width+1) * float(progress_count) / ncombinations)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n))) # for progress bar
        progress_count += 1

        # Remove combinations that don't result in any detrending being used
        if not np.any(combination):
            continue

        # switch on/off the detrend inputs
        combination = np.array(combination)
        MODEL_INPUTS = model_inputs[combination]
        n_MODEL_INPUTS = len(np.where(combination)[0])

        # We are only considering a maximum of a linear polynomial in time! Therefore time needs to be parsed as the first detrending vector in fitting_input.txt
        if combination[0]: # combination[0] is time, therefore if time is True, set the maximum order to 2
            first_poly_max_order = 2
        else:
            first_poly_max_order = max_order

        for order_1 in range(1,first_poly_max_order+1):

            if n_MODEL_INPUTS > 1:
                for order_2 in range(1,max_order+1):

                    if n_MODEL_INPUTS > 2:
                        for order_3 in range(1,max_order+1):

                            if n_MODEL_INPUTS > 3:
                                for order_4 in range(1,max_order+1):

                                    if n_MODEL_INPUTS > 4:
                                        for order_5 in range(1,max_order+1):

                                            if n_MODEL_INPUTS > 5:
                                                for order_6 in range(1,max_order+1):

                                                    if n_MODEL_INPUTS > 6:
                                                        for order_7 in range(1,max_order+1):

                                                            if n_MODEL_INPUTS > 7:
                                                                for order_8 in range(1,max_order+1):

                                                                    polynomial_orders = np.array([order_1,order_2,order_3,order_4,order_5,order_6,order_7,order_8])
                                                                    current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                                                                    if rChi2 < best_rChi2:
                                                                        best_rChi2 = rChi2
                                                                        best_rChi2_model = current_model.polynomial_orders
                                                                        best_rChi2_inputs = combination

                                                                    if BIC < best_BIC:
                                                                        best_BIC = BIC
                                                                        best_BIC_model = current_model.polynomial_orders
                                                                        best_BIC_inputs = combination

                                                                    if AIC < best_AIC:
                                                                        best_AIC = AIC
                                                                        best_AIC_model = current_model.polynomial_orders
                                                                        best_AIC_inputs = combination

                                                                    if rms < best_rms:
                                                                        best_rms = rms
                                                                        best_rms_model = current_model.polynomial_orders
                                                                        best_rms_inputs = combination

                                                                    if beta < best_beta:
                                                                        best_beta = beta
                                                                        best_beta_model = current_model.polynomial_orders
                                                                        best_beta_inputs = combination

                                                            else:
                                                                polynomial_orders = np.array([order_1,order_2,order_3,order_4,order_5,order_6,order_7])
                                                                current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                                                                if rChi2 < best_rChi2:
                                                                    best_rChi2 = rChi2
                                                                    best_rChi2_model = current_model.polynomial_orders
                                                                    best_rChi2_inputs = combination

                                                                if BIC < best_BIC:
                                                                    best_BIC = BIC
                                                                    best_BIC_model = current_model.polynomial_orders
                                                                    best_BIC_inputs = combination

                                                                if AIC < best_AIC:
                                                                    best_AIC = AIC
                                                                    best_AIC_model = current_model.polynomial_orders
                                                                    best_AIC_inputs = combination

                                                                if rms < best_rms:
                                                                    best_rms = rms
                                                                    best_rms_model = current_model.polynomial_orders
                                                                    best_rms_inputs = combination

                                                                if beta < best_beta:
                                                                    best_beta = beta
                                                                    best_beta_model = current_model.polynomial_orders
                                                                    best_beta_inputs = combination

                                                    else:
                                                        polynomial_orders = np.array([order_1,order_2,order_3,order_4,order_5,order_6])
                                                        current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                                                        if rChi2 < best_rChi2:
                                                            best_rChi2 = rChi2
                                                            best_rChi2_model = current_model.polynomial_orders
                                                            best_rChi2_inputs = combination

                                                        if BIC < best_BIC:
                                                            best_BIC = BIC
                                                            best_BIC_model = current_model.polynomial_orders
                                                            best_BIC_inputs = combination

                                                        if AIC < best_AIC:
                                                            best_AIC = AIC
                                                            best_AIC_model = current_model.polynomial_orders
                                                            best_AIC_inputs = combination

                                                        if rms < best_rms:
                                                            best_rms = rms
                                                            best_rms_model = current_model.polynomial_orders
                                                            best_rms_inputs = combination

                                                        if beta < best_beta:
                                                            best_beta = beta
                                                            best_beta_model = current_model.polynomial_orders
                                                            best_beta_inputs = combination

                                            else:
                                                polynomial_orders = np.array([order_1,order_2,order_3,order_4,order_5])
                                                current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                                                if rChi2 < best_rChi2:
                                                    best_rChi2 = rChi2
                                                    best_rChi2_model = current_model.polynomial_orders
                                                    best_rChi2_inputs = combination

                                                if BIC < best_BIC:
                                                    best_BIC = BIC
                                                    best_BIC_model = current_model.polynomial_orders
                                                    best_BIC_inputs = combination

                                                if AIC < best_AIC:
                                                    best_AIC = AIC
                                                    best_AIC_model = current_model.polynomial_orders
                                                    best_AIC_inputs = combination

                                                if rms < best_rms:
                                                    best_rms = rms
                                                    best_rms_model = current_model.polynomial_orders
                                                    best_rms_inputs = combination

                                                if beta < best_beta:
                                                    best_beta = beta
                                                    best_beta_model = current_model.polynomial_orders
                                                    best_beta_inputs = combination


                                    else:
                                        polynomial_orders = np.array([order_1,order_2,order_3,order_4])
                                        current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                                        if rChi2 < best_rChi2:
                                            best_rChi2 = rChi2
                                            best_rChi2_model = current_model.polynomial_orders
                                            best_rChi2_inputs = combination

                                        if BIC < best_BIC:
                                            best_BIC = BIC
                                            best_BIC_model = current_model.polynomial_orders
                                            best_BIC_inputs = combination

                                        if AIC < best_AIC:
                                            best_AIC = AIC
                                            best_AIC_model = current_model.polynomial_orders
                                            best_AIC_inputs = combination

                                        if rms < best_rms:
                                            best_rms = rms
                                            best_rms_model = current_model.polynomial_orders
                                            best_rms_inputs = combination

                                        if beta < best_beta:
                                            best_beta = beta
                                            best_beta_model = current_model.polynomial_orders
                                            best_beta_inputs = combination

                            else:
                                polynomial_orders = np.array([order_1,order_2,order_3])
                                current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                                if rChi2 < best_rChi2:
                                    best_rChi2 = rChi2
                                    best_rChi2_model = current_model.polynomial_orders
                                    best_rChi2_inputs = combination

                                if BIC < best_BIC:
                                    best_BIC = BIC
                                    best_BIC_model = current_model.polynomial_orders
                                    best_BIC_inputs = combination

                                if AIC < best_AIC:
                                    best_AIC = AIC
                                    best_AIC_model = current_model.polynomial_orders
                                    best_AIC_inputs = combination

                                if rms < best_rms:
                                    best_rms = rms
                                    best_rms_model = current_model.polynomial_orders
                                    best_rms_inputs = combination

                                if beta < best_beta:
                                    best_beta = beta
                                    best_beta_model = current_model.polynomial_orders
                                    best_beta_inputs = combination


                    else:
                        polynomial_orders = np.array([order_1,order_2])
                        current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                        if rChi2 < best_rChi2:
                            best_rChi2 = rChi2
                            best_rChi2_model = current_model.polynomial_orders
                            best_rChi2_inputs = combination

                        if BIC < best_BIC:
                            best_BIC = BIC
                            best_BIC_model = current_model.polynomial_orders
                            best_BIC_inputs = combination

                        if AIC < best_AIC:
                            best_AIC = AIC
                            best_AIC_model = current_model.polynomial_orders
                            best_AIC_inputs = combination

                        if rms < best_rms:
                            best_rms = rms
                            best_rms_model = current_model.polynomial_orders
                            best_rms_inputs = combination

                        if beta < best_beta:
                            best_beta = beta
                            best_beta_model = current_model.polynomial_orders
                            best_beta_inputs = combination


            else:
                polynomial_orders = np.array([order_1])
                current_model,rChi2,BIC,AIC,rms,beta = set_and_fit_model(starting_model,polynomial_orders,MODEL_INPUTS,time_input,flux_input,error_input,sys_priors,rms_cut,combination)

                if rChi2 < best_rChi2:
                    best_rChi2 = rChi2
                    best_rChi2_model = current_model.polynomial_orders
                    best_rChi2_inputs = combination

                if BIC < best_BIC:
                    best_BIC = BIC
                    best_BIC_model = current_model.polynomial_orders
                    best_BIC_inputs = combination

                if AIC < best_AIC:
                    best_AIC = AIC
                    best_AIC_model = current_model.polynomial_orders
                    best_AIC_inputs = combination

                if rms < best_rms:
                    best_rms = rms
                    best_rms_model = current_model.polynomial_orders
                    best_rms_inputs = combination

                if beta < best_beta:
                    best_beta = beta
                    best_beta_model = current_model.polynomial_orders
                    best_beta_inputs = combination

    # make it clear which is the best model by saving at the bottom of the table
    results_tab = open('white_light_parametric_model_fits/successful_fits_results_tab.txt','a')
    results_tab.write('\n\n**********\n')
    results_tab.write('BEST REDUCED CHI2 = %.2f; BEST INPUTS USED = %s; BEST POLYNOMIAL ORDERS = %s\n'%(best_rChi2,best_rChi2_inputs,best_rChi2_model))
    results_tab.write('\n\n**********\n')
    results_tab.write('BEST BIC = %.2f; BEST INPUTS USED = %s; BEST BIC MODEL = %s\n'%(best_BIC,best_BIC_inputs,best_BIC_model))
    results_tab.write('\n\n**********\n')
    results_tab.write('BEST AIC = %.2f; BEST INPUTS USED = %s; BEST AIC MODEL = %s\n'%(best_AIC,best_AIC_inputs,best_AIC_model))
    results_tab.write('\n\n**********\n')
    results_tab.write('BEST RMS = %d; BEST INPUTS USED = %s; BEST RMS MODEL = %s\n'%(best_rms*1e6,best_rms_inputs,best_rms_model))
    results_tab.write('\n\n**********\n')
    results_tab.write('BEST RED NOISE BETA = %.4f; BEST INPUTS USED = %s; BEST BETA MODEL = %s\n'%(best_beta,best_beta_inputs,best_beta_model))
    results_tab.close()

    return


def save_results(fitted_model,time_input,flux_input,error_input,rms_cut,inputs_used):
    """A function used by fit_all_polynomial_combinations to save the output from each combination."""

    rChi2 = fitted_model.reducedChisq(time_input,flux_input,error_input)
    BIC = fitted_model.BIC(time_input,flux_input,error_input)
    AIC = fitted_model.AIC(time_input,flux_input,error_input)
    rms = fitted_model.rms(time_input,flux_input)
    beta = fitted_model.red_noise_beta(time_input,flux_input,error_input)

    # if the fit is successful and passes the chi squared cut save in successful fits output
    if rms < rms_cut:
        results_tab = open('white_light_parametric_model_fits/successful_fits_results_tab.txt','a')
        success = True
    # if unsuccessful, save to failed results table
    else:
        results_tab = open('white_light_parametric_model_fits/failed_fits_results_tab.txt','a')
        success = False

    polynomial_orders_used = fitted_model.polynomial_orders

    results_tab.write('---\nInput vectors used = %s \n'%str(inputs_used))
    results_tab.write('Polynomial orders used = %s \n'%str(polynomial_orders_used))
    results_tab.write('Fitted coefficients = %s \n'%([fitted_model.pars['c%d'%i].currVal for i in range(1,fitted_model.polynomial_orders.sum()+2)]))
    results_tab.write('Reduced chisquared = %.2f \n'%rChi2)
    results_tab.write('Standard deviation of residuals = %d ppm \n'%(rms*1e6))
    results_tab.write('BIC = %.2f \n'%BIC)
    results_tab.write('AIC = %.2f \n'%AIC)
    results_tab.write('RMS = %d \n'%(rms*1e6))
    results_tab.write('Red noise Beta = %.4f \n'%beta)

    results_tab.close()

    # now generate and save plots for successful fits which pass the chi squared cut
    # ~ if success:
        # ~ pu.plot_single_model(fitted_model,time_input,flux_input,error_input,save_fig=True,deconstruct=True)
        # ~ os.system("mv fitted_model.pdf white_light_parametric_model_fits/fitted_model_%s.pdf"%(str(fitted_model.polynomial_orders).join('.')))
    return


def set_and_fit_model(starting_model,polynomial_orders,model_inputs,time_input,flux_input,error_input,sys_priors,rms_cut,inputs_used):
    """A function used by fit_all_polynomial_combinations to generate and fit the new model for each combination of polynomial."""
    d = OrderedDict()
    for k,v in zip(starting_model.pars.keys(),starting_model.pars.values()):
        if k[0] != "c":
            d[k] = v

    d['c1'] = tmgp.Param(1.0)
    for i in range(1,polynomial_orders.sum()+1):
        d['c%d'%(i+1)] = tmgp.Param(1e-3)

    current_model = tmgp.TransitModelGPPM(d,model_inputs,None,error_input,time_input,None,False,False,starting_model.ld_std_priors,polynomial_orders,starting_model.ld_law)

    fitted_model,_ = current_model.optimise_params(time_input,flux_input,error_input,full_model=True,sys_priors=sys_priors,verbose=False)

    # Rescale uncertainties to give rChi2 = 1 - this wipes out the usefulness of the chi2 but makes RMS, BIC, AIC and red noise beta more comparable
    rescaled_errors = error_input*np.sqrt(fitted_model.reducedChisq(time_input,flux_input,error_input))

    # Refit the data
    fitted_model,_ = fitted_model.optimise_params(time_input,flux_input,rescaled_errors,full_model=True,sys_priors=sys_priors,verbose=False)

    save_results(fitted_model,time_input,flux_input,rescaled_errors,rms_cut,inputs_used)

    rChi2 = fitted_model.reducedChisq(time_input,flux_input,rescaled_errors)
    BIC = fitted_model.BIC(time_input,flux_input,rescaled_errors)
    AIC = fitted_model.AIC(time_input,flux_input,rescaled_errors)
    rms = fitted_model.rms(time_input,flux_input,rescaled_errors)
    beta = fitted_model.red_noise_beta(time_input,flux_input,rescaled_errors)

    return current_model,rChi2,BIC,AIC,rms,beta
