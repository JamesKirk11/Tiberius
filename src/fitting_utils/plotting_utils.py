#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import astropy.constants as c
import pickle
import glob
from matplotlib.ticker import AutoMinorLocator
from global_utils import parseInput
from fitting_utils import mcmc_utils as mc
from scipy.stats import chi2 as c2
from scipy.special import erfinv


### FUNCTIONS USEFUL FOR THE PLOTTING OF DATA


def mjd2hours(x,t0):
    """Use time of midtransit to return time array as time from centre of transit in hours.

    Input:
    x - array of times
    t0 - the time of mid-transit

    Returns:
    array of times given in hours respective to mid-transit time"""
    t0_subtracted = x - t0
    return t0_subtracted*24



def calc_scale_height(g,Teq):
    """
    Caculate the scale height of the planet given its surface gravity and equilibrium temperature

    Input:
    g - surface gravity in m/s/s
    Teq - equilbrium temperature in K

    Returns:
    scale height (float), in metres
    """

    mu = 2.3 # g/mole, assuming Jupiter's mean molecular mass
    H = ((1.38e-23)*Teq)/(mu*(1.67e-27)*g) # Scale height, in metres
    return H



def straight_line(x,m,c):
    """This is the function which fits a straight line to Rp vs ln(lambda) where the slope is given by -4*H

    Input:
    x - ln(lambda)
    m - gradient
    c - y intercept

    Returns:
    mx + c"""

    return m*x + c



def chi2_trans_models(model,data,up_error,low_error,NDOF=0):

    """Using the error depending on whether model is above or below data point.

    Input:
    model - transmission spectrum model
    data - array of Rp/Rs values
    up_error - array of positive errors on Rp/Rs values
    low_error - array of negative errors on Rp/Rs values
    NDOF - number of degrees of freedom

    Returns:
    (chi squared of transmission model, reduced chi2 of transmission model)"""

    resids = model - data
    ey = []
    for i, r in enumerate(resids):
        if r >= 0:
            ey.append(up_error[i])
        else:
            ey.append(low_error[i])

    ey = np.array(ey)
    chi_squared = np.sum((resids/ey)**2)
    reduced_chi_squared = chi_squared/(len(data) - NDOF - 1)
    print("Chi2 = %.2f ; Reduced Chi2 = %.2f ; NDOF = %d"%(np.round(chi_squared,2),np.round(reduced_chi_squared,2),len(data) - NDOF - 1))
    return chi_squared,reduced_chi_squared




def rayleigh_slope(Teq,logg,Rp,Rstar,k,k_up,k_low,wvl_bin_centres,save_output=False,verbose=False):
    """Generate a Rayleigh scattering slope given the planet and host stars parameters. Note: currently not tested for new transit model classes.

    Inputs:
    Teq - equilbrium temperature of the planet, in K
    logg - surface gravity of the planet, in c.g.s.
    Rp - radius of the planet, in Jupiter radii
    Rstar - radius of the star, in solar radii
    k - the list of Rp/Rs as measured for the transmission spectrum
    k_up - the positive errors in k
    k_low - the negative errors in k
    wvl_bin_centres - the centres of the wavelength bins as used in the transmission spectrum, in Angstroms
    save_output - True/False - use this to save the chi squared of fits of a flat line, Rayleigh slope at Teq and a Rayleigh slope with a fitted temperature to a table called 'trans_models_statistics.txt'. Default=False
    verbose - True/False - use this to plot the different slopes along with the transmission spectrum. Default=False

    Returns:
    resampled_x - a finer sampling of the wavelength inputs, for nicer plots
    equilibrium_slope_resampled - the Rayleigh slope at Teq sampled at the finer wavelength scale
    fitted_slope_resampled - the Rayleigh slope at the fitted temperature sampled at the finer wavelength scale
    flat_line - the flat (cloudy) transmission spectrum, evaluated at the wvl_bin_centres"""

    g = (10**logg)/100
    rp = Rp*c.R_jup.value
    rs = Rstar*c.R_sun.value

    H = calc_scale_height(g,Teq)
    scale_height_radius = rp/H
    rp_rs = k
    rp_H = rp_rs*rs/H - scale_height_radius
    rs_H = rs/H

    # To initialise Rayleigh fits
    guess_slope = -4*H
    guess_intercept = max(k*rs)
    rp_error = (np.mean((k_up,k_low),axis=0)/k)*(k*rs) # weights on fit
    xdata = np.log(wvl_bin_centres*1e-10) # Need log(lambda) for Rayleigh fit xdata
    ydata = (k*rs)

    popt,pcov=curve_fit(straight_line,xdata,ydata,p0=[guess_slope,guess_intercept],sigma=rp_error)

    H_fit = popt[0]/-4.
    T_fit = (popt[0]*2.3*1.67e-27*g)/(-4*1.38e-23)
    perr = np.sqrt(np.diag(pcov))
    T_fit_err = abs(perr[0]/popt[0])*T_fit

    print('Fitted T of whole range = %d +/- %d'%(T_fit,abs(T_fit_err)))

    # Resample x axis for better plotting
    resampled_x = np.linspace(2000,12000,30)

    fitted_slope_resampled = -4*H_fit*np.log(resampled_x)/rs

    fitted_slope = -4*H_fit*np.log(wvl_bin_centres)/rs
    offset_fit = k.mean()-fitted_slope.mean()

    fitted_slope = fitted_slope + offset_fit # offset applied
    fitted_slope_resampled = fitted_slope_resampled + offset_fit # offset applied
    chi2_Tfit,rchi2_Tfit = chi2_trans_models(fitted_slope,k,k_up,k_low,NDOF=1)
    BIC_Tfit = chi2_Tfit+1*np.log(len(k))

    equilibrium_slope_resampled = -4*H*np.log(resampled_x)/rs


    equilibrium_slope = -4*H*np.log(wvl_bin_centres)/rs
    offset_eq = (k.mean()-equilibrium_slope.mean())

    equilibrium_slope = equilibrium_slope + offset_eq # offset applied
    equilibrium_slope_resampled = equilibrium_slope_resampled + offset_eq
    chi2_Teq,rchi2_Teq = chi2_trans_models(equilibrium_slope,k,k_up,k_low)

    flat_line = np.average(k,weights=1/(np.sqrt(k_up**2+k_low**2)))
    flat_line = np.array([flat_line]*len(k))
    chi2_flat,rchi2_flat = chi2_trans_models(flat_line,k,k_up,k_low)

    if verbose:
        plt.figure()
        plt.errorbar(wvl_bin_centres,k,yerr=(k_low,k_up),fmt='o',color='k',ecolor='k')
        plt.plot(resampled_x,equilibrium_slope_resampled,'g--')
        plt.plot(wvl_bin_centres,equilibrium_slope,'g')
        plt.plot(resampled_x,fitted_slope_resampled,'r--')
        plt.plot(wvl_bin_centres,fitted_slope,'r')
        plt.plot(wvl_bin_centres,flat_line,'b')
        plt.xlabel('Wavelength ($AA$)')
        plt.ylabel('$R_P/R_*$')
        plt.show()

    print("Chi2 of Rayleigh at T_eq = %f; reduced chi2 = %f; BIC = %f; DOF = 0"%(chi2_Teq,rchi2_Teq,chi2_Teq))
    print("Chi2 of Rayleigh at T_fit = %f; reduced chi2 = %f; BIC = %f; DOF = 1"%(chi2_Tfit,rchi2_Tfit,BIC_Tfit))
    print("Chi2 of flat line = %f; reduced chi2 = %f; BIC = %f; DOF = 0"%(chi2_flat,rchi2_flat,chi2_flat))

    if save_output:
        tab = open('trans_models_statistics.txt','w')
        tab.write("Chi2 of Rayleigh at T_eq = %f; reduced chi2 = %f; BIC = %f; DOF = 0 \n"%(chi2_Teq,rchi2_Teq,chi2_Teq))
        tab.write("Chi2 of Rayleigh at T_fit = %f; reduced chi2 = %f; BIC = %f; DOF = 1 \n"%(chi2_Tfit,rchi2_Tfit,BIC_Tfit))
        tab.write("Chi2 of flat line = %f; reduced chi2 = %f; BIC = %f; DOF = 0 \n"%(chi2_flat,rchi2_flat,chi2_flat))
        tab.close()

    return resampled_x,equilibrium_slope_resampled,fitted_slope_resampled,flat_line


def calc_sigma_confidence(chi2,DOF):
    """Calculating the sigma confidence at which a model is ruled out after first calculating the p-value.

    Inputs:
    chi2 - the chi2 of the fitted model to the data
    DOF - the degrees of freedom

    Returns:
    sigma_rejection - the sigma confidence at which the model is ruled out"""
    p_value = 1 - c2.cdf(chi2, DOF)
    sigma_rejection = erfinv(1-p_value)*np.sqrt(2)
    print("p_value = %f"%(p_value))
    print("sigma_rejection = %f sigma"%(sigma_rejection))
    return sigma_rejection



def plot_models(model_list,time,flux_array,error_array,wvl_centre,rebin_data=None,save_fig=False,gp=False):

    """Plot the models fitted to the wavelength binned light curves. This will work for fits both with TransitModel and TransitModelGP objects.

    Input:
    model_list - list of TransitModel/TransitModelGP objects
    time - array of times, either with shape (1,) or shape (nbins,ndatapoints)
    flux_array - array of fluxes with shape (nbins,ndatapoints)
    error_array - array of errors on fluxes with shape (nbins,ndatapoints)
    wvl_centre - array of wavelength bin centres, for annotation
    rebin_data - set to integer if wanting to re-bin the data. Default = None (no binning)
    save_fig - True/False: save the figure to file? Default=False
    gp - True/False: are we plotting GP objects or not? Default=False

    Returns:
    matplotlib figure object
    """

    # define nbins as the minimum length of the following: this is done if we are plotting the models while the MCMC still has to run on remaining bins
    nbins = min(len(model_list),len(flux_array),len(wvl_centre))

    # define the offsets in y for each light curve so they're not overlapping
    offsets = [0.015 * i for i in range(nbins)]

    # figure iut whether this is a white light curve
    try:
        tc = model_list[0].t0
    except:
        try:
            tc = model_list[0].pars['t0'].currVal
        except:
            tc = model_list[0].pars['t0']

    fig = plt.figure(figsize=(8,10))

    ax1 = plt.subplot(1,2,1)
    # ax1a = ax1.twinx()

    ax2 = plt.subplot(1,2,2)
    ax2a = ax2.twinx()

    rms = []


    for i in range(nbins):

        # figure out whether we're using a common time array
        if len(np.shape(time)) != 1 or isinstance(time,list) or time.shape[0] > 1:
            t = time[i]
        else:
            t = time

        # convert days to hours from mid-transit
        hours = mjd2hours(t,tc)

        # calculate transit model
        model_y = model_list[i].calc(t)

        if gp:
            mu,std = model_list[i].calc_gp_component(t,flux_array[i],error_array[i])
            residuals = flux_array[i] - model_y - mu
            RMS = model_list[i].rms(t,flux_array[i],error_array[i])
        else:
            residuals = flux_array[i]-model_y
            RMS = model_list[i].rms(t,flux_array[i])

        print("RMS/photon noise = %.2f"%(RMS/error_array[i].mean()))

        rms.append(RMS)

        if rebin_data is not None:
            xp,yp,ep = rebin(np.linspace(t[0],t[-1],rebin_data),t,flux_array[i],e=error_array[i])
            _,yr,_ = rebin(np.linspace(t[0],t[-1],rebin_data),t,residuals,e=error_array[i])
            hp = mjd2hours(xp,tc)
            if gp:
                _,mu,_ = rebin(np.linspace(t[0],t[-1],rebin_data),t,mu,e=error_array[i])
                _,std,_ = rebin(np.linspace(t[0],t[-1],rebin_data),t,std,e=error_array[i])
            _,model_y,_ = rebin(np.linspace(t[0],t[-1],rebin_data),t,model_y,e=error_array[i])



        else:
            hp,yp,ep,yr = hours,flux_array[i],error_array[i],residuals

        ax1.errorbar(hp,yp-offsets[i],ep,fmt='o',capsize=0,color='k',ecolor='k',ms=3,alpha=0.5,zorder=2)

        if gp:
            # ax1.plot(hours,model_y-offsets[i],color='0.75',ls='--') # Plot transit model alone
            ax1.plot(hp,mu+model_y-offsets[i],color='r',alpha=1,lw=1,zorder=100)
            ax1.plot(hp,mu+1-offsets[i],color='g',alpha=1,lw=1) # Plot GP model alone
        else:
            ax1.plot(hp,model_y-offsets[i],color='r',alpha=1,lw=2,zorder=100)

        ax2.errorbar(hp,yr-offsets[i],ep,fmt='o',capsize=0,color='k',ecolor='k',ms=3,alpha=0.5,zorder=2)
        ax2.axhline(-offsets[i],ls='--',color='r',lw=2)

    ax1.set_ylabel('Normalised flux + offset',fontsize=18)

    # ax1a.set_ylim(ax1.get_ylim())
    # ax1a.set_yticks(1-np.array(offsets))
    ax1.set_yticks(1-np.array(offsets))
    ax1.set_title("Differential flux")

    # set second y axis label to wavelength bin centres
    rounded_wavelength = (5 * np.round(wvl_centre/5))
    # wavelength_labels = np.array(['%s$\AA$'% i for i in rounded_wavelength.astype(int)])

    # ax1a.set_yticklabels(wavelength_labels)

    ax2.set_ylim(ax1.get_ylim()[0]-1,ax1.get_ylim()[1]-1)
    ax2.set_yticks(-np.array(offsets))
    ax2.set_yticklabels([])
    ax2.set_title('Residuals')#,fontsize=18)

    ax2a.set_ylim(ax2.get_ylim())
    ax2a.set_yticks(-np.array(offsets))

    # set second y axis label to RMS of residuals
    if determine_wvl_units(wvl_centre) == "$\AA$":
        rms_labels = np.array(["%s$\AA$, %d ppm" %(j,i*1e6) for (i,j) in zip(rms,rounded_wavelength.astype(int))])
    else:
        rms_labels = np.array(["%s%s, %d ppm" %(j,determine_wvl_units(wvl_centre),i*1e6) for (i,j) in zip(rms,wvl_centre)])

    ax2a.set_yticklabels(rms_labels)


    print("\nMedian RMS (ppm) = %d \n"%(np.median(rms)*1e6))
    fig.subplots_adjust(wspace=0.02)

    fig.text(0.5, 0.04, 'Time from mid-transit (hours)', ha='center',fontsize=18)
    # fig.tight_layout()

    if save_fig:
        if rebin_data is None:
            plt.savefig('fitted_model.png',bbox_inches='tight',dpi=360)
            # ~ plt.savefig('fitted_model.pdf',bbox_inches='tight')
        else:
            plt.savefig('fitted_model_rebin_%d.png'%rebin_data,bbox_inches='tight',dpi=360)

        plt.close()
        # plt.show()

    else:
        plt.show()

    return fig


def plot_single_model(model,time,flux,error,rebin_data=None,save_fig=False,wavelength_bin=None,deconstruct=True,plot_residual_std=0,systematics_model_inputs=None):
    """
    Plot a single light curve with model.

    Input:
    model - TransitModel/TransitModelGP object
    time - array of times
    flux - array of fluxes
    error - array of errors on fluxes
    rebin_data - set to integer if wanting to re-bin the data. Default = None (no binning)
    save_fig - True/False: save the figure to file? Default=False
    wavelength_bin - the number of the wavelength bin being plotted, useful for saving to file. Default=None
    deconstruct_gp - True/False: use if wanting to plot the contributions of each kernel to the overall GP fit. Default=True. Note: this used to be more informative when not using a single amplitude for the GP. This could do with improving.
    plot_residual_std - This can be used to overplot the standard deviation on the residuals subplot, which is useful when clipping data based on this. Set this parameter to the number of standard deviations desired. Default=0
    systematics_model_inputs - the model inputs to feed to the systematics model

    Returns:
    matplotlib figure object"""

    # figure out whether it's a white light curve
    try:
        tc = model.t0.currVal
    except:
        try:
            tc = model.pars['t0'].currVal
        except:
            tc = model.pars['t0']

    fig = plt.figure()

    if deconstruct:
        nsubplots = 3
    else:
        nsubplots = 2

    gp = model.GP_used
    poly = model.poly_used
    exp = model.exp_ramp_used
    step = model.step_func_used

    # convert times from days to hours from mid-transit
    hours = mjd2hours(time,tc)

    # calculate M&A transit model
    model_y = model.calc(time,systematics_model_inputs)
    oot = 1

    if poly:# and not gp:
        if deconstruct:
            oot,poly_components = model.red_noise_poly(time,systematics_model_inputs,deconstruct_polys=True)
        else:
            oot = model.red_noise_poly(time,systematics_model_inputs)

    if exp:
        exp_ramp = model.exponential_ramp(time)
        oot *= exp_ramp

    if step:
        step_func = model.step_function(time)
        oot *= step_func

    if gp:
        if deconstruct:
            mu,std,mu_components = model.calc_gp_component(time,flux,error,deconstruct_gp=True)
        else:
            mu,std = model.calc_gp_component(time,flux,error,deconstruct_gp=False)
        residuals = flux - model_y - mu
    else:
        residuals = flux - model_y

    if rebin_data is not None:
        xp,yp,ep = rebin(np.linspace(time[0],time[-1],rebin_data),time,flux,e=error,errors_from_rms=False)
        _,yr,_ = rebin(np.linspace(time[0],time[-1],rebin_data),time,residuals,e=error)
        hp = mjd2hours(xp,tc)

    else:
        hp,yp,ep,yr = hours,flux,error,residuals

    subplot = 1
    ax1 = plt.subplot(nsubplots,1,subplot)

    if rebin_data is not None:
        ax1.errorbar(hours,flux,error,fmt='.',capsize=0,color='gray',ecolor='gray',alpha=0.25,zorder=0)
        ax1.errorbar(hp,yp,ep,fmt='o',capsize=2,color='k',ecolor='k',mfc='white',ms=4,alpha=1,mew=2,lw=2)
    else:
        ax1.errorbar(hp,yp,ep,fmt='o',capsize=0,color='k',ecolor='k',ms=4,alpha=0.5)


    if gp:
        ax1.plot(hours,mu+model_y,color='r',zorder=10,label='GP & transit model')
        ax1.plot(hours,mu+1,color='g',label='GP')
        ax1.plot(hours,model_y,color='0.75',ls='--',zorder=9,label='Transit model')
        NCOL = 1

    if gp:
        ax1.legend(ncol=NCOL,fontsize=6)

    if poly and not gp or exp and not gp:
        ax1.plot(hours,model_y,color='r',zorder=10,label='Systematics & transit model',lw=1)
        ax1.plot(hours,oot,color='g',label='Systematics model',lw=1)
        ax1.plot(hours,model_y/oot,color='0.75',ls='--',zorder=9,label='Transit model',lw=1)
        NCOL = 1
        ax1.legend(ncol=NCOL,fontsize=6)

    ax1.tick_params(bottom=True,top=True,left=True,right=True,direction="inout")
    ax1.tick_params(which='minor',bottom=True,top=True,left=True,right=True,direction="inout")#,labelsize=fontsize-4,length=4,width=1.)
    plt.xticks(visible=False)
    ax1.set_ylabel('Normalised flux')

    subplot += 1

    if deconstruct: # plotting contributions of each GP kernel, now on a separate subplot

        model_ax = plt.subplot(nsubplots,1,subplot)

        if gp:
            if len(mu_components) > 1:
                alpha = 0.5
            else:
                alpha = 1
            for i,m in enumerate(mu_components):
                model_ax.plot(hours,(m*1e6)-(m*1e6).mean(),label='kernel %d'%(i+1),alpha=alpha,lw=1)

        if poly:
            if len(poly_components) > 1:
                alpha = 0.5
            else:
                alpha = 1
            for i,m in enumerate(poly_components):
                model_ax.plot(hours,(m*1e6)-(m*1e6).mean(),label='poly %d'%(i+1),alpha=alpha,lw=1)

        if exp:
            model_ax.plot(hours,(exp_ramp*1e6)-(exp_ramp*1e6).mean(),label='exponential ramp',alpha=1,lw=1)

        if step:
            model_ax.plot(hours,(step_func*1e6)-(step_func*1e6).mean(),label='step function',alpha=1,lw=1)

        NCOL = 2
        subplot += 1
        model_ax.ticklabel_format(useOffset=False)
        plt.xticks(visible=False)
        model_ax.legend(loc = 'lower right',ncol=NCOL,fontsize=6)
        model_ax.set_ylabel('RN components\n(ppm)')


    ax2 = plt.subplot(nsubplots,1,subplot)

    if rebin_data is not None:
        ax2.errorbar(hours,1e6*residuals,1e6*error,fmt='.',capsize=0,color='gray',ecolor='gray',alpha=0.25,zorder=0)
        ax2.errorbar(hp,yr*1e6,ep*1e6,fmt='o',capsize=2,color='k',ecolor='k',mfc='white',ms=4,alpha=1,lw=2,mew=2)
    else:
        ax2.errorbar(hp,yr*1e6,ep*1e6,fmt='o',capsize=0,color='k',ecolor='k',ms=4,alpha=0.5)

    ax2.axhline(0,ls='--',color='k')

    if plot_residual_std > 0:
        print("plotting outliers")
        rms = np.sqrt(np.mean(yr**2))*1e6

        ax2.axhline(plot_residual_std*rms,ls='--',color='r')
        ax2.axhline(-plot_residual_std*rms,ls='--',color='r')

        if gp:
            wn_var = np.exp(model.starting_gp_object.white_noise.get_value(time))
            wn_std = np.sqrt(wn_var)
            plt.fill_between(hours,-plot_residual_std*(std+wn_std),+plot_residual_std*(std+wn_std),
                    color="k", alpha=0.2)

    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time from mid-transit (hours)')
    ax2.tick_params(bottom=True,top=True,left=True,right=True,direction="inout")
    ax2.tick_params(which='minor',bottom=True,top=True,left=True,right=True,direction="inout")#,labelsize=fontsize-4,length=4,width=1.)


    if save_fig:
        if wavelength_bin is not None:
            wb = '_wb%s'%(str(wavelength_bin+1).zfill(4))
        else:
            wb = ''

        if rebin_data is None:
            # ~ plt.savefig('fitted_model%s.pdf'%wb,bbox_inches='tight')
            plt.savefig('fitted_model%s.png'%wb,bbox_inches='tight',dpi=200)
        else:
            plt.savefig('fitted_model%s_rebin_%d.png'%(wb,rebin_data),bbox_inches='tight',dpi=200)

        plt.close()

    else:
        # plt.show()
        plt.show(block=False) # only show for 5 seconds. This is necessary when running fits to multiple bins so that the code doesn't have to wait for user to manually close windows before continuing.
        plt.pause(5)
        plt.close()

    return fig


def rebin(xbins,x,y,e=None,weighted=False,errors_from_rms=False):

    """
    Rebin time, flux and error arrays into a given number of bins. These can either be weighted or unweighted means.

    Input:
    xbins - the bins, in time, over which to bin the data arrays.
    x - the time array
    y - the flux array
    e - the error array
    weighted - True/False: if True, compute weighted mean, otherwise unweighted. Default = False.
    errors_from_rms - True/False: if True, the errors on the binned fluxes will be equal to the standard deviation of the fluxes in the bin. If False, add the errors on the data points in quadrature. Default = False.

    Returns:
    (array of binned times, array of binned fluxes, array of binned errors)

    """
    digitized = np.digitize(x,xbins)
    xbin = []
    ybin = []
    ebin = []
    for i in range(1,len(xbins)):
        bin_y_vals = y[digitized == i]
        bin_x_vals = x[digitized == i]

        if weighted:
            if e is None:
                raise Exception('Cannot compute weighted mean without Falseerrors')
            bin_e_vals = e[digitized == i]
            weights = 1.0/bin_e_vals**2
            xbin.append( np.sum(weights*bin_x_vals) / np.sum(weights) )
            ybin.append( np.sum(weights*bin_y_vals) / np.sum(weights) )
            if errors_from_rms:
                ebin.append(np.std(bin_y_vals))
            else:
                ebin.append( np.sqrt(1.0/np.sum(weights) ) )
        else:
            xbin.append(bin_x_vals.mean())
            ybin.append(bin_y_vals.mean())
            #xbin.append(stats.nanmean(bin_x_vals))
            #ybin.append(stats.nanmean(bin_y_vals))
            if errors_from_rms:
                ebin.append(np.std(bin_y_vals))
            else:
                try:
                    bin_e_vals = e[digitized == i]
                    ebin.append(np.sqrt(np.sum(bin_e_vals**2)) / len(bin_e_vals))
                except:
                    raise Exception('Must either supply errors, or calculate from rms')
    xbin = np.array(xbin)
    ybin = np.array(ybin)
    ebin = np.array(ebin)
    return (xbin,ybin,ebin)

def recover_transmission_spectrum(directory,save_fig=False,plot_fig=True,bin_mask=None,save_to_tab=False,print_RpErr_over_RMS=False,iib=False,plot_depths=False):
    """
    A function that generates/recovers the transmission spectrum from the table of best fit parameters resulting from pm_fit.py and gp_fit.py.

    Input:
    directory - the directory containing the best_fit_parameters.dat, fitting_input.txt, prod_model*.pickle and LD_coefficients.dat files
    save_fig - True/False: save the outputted transmission spectrum or not? Default=False
    plot_fig - True/False: plot the outputted transmission spectrum or not? If False, code returns numpy arrays of Rp/Rs and errors. Default=True
    bin_mask - set to a list of integers to mask certain wavelength bins from the transmission spectrum if desired. Indexed from 0. Default = None (no masking).
    save_to_tab - True/False: if True, saves transmission spectrum to .dat text file. Default=False
    print_RpErr_over_RMS - True/False - use this to increase verbosity and print how the errors in Rp/Rs compare to the RMS of the residuals and the photon noise. Default=False
    iib - True/False: - If this is an iib fit to Na or K then plot the transmission spectrum with wvl_error on x-axis.
    plot_depths - True/False: - Use this to plot in transit depth rather than Rp/Rs. Default=False (Rp/Rs).

    Returns:
    matplotlib figure object if plot_fig = True
    OR
    (Rp/Rs,Rp/Rs +ve error, -ve error) if plot_fig = False
    """

    try:
        best_dict = parseInput(directory+'/best_fit_parameters_GP.txt')
    except:
        best_dict = parseInput(directory+'/best_fit_parameters.txt')

    input_dict = parseInput(directory+'/fitting_input.txt')

    # load in data
    x,y,e,e_r,m,m_in,w,we,completed_bins,nbins = load_completed_bins(directory,bin_mask)

    k = []
    k_up = []
    k_low = []
    d = []
    d_up = []
    d_low = []

    if not bool(int(input_dict['fix_u1'])):
        u1 = []
        u1_up = []
        u1_low = []

    if not bool(int(input_dict['fix_u2'])):
        u2 = []
        u2_up = []
        u2_low = []

    if input_dict["ld_law"] == "nonlinear":
        if not bool(int(input_dict['fix_u3'])):
            u3 = []
            u3_up = []
            u3_low = []

        if not bool(int(input_dict['fix_u4'])):
            u4 = []
            u4_up = []
            u4_low = []

    # calculate the planet's atmospheric scale height which is useful for plotting
    try:
        H = (c.k_B.value*float(input_dict['Teq']))/(2.3*c.m_p.value*10**float(input_dict['logg'])/100.)
    except:
        H = (c.k_B.value*float(input_dict['Teq']))/(2.3*c.m_p.value*float(input_dict['g']))
    Rs = float(input_dict['rs'])*c.R_sun.value

    if plot_depths:
        H_Rs = (2*float(input_dict['rp'])*c.R_jup.value*H)/Rs**2
    else:
        H_Rs = H/Rs

    for i,wb in enumerate(completed_bins):

        k_curr,k_up_curr,k_low_curr = mc.parseParam(best_dict['k_%d'%wb])
        k.append(k_curr)
        k_up.append(k_up_curr)
        k_low.append(k_low_curr)

        transit_depth = k_curr**2
        transit_depth_err_up = transit_depth*2*k_up_curr/k_curr
        transit_depth_err_low = transit_depth*2*k_low_curr/k_curr
        transit_depth_err = np.mean((transit_depth_err_up,transit_depth_err_low))

        d.append(transit_depth)
        d_up.append(transit_depth_err_up)
        d_low.append(transit_depth_err_low)


        if print_RpErr_over_RMS:


            GP_model = m[i].GP_used
            if GP_model:
                rms = m[i].rms(x[i],y[i],e[i])
            else:
                rms = m[i].rms(x[i],y[i])

            print('Wavelength bin %d: sigma(Rp/Rs)/H = %.2f'%(i+1,np.maximum(k_up_curr,k_low_curr)/H_Rs))

            if e_r is not None:
                # ~ e_r = pickle.load(open(error_list_rescaled[counter],'rb'))
                if GP_model:
                    rms_r = m[i].rms(x[i],y[i],e_r[i])
                else:
                    rms_r = m[i].rms(x[i],y[i])

        if not bool(int(input_dict['fix_u1'])):
            u1_curr,u1_up_curr,u1_low_curr = mc.parseParam(best_dict['u1_%d'%(wb)])
            u1.append(u1_curr)
            u1_up.append(u1_up_curr)
            u1_low.append(u1_low_curr)

        if not bool(int(input_dict['fix_u2'])):
            u2_curr,u2_up_curr,u2_low_curr = mc.parseParam(best_dict['u2_%d'%(wb)])
            u2.append(u2_curr)
            u2_up.append(u2_up_curr)
            u2_low.append(u2_low_curr)

        if input_dict["ld_law"] == "nonlinear":
            if not bool(int(input_dict['fix_u3'])):
                u3_curr,u3_up_curr,u3_low_curr = mc.parseParam(best_dict['u3_%d'%(wb)])
                u3.append(u3_curr)
                u3_up.append(u3_up_curr)
                u3_low.append(u3_low_curr)

            if not bool(int(input_dict['fix_u4'])):
                u4_curr,u4_up_curr,u4_low_curr = mc.parseParam(best_dict['u4_%d'%(wb)])
                u4.append(u4_curr)
                u4_up.append(u4_up_curr)
                u4_low.append(u4_low_curr)

    k,k_up,k_low,d,d_up,d_low = np.array(k),np.array(k_up),np.array(k_low),np.array(d),np.array(d_up),np.array(d_low)
    if not bool(int(input_dict['fix_u1'])):
        u1,u1_up,u1_low = np.array(u1),np.array(u1_up),np.array(u1_low)

    if not bool(int(input_dict['fix_u2'])):
        u2,u2_up,u2_low = np.array(u2),np.array(u2_up),np.array(u2_low)

    if input_dict["ld_law"] == "nonlinear":
        if not bool(int(input_dict['fix_u3'])):
            u3,u3_up,u3_low = np.array(u3),np.array(u3_up),np.array(u3_low)
        if not bool(int(input_dict['fix_u4'])):
            u4,u4_up,u4_low = np.array(u4),np.array(u4_up),np.array(u4_low)

    print("\nMedian Rp/Rs = %.6f ;  Median Rp/Rs +ve error (ppm) = %d ; Median Rp/Rs -ve error (ppm) = %d \n"%(np.nanmedian(k),np.nanmedian(k_up)*1e6,np.nanmedian(k_low)*1e6))

    if bin_mask is not None:
        k = k[bin_mask]
        k_up = k_up[bin_mask]
        k_low = k_low[bin_mask]
        d = d[bin_mask]
        d_up = d_up[bin_mask]
        d_low = d_low[bin_mask]
        w = w[bin_mask]
        we = we[bin_mask]
        nbins = len(k)

    if save_to_tab:

        new_tab = open('transmission_spectrum.txt','w')
        new_tab.write('# Wavelength bin centre (%s), wavelength bin full width (%s), Rp/Rs, Rp/Rs +ve error, Rp/Rs -ve error'%(determine_wvl_units(w),determine_wvl_units(w)))

        new_tab_2 = open('transmission_spectrum_depths.txt','w')
        if np.all(d_up == d_low):
            new_tab_2.write('# Wavelength bin centre (%s), wavelength bin full width (%s), Transit depth, Transit depth error'%(determine_wvl_units(w),determine_wvl_units(w)))
        else:
            new_tab_2.write('# Wavelength bin centre (%s), wavelength bin full width (%s), Transit depth, Transit depth +ve error, Transit depth -ve error'%(determine_wvl_units(w),determine_wvl_units(w)))

        if not bool(int(input_dict['fix_u1'])):
            new_tab.write(', u1, u1 +ve error, u1 -ve error')
            # ~ new_tab_2.write(', u1, u1 +ve error, u1 -ve error')
        if not bool(int(input_dict['fix_u2'])):
            new_tab.write(', u2, u2 +ve error, u2 -ve error')
            # ~ new_tab_2.write(', u2, u2 +ve error, u2 -ve error')

        if input_dict["ld_law"] == "nonlinear":
            if not bool(int(input_dict['fix_u3'])):
                new_tab.write(', u3, u3 +ve error, u3 -ve error')
                # ~ new_tab_2.write(', u3, u3 +ve error, u3 -ve error')
            if not bool(int(input_dict['fix_u4'])):
                new_tab.write(', u4, u4 +ve error, u4 -ve error')
                # ~ new_tab_2.write(', u4, u4 +ve error, u4 -ve error')

        new_tab.write('\n')
        new_tab_2.write('\n')

        for i in range(nbins):
            new_tab.write('%f %f %.6f %.6f %.6f'%(w[i],we[i],k[i],k_up[i],k_low[i]))

            # only save one depth error column if they're equal. But don't do this for Rp/Rs since compare_transmission_spectra.py relies on 5 column input
            if np.all(d_up == d_low):
                new_tab_2.write('%f %f %.6f %.6f'%(w[i],we[i],d[i],d_up[i]))
            else:
                new_tab_2.write('%f %f %.6f %.6f %.6f'%(w[i],we[i],d[i],d_up[i],d_low[i]))

            if not bool(int(input_dict['fix_u1'])):
                new_tab.write( ' %.2f %.2f %.2f'%(u1[i],u1_up[i],u1_low[i]))
                # ~ new_tab_2.write( ' %.2f %.2f %.2f'%(u1[i],u1_up[i],u1_low[i]))
            if not bool(int(input_dict['fix_u2'])):
                new_tab.write( ' %.1f %.1f %.1f'%(u2[i],u2_up[i],u2_low[i]))
                # ~ new_tab_2.write( ' %.1f %.1f %.1f'%(u2[i],u2_up[i],u2_low[i]))
            if input_dict["ld_law"] == "nonlinear":
                if not bool(int(input_dict['fix_u3'])):
                    new_tab.write( ' %.1f %.1f %.1f'%(u3[i],u3_up[i],u3_low[i]))
                    # ~ new_tab_2.write( ' %.1f %.1f %.1f'%(u3[i],u3_up[i],u3_low[i]))
                if not bool(int(input_dict['fix_u4'])):
                    new_tab.write( ' %.1f %.1f %.1f'%(u4[i],u4_up[i],u4_low[i]))
                    # ~ new_tab_2.write( ' %.1f %.1f %.1f'%(u4[i],u4_up[i],u4_low[i]))

            new_tab.write('\n')
            new_tab_2.write('\n')

        new_tab.close()
        new_tab_2.close()

    if plot_fig:
        if iib:
            if plot_depths:
                fig = plot_transmission_spectrum(d,d_up,d_low,calibrated_wvl=we,wvl_errors=None,save_fig=save_fig,scale_height=H_Rs**2,iib=True,plot_depths=True)
            else:
                fig = plot_transmission_spectrum(k,k_up,k_low,calibrated_wvl=we,wvl_errors=None,save_fig=save_fig,scale_height=H_Rs,iib=True)
        else:
            if plot_depths:
                fig = plot_transmission_spectrum(d,d_up,d_low,calibrated_wvl=w,wvl_errors=we/2,save_fig=save_fig,scale_height=H_Rs,plot_depths=True)
            else:
                fig = plot_transmission_spectrum(k,k_up,k_low,calibrated_wvl=w,wvl_errors=we/2,save_fig=save_fig,scale_height=H_Rs)
        return fig
    else:
        if plot_depths:
            return d,d_up,d_low,w,we,H_Rs
        else:
            return np.array(k),np.array(k_up),np.array(k_low),w,we,H_Rs

def plot_multi_trans_spec(directory_lists,save_fig=False,plot_fig=False):
    """
    A function to plot transmission spectra from multiple nights on a single figure.

    Input:
    directory_lists - a list of the locations of the directories containing the transmission spectra to be plotted
    save_fig - True/False - save the figure or not. Default=False
    plot_fig - True/False - either plot the figure (True) or return the concatenated transmission spectra (False). Default=False

    Returns:
    matplotlib figure object - if plot_fig = True
    np.array(k_all),np.array(k_up_all),np.array(k_low_all),np.array(w_all),np.array(we_all),H_Rs - the concatenated Rp/Rs, errors, wavelength bin centres and errors, and atmospheric scale height - if plot_fig = False

    Returns:"""

    k_all = np.array([])
    k_up_all = np.array([])
    k_low_all = np.array([])
    w_all = np.array([])
    we_all = np.array([])

    for d in directory_lists:
        try:
            k,k_up,k_low,w,we,H_Rs = recover_transmission_spectrum(d+'best_fit_parameters_GP.txt',d+'fitting_input.txt',False,False)
        except:
            k,k_up,k_low,w,we,H_Rs = recover_transmission_spectrum(d+'best_fit_parameters_noGP.txt',d+'fitting_input.txt',False,False)
        k_all = np.hstack((k_all,k))
        k_up_all = np.hstack((k_up_all,k_up))
        k_low_all = np.hstack((k_low_all,k_low))
        w_all = np.hstack((w_all,w))
        we_all = np.hstack((we_all,we))

    if plot_fig:
        fig = plot_transmission_spectrum(np.array(k_all),np.array(k_up_all),np.array(k_low_all),calibrated_wvl=np.array(w_all),wvl_errors=np.array(we_all),save_fig=save_fig,scale_height=H_Rs)
        return fig
    else:
        return np.array(k_all),np.array(k_up_all),np.array(k_low_all),np.array(w_all),np.array(we_all),H_Rs


def plot_transmission_spectrum(k_array,k_upper=None,k_lower=None,calibrated_wvl=None,wvl_errors=None,bin_width=250,save_fig=False,scale_height=None,model_atmos=None,iib=False,plot_depths=False):

    """
    Function that plots the transmission spectrum (Rp/Rs vs wavelength in Angstroms).

    Inputs:
    k_array - the array of Rp/Rs values
    k_upper - the positive errors in Rp/Rs. Default=None
    k_lower - the negative errors in Rp/Rs. Default=None
    calibrated_wvl - the wavelength bin centres. If this is not supplied, the figure will estimate the wavelength bins as an array from 3000-9000A with a length=len(k_array). Default=None
    wvl_errors - the wavelength bin half widths, in Angstroms
    bin_width - if calibrated_wvl = None, this defines the xerror. Default=250.
    save_fig - True/False - save the outputted figure or not. Default=False
    scale_height - if wanting the plot the Rp/Rs values in terms of the scale heights on the right-hand y axis, supply this value here as the atmospheric scale height divided by the radius of the star. Default=None (no plotting of this)
    model_atmos - if wanting to overplot model atmospheres on the transmission spectrum. This should be given as a dictionary as {'binned_wvl':,'binned_k':,'unbinned_wvl':,'unbinned_k'}. Default=None (no models plotted)

    Returns:
    matplotlib figure object
    """

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    #if calibrated_wvl == None:
    if calibrated_wvl is None:
          wvl = np.linspace(3000,9000,len(k_array))
          xerror = bin_width
    else:
          wvl = calibrated_wvl
          xerror = wvl_errors

    if k_upper is not None:
        e = (k_lower,k_upper)

    else:
        e = None

    if iib:
        ax.errorbar(wvl,k_array,yerr=e,xerr=xerror,fmt='o',ecolor='k',zorder=10,color='k')
    else:
        ax.errorbar(wvl,k_array,yerr=e,xerr=xerror,fmt='.',ecolor='k',zorder=10,capsize=2,mfc='white',mec='k')

    if model_atmos is not None:
        try:
            ax.plot(model_atmos['unbinned_wvl'],model_atmos['unbinned_k'],'b')
        except:
            pass
        ax.plot(model_atmos['binned_wvl'],model_atmos['binned_k'],'ro')

    if plot_depths:
        ax.set_ylabel('Transit depth $(R_{P}/R_{S})^2$',fontsize=12)
    else:
        ax.set_ylabel('$R_{P}/R_{S}$',fontsize=12)

    if iib:
        ax.set_xlim(wvl[0]-4,wvl[-1]+4)
        ax.set_xlabel('Bin width ($\AA$)',fontsize=12)
    else:
        ax.set_xlabel('Wavelength (%s)'%determine_wvl_units(wvl),fontsize=12)

    x_minor_locator = AutoMinorLocator(10)
    y_minor_locator = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)

    ax.tick_params(bottom=True,top=True,left=True,right=True,direction="inout",labelsize=10,length=6,width=1.,pad=3)
    ax.tick_params(which='minor',bottom=True,top=True,left=True,right=True,direction="inout",labelsize=8,\
                       length=4,width=1.)


    if scale_height is not None:

        H_rs = scale_height

        yticks = ax.get_yticks()
        ax2 = ax.twinx()

        scale_height_tick_range = (yticks[-1]-yticks[0])/H_rs
        ax2.set_ylim(-scale_height_tick_range/2.,scale_height_tick_range/2.)
        ax2.set_ylabel('Atmospheric Scale Heights (H)',fontsize=12)

        ax2.yaxis.set_minor_locator(y_minor_locator)
        ax2.tick_params(right=True,direction="inout",labelsize=10,length=6,width=1.,pad=3)
        ax2.tick_params(which='minor',right=True,direction="inout",labelsize=8,\
                       length=4,width=1.)

    if save_fig:
        plt.savefig('transmission_spectrum.pdf',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig

def expected_vs_calculated_ldcs(directory='.',save_fig=False,bin_mask=None):

    """
    Function to plot the expected (LDTk-generated) quadratic limb darkening coefficients vs. the actual fitted limb darkening coefficients.

    This function loads the limb darkening coefficients from files within the cwd, so these do not need to be supplied to the function.

    Input:
    save_fig - True/False - save the resulting figure? Default=False
    bin_mask - a list of wavelength bins to be ignored by the plot. Default=None (no masking of bins)

    Returns:
    Nothing, it just plots the figure"""

    wvl_centre,wvl_error,ldtk_u1,ldtk_u1_err,ldtk_u2,ldtk_u2_err,ldtk_u3,ldtk_u3_err,ldtk_u4,ldtk_u4_err = np.loadtxt('%s/LD_coefficients.txt'%directory,unpack=True)
    wvl_error = wvl_error/2

    try:
        best_dict = parseInput('%s/best_fit_parameters.txt'%directory)
    except:
        best_dict = parseInput('%sbest_fit_parameters_GP.txt'%directory)

    model_list = glob.glob('%s/prod_model_*.pickle'%directory)
    nbins = len(model_list)

    completed_bins = load_completed_bins(directory,return_index_only=True,mask=bin_mask)

    wvl_centre,wvl_error,ldtk_u1,ldtk_u1_err,ldtk_u2,ldtk_u2_err,ldtk_u3,ldtk_u3_err,ldtk_u4,ldtk_u4_err = np.atleast_1d(wvl_centre)[completed_bins],np.atleast_1d(wvl_error)[completed_bins],\
    np.atleast_1d(ldtk_u1)[completed_bins],np.atleast_1d(ldtk_u1_err)[completed_bins],np.atleast_1d(ldtk_u2)[completed_bins],np.atleast_1d(ldtk_u2_err)[completed_bins],\
    np.atleast_1d(ldtk_u3)[completed_bins],np.atleast_1d(ldtk_u3_err)[completed_bins],np.atleast_1d(ldtk_u4)[completed_bins],np.atleast_1d(ldtk_u4_err)[completed_bins]

    m = pickle.load(open(model_list[0],'rb'))
    fix_u1 = m.fix_u1
    fix_u2 = m.fix_u2
    if m.ld_law == "nonlinear":
        fix_u3 = m.fix_u3
        fix_u4 = m.fix_u4
    else:
        fix_u3 = True # note these are not used in this case
        fix_u4 = True

    u1,u1_up,u1_low = [],[],[]
    u2,u2_up,u2_low = [],[],[]
    u3,u3_up,u3_low = [],[],[]
    u4,u4_up,u4_low = [],[],[]

    for i in range(nbins):
        if not fix_u1:
            try:
                u1_curr,u1_up_curr,u1_low_curr = mc.parseParam(best_dict['u1_%d'%(i+1)])
                u1.append(u1_curr)
                u1_up.append(u1_up_curr)
                u1_low.append(u1_low_curr)
            except:
                u1.append(np.nan)
                u1_up.append(np.nan)
                u1_low.append(np.nan)

        if not fix_u2:
            try:
                u2_curr,u2_up_curr,u2_low_curr = mc.parseParam(best_dict['u2_%d'%(i+1)])
                u2.append(u2_curr)
                u2_up.append(u2_up_curr)
                u2_low.append(u2_low_curr)
            except:
                u2.append(np.nan)
                u2_up.append(np.nan)
                u2_low.append(np.nan)

        if not fix_u3:
            try:
                u3_curr,u3_up_curr,u3_low_curr = mc.parseParam(best_dict['u3_%d'%(i+1)])
                u3.append(u3_curr)
                u3_up.append(u3_up_curr)
                u3_low.append(u3_low_curr)
            except:
                u3.append(np.nan)
                u3_up.append(np.nan)
                u3_low.append(np.nan)

        if not fix_u4:
            try:
                u4_curr,u4_up_curr,u4_low_curr = mc.parseParam(best_dict['u4_%d'%(i+1)])
                u4.append(u4_curr)
                u4_up.append(u4_up_curr)
                u4_low.append(u4_low_curr)
            except:
                u4.append(np.nan)
                u4_up.append(np.nan)
                u4_low.append(np.nan)

    u1,u1_up,u1_low,u2,u2_up,u2_low,u3,u3_up,u3_low,u4,u4_up,u4_low = np.array(u1),np.array(u1_up),np.array(u1_low),np.array(u2),np.array(u2_up),np.array(u2_low),\
    np.array(u3),np.array(u3_up),np.array(u3_low),np.array(u4),np.array(u4_up),np.array(u4_low)

    if bin_mask is not None:

        wvl_centre = wvl_centre[bin_mask]
        wvl_error = wvl_error[bin_mask]
        ldtk_u1 = ldtk_u1[bin_mask]
        ldtk_u1_err = ldtk_u1_err[bin_mask]
        ldtk_u2 = ldtk_u2[bin_mask]
        ldtk_u2_err = ldtk_u2_err[bin_mask]
        ldtk_u3 = ldtk_u3[bin_mask]
        ldtk_u3_err = ldtk_u3_err[bin_mask]
        ldtk_u4 = ldtk_u4[bin_mask]
        ldtk_u4_err = ldtk_u4_err[bin_mask]

        if not fix_u1:
            u1 = u1[bin_mask]
            u1_up = u1_up[bin_mask]
            u1_low = u1_low[bin_mask]

        if not fix_u2:
            u2 = u2[bin_mask]
            u2_up = u2_up[bin_mask]
            u2_low = u2_low[bin_mask]

        if not fix_u3:
            u3 = u3[bin_mask]
            u3_up = u3_up[bin_mask]
            u3_low = u3_low[bin_mask]

        if not fix_u4:
            u4 = u4[bin_mask]
            u4_up = u4_up[bin_mask]
            u4_low = u4_low[bin_mask]

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(wvl_centre,ldtk_u1,color='C0',label='Calculated ($u1$)')
    ax.fill_between(wvl_centre,ldtk_u1,ldtk_u1+ldtk_u1_err,color='lightgrey')
    ax.fill_between(wvl_centre,ldtk_u1,ldtk_u1-ldtk_u1_err,color='lightgrey')

    if not fix_u1:
        ax.errorbar(wvl_centre,u1,xerr=wvl_error,yerr=(u1_low,u1_up),ecolor='C0',fmt='.',color='C0',mec='k',capsize=2,label="Fitted u1",mfc='white')

    ax.plot(wvl_centre,ldtk_u2,color='C1',label='Calculated ($u2$)')
    ax.fill_between(wvl_centre,ldtk_u2,ldtk_u2+ldtk_u2_err,color='lightgrey')
    ax.fill_between(wvl_centre,ldtk_u2,ldtk_u2-ldtk_u2_err,color='lightgrey')

    if not fix_u2:
        ax.errorbar(wvl_centre,u2,xerr=wvl_error,yerr=(u2_low,u2_up),ecolor='C1',fmt='.',color='C1',mec='k',capsize=2,label="Fitted u2",mfc='white')

    if not fix_u3 and np.any(np.isfinite(u3)):
        ax.plot(wvl_centre,ldtk_u3,color='C2',label='Calculated ($u3$)')
        ax.fill_between(wvl_centre,ldtk_u3,ldtk_u3+ldtk_u3_err,color='lightgrey')
        ax.fill_between(wvl_centre,ldtk_u3,ldtk_u3-ldtk_u3_err,color='lightgrey')
        ax.errorbar(wvl_centre,u3,xerr=wvl_error,yerr=(u3_low,u3_up),ecolor='C2',fmt='.',color='C2',mec='k',capsize=2,label="Fitted u3",mfc='white')

    if not fix_u4 and np.any(np.isfinite(u4)):
        ax.plot(wvl_centre,ldtk_u4,color='C3',label='Calculated ($u3$)')
        ax.fill_between(wvl_centre,ldtk_u4,ldtk_u4+ldtk_u4_err,color='lightgrey')
        ax.fill_between(wvl_centre,ldtk_u4,ldtk_u4-ldtk_u4_err,color='lightgrey')
        ax.errorbar(wvl_centre,u4,xerr=wvl_error,yerr=(u4_low,u4_up),ecolor='C2',fmt='.',color='C3',mec='k',capsize=2,label="Fitted u4",mfc='white')


    ax.set_xlabel('Wavelength (%s)'%determine_wvl_units(wvl_centre),fontsize=14)
    ax.set_ylabel('Coefficient value',fontsize=14)
    ax.legend()
    x_minor_locator = AutoMinorLocator(10)
    y_minor_locator = AutoMinorLocator(4)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)

    ax.tick_params(bottom=True,top=True,left=True,right=True,direction="inout",labelsize=10,length=6,width=1.,pad=3)
    ax.tick_params(which='minor',bottom=True,top=True,left=True,right=True,direction="inout",labelsize=8,\
                       length=4,width=1.)


    if save_fig:
        plt.savefig('%s/expected_vs_calculated_ldcs.pdf'%directory,bbox_inches='tight')
        plt.savefig('%s/expected_vs_calculated_ldcs.png'%directory,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return

def determine_wvl_units(wvl_array):
    """A function which guesses the units of a wavelength array

    Input: wvl_array - the array of wavelengths
    Returns: the string of units: microns/nm/A in pythonic latex"""

    logged = np.log10(wvl_array)

    if logged.mean() >= 3:
        units = "$\AA$"
    if 2 <= logged.mean() <= 3:
        units = "nm"
    if logged.mean() <= 2:
        units = "$\mu$m"

    return units

def load_exotransmit_model(path):
    """
    Function to load in model atmospheres as produced by Exo-Transmit.

    Input:
    path - the path to the model to be loaded

    Returns:
    exo_wvl - the wavelength of the model (in Angstroms)
    exo_flux - the modelled Rp/Rs values
    """
    exo_wvl,eos_mod = np.loadtxt(path,skiprows=2,unpack=True)

    # Convert from metres to Angstroms
    exo_wvl = exo_wvl*1e10

    # Convert from depth to Rp/R*
    eos_mod = np.sqrt(eos_mod/100.)

    return exo_wvl,eos_mod

def bin_model_to_data(model_wvl,model_data,data_wvl,data_wvl_e,bin_break=None):
    """
    A function to bin model atmospheres (such as produced by Exo-Transmit) to the resolution of the transmission spectrum.

    Input:
    model_wvl - the model's wavelength array
    model_data - the model's Rp/Rs array
    data_wvl - the data's wavlength array (wavelength bin centres)
    data_wvl_e - the wavelength bin's half width
    bin_break - use this if wanting to ignore a certain bin from the resulting transmission spectrum (e.g. the telluric O2 A-band). Use with care. Default=None (no bins are ignored)

    Returns:
    binned_model_wvl - the model's wavlength array binned to the resolution of the transmission spectrum
    binned_model_data - the model's Rp/Rs array binned to the resolution of the transmission spectrum
    """

    n_data_bins = len(data_wvl)

    binned_model_data = []
    binned_model_wvl = []

    if bin_break is not None:
        bins1 = np.hstack((data_wvl[:bin_break]-data_wvl_e[:bin_break],data_wvl[bin_break-1]+data_wvl_e[bin_break-1]))
        bins2 = np.hstack((data_wvl[bin_break:]-data_wvl_e[bin_break:],data_wvl[-1]+data_wvl_e[-1]))

        digitized1 = np.digitize(model_wvl,bins1)
        digitized2 = np.digitize(model_wvl,bins2)

        for i in range(1,len(bins1)):
            binned_model_data.append(model_data[digitized1 == i].mean())
            binned_model_wvl.append(model_wvl[digitized1 == i].mean())

        for i in range(1,len(bins2)):
            binned_model_data.append(model_data[digitized2 == i].mean())
            binned_model_wvl.append(model_wvl[digitized2 == i].mean())

    else:
        bins = np.hstack((data_wvl-data_wvl_e/2,data_wvl[-1]+data_wvl_e[-1]/2))
        digitized = np.digitize(model_wvl,bins)

        for i in range(1,len(bins)):
            binned_model_data.append(model_data[digitized == i].mean())
            binned_model_wvl.append(model_wvl[digitized == i].mean())

    return np.array(binned_model_wvl),np.array(binned_model_data)



def weighted_mean_uneven_errors(k,k_up,k_low,model=1):
    """A function to calculate the weighted mean of multiple, concatenated, transmission spectra that have un-even (non-symmetric) uncertainties.

    This uses the models of Barlow 2003.

    Inputs:
    k - the concatenated Rp/Rs values
    k_up - the concatenated positive uncertainties in Rp/Rs
    k_low - the concatenated negative uncertainties in Rp/Rs
    model - the number of the model as given in Barlow 2003 (either 1 or 2)

    Returns:
    weighted mean Rp/Rs
    the uncertainties in the weighted mean Rp/Rs values"""

    nvalues = len(k)

    sigma = {}
    alpha = {}
    V = {}
    b = {}
    w = {}

    x_numerator = 0
    x_denominator = 0

    e_numerator = 0
    e_denominator = 0

    for i in range(nvalues):

        sigma[i+1] = (k_up[i]+k_low[i])/2. # eqn 1
        alpha[i+1] = (k_up[i]-k_low[i])/2. # eqn 1


        if model == 1:
            V[i+1] = sigma[i+1]**2 + (1 - 2/np.pi)*alpha[i+1]**2 # eqn 18
            b[i+1] = (k_up[i]-k_low[i])/np.sqrt(2*np.pi) # eqn 17

        if model == 2:
            V[i+1] = sigma[i+1]**2 + 2*alpha[i+1]**2 # eqn 18
            b[i+1] = alpha[i+1] # eqn 17

        w[i+1] = 1/V[i+1]

        x_numerator += (w[i+1]*(k[i]-b[i+1])) # eqn 16
        x_denominator += (w[i+1])

        e_numerator += (w[i+1]**2)*V[i+1] # below eqn 17
        e_denominator += w[i+1]

    return x_numerator/x_denominator, np.sqrt(e_numerator/(e_denominator**2))


def load_completed_bins(directory=".",start_bin=None,end_bin=None,mask=None,return_index_only=False):
    """A function that loads in all model, time, flux, and error files within the current directory, while working out which bins have successfully completed fitting.

    Inputs (all optional):
    directory: the path to the files to load in. The default is the current working directory, which is nearly always correct
    start_bin: if wanting to ignore the first N bins, define this number. Default=None
    end_bin: if wanting to ignore the last N bins, define this number. Default=None
    mask: if wanting to mask certain bins, parse these bin indices as an array here. Default=None
    return_index_only - True/False : if wanting to only return the indices of the completed bins, set this to True. Default=False.

    Returns:
    if return_index_only:
        completed_bins - the indices of the completed bin fits
    else:
        x,y,e,e_r,m,m_in,w,we,completed_bins,nbins - arrays of time, flux, error, rescaled errors, TransitGPPM models, model input files, wavelength bin centres, wavelength bin widths, the indices of the completed bin fits, the number of bins with completed fits"
    """

    model_files = np.array(sorted(glob.glob('%s/prod_model_*.pickle'%directory)))

    # determine the completed bins by finding the XXX number in the "_wbXXX" in the file names
    completed_bins = np.array([int(m.split("wb")[-1].split(".")[0]) for m in model_files])
    if return_index_only:
        return completed_bins-1 # subtract 1 to account for 0 indexing
    nbins = len(completed_bins)

    ### Load in data arrays
    time_files = np.array(["%s/sigma_clipped_time_wb%s.pickle"%(directory,str(i).zfill(4)) for i in completed_bins])
    flux_files = np.array(["%s/sigma_clipped_flux_wb%s.pickle"%(directory,str(i).zfill(4)) for i in completed_bins])
    error_files = np.array(["%s/sigma_clipped_error_wb%s.pickle"%(directory,str(i).zfill(4)) for i in completed_bins])
    model_input_files = np.array(["%s/sigma_clipped_model_inputs_wb%s.pickle"%(directory,str(i).zfill(4)) for i in completed_bins])

    # For the error, we preferentially used rescaled errors. Either by reduced chi2 for PM fits or by white noise kernel in GP fits.
    rescaled_error_files = np.array(["%s/rescaled_errors_wb%s.pickle"%(directory,str(i).zfill(4)) for i in completed_bins])

    x = [pickle.load(open(i,'rb')) for i in time_files]
    y = [pickle.load(open(i,'rb')) for i in flux_files]
    e = [pickle.load(open(i,'rb')) for i in error_files]
    m_in = [pickle.load(open(i,'rb')) for i in model_input_files]
    try:
        e_r = [pickle.load(open(i,'rb')) for i in rescaled_error_files]
    except: # we might not have used the rescale error option, so we can't load anything in
        e_r = None

    ### Load in LD coefficients table for the wavelength centres and widths of the bins
    w,we = np.loadtxt('%s/LD_coefficients.txt'%directory,unpack=True,usecols=[0,1])
    w,we = np.atleast_1d(w)[completed_bins-1],np.atleast_1d(we)[completed_bins-1]

    ### Bin mask
    if mask is not None:
        bin_mask = np.array([1]*nbins) # start by including all bins
        bin_mask[mask] = 0 # replace mask bins with 0 in mask array
        bin_mask = bin_mask.astype(bool) # take boolean of array

        # mask the inputs
        x = x[mask]
        y = y[mask]
        e = e[mask]
        e_r = e_r[mask]
        model_files = model_files[mask]
        m_in = m_in[mask]
        w = w[mask]
        we = w[mask]

    ### Start and end bin cuts
    if start_bin is not None and end_bin is not None:
        x = x[start_bin:end_bin]
        y = y[start_bin:end_bin]
        e = e[start_bin:end_bin]
        e_r = e_r[start_bin:end_bin]
        model_files = model_files[start_bin:end_bin]
        m_in = m_in[start_bin:end_bin]
        w = w[start_bin:end_bin]
        we = we[start_bin:end_bin]

    # Now finally load in model files. Note: this is not converted to an array as doing so loses the object structure and only returns the parameter values.
    m = [pickle.load(open(i,'rb')) for i in model_files]

    return x,y,e,e_r,m,m_in,w,we,completed_bins,nbins


def bin_wave_to_R(w,R):
    """A function to bin a wavelength grid to a specified resolution

    Parameters
    ----------
    w : list of float or numpy array of float
    Wavelength axis to be rebinned
    R : float or int
    Resolution to bin axis to

    Returns
    -------
    list of float
    New wavelength axis at specified resolution
    """

    starting_wvl = w[0]
    stopping_wvl = w[-1]
    i = starting_wvl
    wvls = [starting_wvl]
    while i < stopping_wvl:
        delta_lam = i/R
        wvls.append(i+delta_lam)
        i += delta_lam
    # wvls.append(stopping_wvl)
    return np.array(wvls)


def bin_trans_spec(bin_edges,x,y,e1,e2=None):
    """A function to bin a transmission spectrum to a lower resolution using weighted means of depths within a bin.

    Inputs:
    bin_edges -- an array of the *edges* of the bins, not the centres
    x -- the input wavelength array
    y -- the input spectrum (depths or Rp/Rs)
    e1 -- the 1 sigma uncertainties on y. Can be defined as the upper errors if errors are asymmetric
	e2 -- the 1 sigma uncertainties on y. Can be defined as the lower errors if errors are asymmetric

    Returns:
    A dictionary of 'bin_x' the bin *centres*, 'bin_dx' the bin widths, 'bin_y' the binned spectrum, 'bin_dy' the binned uncertainties on the spectrum"""

    digitized = np.digitize(x,bin_edges)

    binned_x = []
    binned_xe = []
    binned_y = []
    binned_e = []

    for i in range(1,len(bin_edges)):
        if len(y[digitized==i]) == 0:
            continue

        if e2 is not None:
            mean_y,mean_e = weighted_mean_uneven_errors(y[digitized==i],e1[digitized==i],e2[digitized==i])
        else:
            mean_y,weights = np.average(y[digitized==i],weights=1/e1[digitized==i]**2,returned=True)
            mean_e = np.sqrt(1/weights)

        binned_x.append(x[digitized==i].mean())
        binned_xe.append(x[digitized==i].max()-x[digitized==i].min())
        binned_y.append(mean_y)
        binned_e.append(mean_e)

    binned = {"bin_x":np.array(binned_x),"bin_dx":np.array(binned_xe),"bin_y":np.array(binned_y),"bin_dy":np.array(binned_e)}

    return binned

def calc_bin_edges_from_centres(bin_centres):
    """A function that calculates the edges of bins from an array of bin centres, assuming that bin edges are halfway between the bin edges.

    Inputs:
    bin_centres - an array of wavelength bin centres

    Returns:
    bin_edges - the array of wavelength bin edges"""

    bin_edges = []
    bin_widths = np.diff(bin_centres)
    bin_edges = bin_centres[:-1] + bin_widths/2
    first_bin_edge = bin_centres[0] - bin_widths[0]/2
    last_bin_edge = bin_centres[-1] + bin_widths[-1]/2
    return np.hstack((first_bin_edge,bin_edges,last_bin_edge))



def make_corner_plot(sample_chains,bin_number,namelist,parameter_modes,save_fig=False,title=None):
    """Use DFM's corner package to make a corner plot of the emcee chains.

    Input:
    sample_chains - the emcee chains
    bin_number - the wavelength bin number, needed for saving the plot to file
    namelist - a list of parameter names corresponding to the chain
    parameter_modes - the modes of the parameter distributions
    save_fig - True/False: do we want to save the figure to file?
    title - set to a string if wanting to define where the plot is saved to, otherwise default is used. Default=None.

    Returns:
    Nothing - just plots the figure
    """

    print('Generating corner plot...')
    ndim = np.shape(sample_chains)[1]

    fig = corner(sample_chains,labels=namelist,quantiles=[0.16, 0.5, 0.84],verbose=False,show_titles=True)
    overplot_lines(fig, parameter_modes)

    if save_fig:
        if title is not None:
            fig.savefig(title)
        else:
            fig.savefig('cornerplot_wb%s.png'%(str(bin_number).zfill(4)))
        plt.close()
    else:
        plt.show()
