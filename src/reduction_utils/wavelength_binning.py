#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats, signal
import os
import pandas as pd
import pysynphot.binning as astrobin
import warnings as warn
from reduction_utils import wavelength_calibration as wc

### define the alkali metal lines, air wavelengths
sodium_d1 = 5890
sodium_d2 = 5896
sodium_centre = np.mean([sodium_d1,sodium_d2])

potassium_d1 = 7665
potassium_d2 = 7699
potassium_centre = np.mean([potassium_d1,potassium_d2])

Halpha = 6562

###

def rebin(xbins,x,y,e=None,weighted=False,errors_from_rms=False):

    """A function to rebin time series flux and errors given bin locations as a function of time.

    Inputs:
    xbins - location of bins
    x - the time series
    y - the data (flux) to be binned
    e - the errors on the flux data points. Can be left as None.
    weighted - should the data be binned as a weighted sum? True/False, default=False
    errors_from_rms - if e=None and you want the returned binned errors to correspond to the standard deviation of the binned y data, set this to True

    Returns:
    np.array(xbin) - the binned time series
    np.array(ybin) - the binned data (flux)
    np.array(ebin) - the binned error
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
                raise Exception('Cannot compute weighted mean without errors')
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


def nan_mean(array,axis=None):
    """A function that calculates the mean of an array with NaNs by ignoring the NaNs. Note, this differs to np.nanmean which replaces nans with zeros.

    Inputs:
    array - the array for which to calculate the mean of.
    axis - the axis along which to calculate the mean

    Returns:
    means - along the chosen axis."""

    not_nans = np.isfinite(array)
    ndimensions = len(array.shape)
    if ndimensions > 1:
        nframes,npixels = array.shape
        means = []
        for i in range(nframes):
            if np.any(not_nans[i]):
                means.append(np.mean(array[i][not_nans[i]]))
            else:
                means.append(np.zeros_like(array[i])*np.nan)
    else:
        means = np.mean(array[not_nans],axis=axis)

    return np.array(means)

def nan_median(array,axis=None):
    """A function that calculates the median of an array with NaNs by ignoring the NaNs. Note, this differs to np.nanmedian which replaces nans with zeros.

    Inputs:
    array - the array for which to calculate the median of.
    axis - the axis along which to calculate the median

    Returns:
    medians - along the chosen axis."""

    not_nans = np.isfinite(array)
    ndimensions = len(array.shape)
    if ndimensions > 1:
        nframes,npixels = array.shape
        medians = []
        for i in range(nframes):
            if np.any(not_nans[i]):
                medians.append(np.median(array[i][not_nans[i]]))
            else:
                medians.append(np.zeros_like(array[i])*np.nan)
    else:
        medians = np.median(array[not_nans],axis=axis)

    return np.array(medians)

def nan_sum(array,axis=None):
    """A function that calculates the sum of an array with NaNs by ignoring the NaNs. Note, this differs to np.nansum which replaces nans with zeros. However, this is equivalent/will give the same result.

    Inputs:
    array - the array for which to calculate the median of.
    axis - the axis along which to calculate the median

    Returns:
    sums - along the chosen axis."""

    not_nans = np.isfinite(array)
    return np.sum(array[not_nans],axis=axis)



def normalise_flux(flux,error,contact1,contact4,airmass=None,airmass_cut=2):
    """A function to normalise light curves to the out of transit flux points, using the median of the out of transit data.

    Inputs:
    flux - the 1D array of light curve flux
    error - the 1D array of errors on the flux data points
    contact1 - the location of the transit's first contact point (in index/data points not time)
    contact4 - the location of the transit's fourth contact point (in index/data points not time)
    airmass - can use this to perform an airmass cut to remove data points that fall above a certain airmass threshold. Default=None
    airmass_cut - if airmass != None, this sets the value at which to set the airmass cut. Default=2

    Returns:
    np.array(norm_flux) - the normalised array of fluxes
    np.array(norm_error) - the normalised array of errors
    """

    flux = np.atleast_2d(flux)
    error = np.atleast_2d(error)

    if airmass is not None:
        cut_flux = flux[:,airmass <= airmass_cut]
        cut_error = error[:,airmass <= airmass_cut]

    else:
        cut_flux = flux
        cut_error = error

    oot_flux = np.concatenate((cut_flux[:,0:contact1],cut_flux[:,contact4:]),axis=1)

    try:
        median_oot = nan_median(oot_flux,axis=1)
    except:
        median_oot = nan_median(oot_flux,axis=0)

    norm_flux = flux/median_oot.reshape(len(flux),1)

    norm_err = error/median_oot.reshape(len(flux),1)

    return norm_flux,norm_err



def plot_all_bins(mjd,flux,error,rebin_data=None):
    """A function that takes the normalised spectroscopic light curve fluxes and errors and plots them all on a single figure.

    Inputs:
    mjd - the 1D array of time series
    flux - the ndarray of the spectroscopic light curve fluxes
    error - the ndarray of the spectroscopic light curve errors
    rebin_data - set this parameter if wanting to rebin the data using the rebin_data function. Set this parameter to the number of bins desired.

    Returns:
    Nothing - just plots a figure
    """


    nbins = min(np.shape(flux))
    offsets = [0.02 * i for i in range(nbins)]

    mjd_off = mjd-int(mjd[0])
    if rebin_data is not None:
        xbins = np.linspace(mjd_off[0],mjd_off[-1],rebin_data)

    plt.figure()
    for i in range(nbins):

        if rebin_data is not None:
            x,y,e = rebin(xbins,mjd_off,flux[i],e=error[i],weighted=True,errors_from_rms=False)
        else:
            x = mjd_off
            y = flux[i]
            e = error[i]

        plt.errorbar(x,y-offsets[i],yerr=e,ecolor='k',fmt='o',color='r',ms=1)

    plt.ylabel('Normalised flux')
    plt.xlabel('MJD + %d'%int(mjd[0]))
    plt.show()
    return



def bin_ancillary_data(data,wavelength_solution,bins,n_tukey_points=0):
    """
    Bin ancillary data (such as x position) to same bins as used for spectra. Can also use Tukey windows here. NOTE: this is now superseded by wvl_bin_data which performs this function
    while constructing the spectroscopic light curves.

    Inputs:
    data - the ndarray of data to be binned
    wavelength_solution - the wavelength solution
    bins - the bin edges, in Angstroms
    n_tukey_points - If wanting to use Tukey bins, set this parameter to the number of points to fall within the Tukey smoothed edges per bin.
                     This downweights those points falling at the edges of bins. I often find this to lead to noisier light curves. Default=0 (no Tukey window is used)

    Returns:
    binned_data - the ndarray of wavelength binned data
    """
    digitized = np.digitize(wavelength_solution,bins)
    nbins = len(bins)

    binned_data = []

    for i in range(1,nbins):

        if n_tukey_points > 0:

            nidx = len(np.where(digitized == i)[0])
            Tukey = float(n_tukey_points)/nidx
            dummy_bin_vals = np.zeros((len(data),nidx+2))
            dummy_bin_vals[:,1:-1] = data[:,digitized == i]

            tukey_bin_vals = dummy_bin_vals*signal.tukey(nidx+2,Tukey)
            bin_vals = tukey_bin_vals[:,1:-1].mean(axis=1)

        else:
            bin_vals = data[:,digitized==i].mean(axis=1)

        binned_data.append(bin_vals)

    return np.array(binned_data)


def wvl_bin_data(flux1,err1,flux2,err2,wvl_solution,bins,ancillary_data=None,weighted=False,n_tukey_points=0,wvl_solution_2=None):

    """A function to bin the spectra of the target and comparison to make spectroscopic light curves for each by summing the flux within the defined wavelength bins.
    The target's light curves are divided by the comparison's light curves to correct for telluric extinction.

    Inputs:
    flux1 - ndarray of spectra of the target
    err1 - ndarray of errors of the target
    flux2 - ndarray of spectra of the comparison
    err2 - ndarray of errors of the comparison
    wvl_solution - the wavelength solution which is used to bin the data. This is assuming that both the target and the comparison have been resampled onto the same x-axis.
                   My tests show that this provides much better light curves than using separate wavelength solutions
    bins - the list of bin edges of the light curves, in Angstroms
    ancillary_data - a dictionary of ancillary data to bin (typically ancillary_data = {"xpos":xpos,"sky":sky})
    weighted - True/False - define whether we want to perform a weighted (by the flux errors) sum or not. Default=False as I have found this often produce better light curves (less noise)
    n_tukey_points - If wanting to use Tukey bins, set this parameter to the number of points to fall within the Tukey smoothed edges per bin.
                     This downweights those points falling at the edges of bins. I often find this to lead to noisier light curves. Default=0 (no Tukey window is used)
     wvl_solution_2 - can pass second wavelength solution for second object if the stars have not been resampled onto the same wavelength grid

    Returns:
    flux_ratio - the differential (target/comparison) spectroscopic light curves
    err_ratio - the flux errors in the differential light curves
    binned_flux1 - the target's spectroscopic light curves
    binned_err1 - the flux errors in the target's spectroscopic light curves
    binned_flux2 - the comparison's spectroscopic light curves
    binned_err2 - the flux errors in the comparison's spectroscopic light curves
    binned_ancillary - a dictionary of the binned ancillary data
    photon_noise_star1 - the photon noise for each bin for the target
    photon_noise_star2 - the photon noise for each bin for the comparison
    SN_1 - the mean of the S/N array (sqrt(flux)) for each wavelength bin for star 1
    SN_2 - the mean of the S/N array (sqrt(flux)) for each wavelength bin for star 2

    """

    nbins = len(bins)
    nframes = len(flux1)

    binned_flux1 = []
    binned_err1 = []

    binned_flux2 = []
    binned_err2 = []

    binned_ancillary_data = {}
    if ancillary_data is not None:
        for k in ancillary_data.keys():
            binned_ancillary_data[k] = []

    photon_noise_star1 = []
    photon_noise_star2 = []

    SN_1 = []
    SN_2 = []

    # make wavelength solution at least 2D to allow for separate wavelength solutions for each frame
    if len(wvl_solution.shape) == 1:
        wvl_solution = np.ones_like(flux1) * wvl_solution

    # allow for second wavelength solution if the stars have not been resampled
    if wvl_solution_2 is not None:
        if len(wvl_solution_2.shape) == 1:
            wvl_solution_2 = np.ones_like(flux2) * wvl_solution_2

    # outer loop: loop through all frames (1D spectra)
    for i in range(nframes):

        current_flux1 = []
        current_flux2 = []

        current_err1 = []
        current_err2 = []

        if ancillary_data is not None:
            current_ancil = {}
            for k in ancillary_data.keys():
                current_ancil[k] = []

        current_photon_noise_1 = []
        current_photon_noise_2 = []

        current_SN_1 = []
        current_SN_2 = []

        # inner loop: loop through all bins
        for j in range(0,nbins-1):

            idx = (wvl_solution[i] >= bins[j]) & (wvl_solution[i] < bins[j+1])
            nidx = len(np.where(flux1[i][idx])[0])

            if wvl_solution_2 is not None:
                idx2 = (wvl_solution_2[i] >= bins[j]) & (wvl_solution_2[i] < bins[j+1])
                nidx2 = len(np.where(flux2[i][idx2])[0])
            else:
                idx2 = idx
                nidx2 = nidx

            if n_tukey_points != 0:

                Tukey1 = float(n_tukey_points)/nidx
                Tukey2 = float(n_tukey_points)/nidx2

                # Create dummy array, 2 points longer than our array so that points weighted as 0 by Tukey window fall outside our desired array and thereofre none of our points are 0 weighted
                dummy_bin1_vals = np.zeros(nidx+2)
                dummy_e1_vals = np.zeros(nidx+2)

                # replace the values between the first and last dummy points with the actual data
                dummy_bin1_vals[1:-1] = flux1[i][idx]
                dummy_e1_vals[1:-1] = err1[i][idx]

                tukey_bin1_vals = dummy_bin1_vals*signal.tukey(nidx+2,Tukey1)
                bin_1_vals = tukey_bin1_vals[1:-1]

                tukey_e1_vals = dummy_e1_vals*signal.tukey(nidx+2,Tukey1)
                bin_e1_vals = tukey_e1_vals[1:-1]

                if flux2 is not None:
                    dummy_bin2_vals = np.zeros(nidx2+2)
                    dummy_e2_vals = np.zeros(nidx2+2)

                    dummy_bin2_vals[1:-1] = flux2[i][idx2]
                    dummy_e2_vals[1:-1] = err2[i][idx2]

                    tukey_bin2_vals = dummy_bin2_vals*signal.tukey(nidx2+2,Tukey2)
                    bin_2_vals = tukey_bin2_vals[1:-1]

                    tukey_e2_vals = dummy_e2_vals*signal.tukey(nidx2+2,Tukey2)
                    bin_e2_vals = tukey_e2_vals[1:-1]

                if ancillary_data is not None:

                    bin_ancillary = {}
                    for k in ancillary_data.keys():
                        dummy_ancil_vals = np.zeros(nidx+2)
                        dummy_ancil_vals[1:-1] = ancillary_data[k][i][idx]
                        tukey_ancil_vals = dummy_ancil_vals*signal.tukey(nidx+2,Tukey1)
                        bin_ancillary[k] = tukey_ancil_vals[1:-1]


            else:
                bin_1_vals = flux1[i][idx]
                bin_e1_vals = err1[i][idx]

                if flux2 is not None:
                    bin_2_vals = flux2[i][idx2]
                    bin_e2_vals = err2[i][idx2]

                if ancillary_data is not None:
                    bin_ancillary = {}
                    for k in ancillary_data.keys():
                        bin_ancillary[k] = ancillary_data[k][i][idx]

            if ancillary_data is not None:
                for k in ancillary_data.keys():
                    current_ancil[k].append(nan_mean(bin_ancillary[k]))

            current_photon_noise_1.append(np.mean(1/np.sqrt(flux1[i][idx])))
            current_SN_1.append(np.sqrt(bin_1_vals).mean())

            if flux2 is not None:
                current_photon_noise_2.append(np.mean(1/np.sqrt(flux2[i][idx2])))
                current_SN_2.append(np.sqrt(bin_2_vals).mean())

            if weighted:

                weights1 = 1.0/bin_e1_vals**2
                current_flux1.append( np.sum(weights1*bin_1_vals) / np.sum(weights1) )
                current_err1.append( np.sqrt(1.0/np.sum(weights1) ) )

                if flux2 is not None:
                    weights2 = 1.0/bin_e2_vals**2
                    current_flux2.append( np.sum(weights2*bin_2_vals) / np.sum(weights2) )
                    current_err2.append( np.sqrt(1.0/np.sum(weights2) ) )

            else:
                current_flux1.append(np.nansum(bin_1_vals))
                current_err1.append(np.sqrt(np.nansum(bin_e1_vals**2)) )

                if flux2 is not None:
                    current_flux2.append(np.nansum(bin_2_vals))
                    current_err2.append(np.sqrt(np.nansum(bin_e2_vals**2)) )

        photon_noise_star1.append(current_photon_noise_1)
        binned_flux1.append(np.array(current_flux1))
        binned_err1.append(np.array(current_err1))

        if flux2 is not None:
            photon_noise_star2.append(current_photon_noise_2)
            binned_flux2.append(np.array(current_flux2))
            binned_err2.append(np.array(current_err2))


        if ancillary_data is not None:
            for k in ancillary_data.keys():
                binned_ancillary_data[k].append(np.array(current_ancil[k]))

        SN_1.append(np.array(current_SN_1))

        if flux2 is not None:
            SN_2.append(np.array(current_SN_2))


    binned_flux1 = np.array(binned_flux1)
    binned_err1 = np.array(binned_err1)

    if flux2 is not None:
        binned_flux2 = np.array(binned_flux2)
        binned_err2 = np.array(binned_err2)

        flux_ratio = (binned_flux1/binned_flux2)
        err_ratio = np.sqrt((binned_err1/binned_flux1)**2 + (binned_err2/binned_flux2)**2)*flux_ratio
    else:
        flux_ratio = binned_flux1
        err_ratio = binned_err1

    ancillary_data_norm = {}
    for k in ancillary_data.keys():
        ancil = np.transpose(binned_ancillary_data[k])
        ancil_norm = [(i - i.mean())/i.std() for i in ancil]
        ancillary_data_norm[k] = np.array(ancil_norm)

    if flux2 is None:
        return np.transpose(flux_ratio),np.transpose(err_ratio),ancillary_data_norm,\
           np.transpose(np.array(photon_noise_star1)),np.transpose(SN_1)

    else:
        return np.transpose(flux_ratio),np.transpose(err_ratio),np.transpose(binned_flux1),\
               np.transpose(binned_err1),np.transpose(binned_flux2),np.transpose(binned_err2),ancillary_data_norm,\
               np.transpose(np.array(photon_noise_star1)),np.transpose(np.array(photon_noise_star2)),np.transpose(SN_1),np.transpose(SN_2)


def create_wvl_bins(wvl_solution,bin_width=None,first_pixel=0,last_pixel=None,native_resolution=False):
    """A function that defines the bin edges and centres given bin width (in pixels) + first and last pixels.

    Inputs:
    wvl_solution - the wavelength solution in A/um
    bin_width - the width of the bins in pixels. Not necessary if native_resolution = Tue
    first_pixel - the first pixel to start the bins at. Default = first pixel
    last_pixel - the final pixel to end the bins at. Default = final pixel
    native_resolution - True/False - are we working at the native resolution of the instrument?
                    If so, the bin centres = wvl solution and bin_width = 1

    Returns:
    bin_edges,bin_centres,bin_widths,nbins - the edges, centres, widths (all in wavelength units) and number of bins with this setup"""

    # ~ if last_pixel is None:
        # ~ last_pixel = len(wvl_solution)

    # ~ if last_pixel == len(wvl_solution):
        # ~ last_pixel -= 1

    if native_resolution:
        bin_centres = wvl_solution[first_pixel:last_pixel]
        bin_widths = np.diff(bin_centres)/2
        bin_widths = np.hstack((bin_widths[0],bin_widths))
        bin_edges = wvl_solution[first_pixel:last_pixel]
    else:
        if last_pixel is None:
            last_pixel = -1
        bin_edges = np.hstack((wvl_solution[first_pixel:last_pixel:bin_width],wvl_solution[last_pixel]))
        bin_centres = bin_edges[:-1]+np.diff(bin_edges)/2.
        bin_widths = np.diff(bin_edges)

    nbins = len(bin_centres)
    print("%d bins created with this setup, with a resolution R between %d--%d and a mean R = %d"%(nbins,np.min(bin_centres/bin_widths),np.max(bin_centres/bin_widths),np.mean(bin_centres/bin_widths)))
    return bin_edges,bin_centres,bin_widths,nbins




def simple_bin(flux,flux_error,ancillary_data,wvl_solution,bin_edges,weighted=True,native_resolution=False,standardise_input=True,):

    """A function that takes advantage of numpy to speed up the binning of large data sets.
    Much preferable to wvl_bin_data() when analysing a single star.

    Inputs:
    flux - the stellar spectra for each frame
    flux_error the uncertainties in the stellar spectra for each frame
    ancillary_data - a dictionary of ancillary data to be binned (e.g., background, x position)
    wvl_solution - the wavelength solution / wavelength array
    bin_edges - the edges of the bins in a 1D array/list
    weighted - True/False - do you want to perform a weighted mean? Default = True
    native_resolution - True/False - are we making light curves at the native resolution? If so, bin_edges and weighted is ignored
    standardise_input - True/False - do you want to standardise the ancillary data? (data-data.mean())/data.std(). Default = True.


    Returns:
    bf - the wavelength binned light curves
    be - the photometric uncertainty in the wavelength binned light curves
    ba - the wavelength binned ancillary data in the form of a dictionary"""

    binned_flux = []
    binned_error = []
    binned_ancillary = {}
    if ancillary_data is not None:
        for k in ancillary_data.keys():
            binned_ancillary[k] = []

    nframes,npixels = flux.shape

    if native_resolution:
        nbins = len(wvl_solution)
    else:
        nbins = len(bin_edges) - 1

    for i in range(nbins):
        print("Working on bin %d/%d"%(i+1,nbins))

        if native_resolution:
            index = np.array([i])

        elif nbins == 1 or i == nbins - 1: # we're working with a white light curve
            index = np.where(((wvl_solution >= bin_edges[i]) & (wvl_solution <= bin_edges[i+1])))[0]
            if len(index) == 0:
                print("**Warning: no data falls within this wavelength bin -- the bin is empty**")
                continue
            bin_left = index.min()
            bin_right = index.max()
            npoints_in_bin = len(index)

        else:
            index = np.where(((wvl_solution >= bin_edges[i]) & (wvl_solution < bin_edges[i+1])))[0]
            if len(index) == 0:
                print("**Warning: no data falls within this wavelength bin -- the bin is empty**")
                continue
            bin_left = index.min()
            bin_right = index.max()
            npoints_in_bin = len(index)


        if weighted and not native_resolution:
            weights = 1/flux_error[:,index]**2
            bf = np.sum(weights*flux[:,index],axis=1)/np.sum(weights,axis=1)
            be = np.sqrt(1.0/np.nansum(weights,axis=1))

            for k in ancillary_data.keys():
                ba = np.sum(weights*ancillary_data[k][:,index],axis=1)/np.sum(weights,axis=1)

                binned_ancillary[k].append(ba)

        else:
            print("N points in bin =",flux[:,index].shape)
            bf = np.nansum(flux[:,index],axis=1)
            be = np.sqrt(np.nansum(flux_error[:,index]**2,axis=1))#/npoints_in_bin

            for k in ancillary_data.keys():
                ba = np.nansum(ancillary_data[k][:,index],axis=1)
                if standardise_input:
                    ba = (ba-np.nanmean(ba))/np.nanstd(ba)
                binned_ancillary[k].append(ba)

        binned_flux.append(bf)
        binned_error.append(be)

    for k in binned_ancillary.keys():
        binned_ancillary[k] = np.array(binned_ancillary[k])

    return np.array(binned_flux),np.array(binned_error),binned_ancillary


def plot_spectra(star1,star2,wvl_solution,wvl_solution_2=None,bin_edges=None,bin_centres=None,alkali=False,save_fig=False,ratio=True,xmin=None,xmax=None):

    """A function that plots the spectra of target and comparison and the ratio of these, along with bin boundaries and ability to plot telluric spectra.

    Inputs:
    star1 - the 1D spectrum of the target
    star2 - the 1D spectrum of the comparison. Can be set to None if working with a single star (e.g., space-based data)
    wvl_solution - the wavelength solution (array of wavelengths)
    wvl_solution_2 - the wavelength solution for star 2, in case the stellar spectra were not resampled onto the same x-axis. Default=None
    bin_edges - a list of the locations of the edges of the wavelength bins if wanting to overplot them. Default=None
    bin_centres - a list of the locations of the centres of the wavelength bins if wanting to overplot them as text on the figure. Default=None
    alkali - True/False - use this if wanting to overplot vertical lines at the locations of Na & K. Default=True
    save_fig - True/False - use this if wanting to save the figure to file, saved as 'wavelength_bins.pdf'. Default=False
    ratio - True/False - use this if wanting to plot the ratio of star1/star2 to identify where residual features exist to avoid setting bin edges there. Default=True
    xmin - set the minimum wavelength to be plotted. Default=None
    xmax - set the maximum wavelength to be plotted. Default=None

    Returns:
    Nothing, it only plots the figure
    """

    if ratio:
        plt.figure()
    else:
        plt.figure(figsize=(7,4))

    if ratio:
        nplots = 2
    else:
        nplots = 1


    plt.subplot(nplots,1,1)

    # ratio of spectra
    if ratio:

        if wvl_solution_2 is not None:
            print("WARNING! If the wavelength solutions differ between the stars, the plotted ratio is not strictly accurate")

        if bin_edges is not None:
            for i in bin_edges:
                plt.axvline(i,color='k',ls='--')

        if bin_centres is not None:
            for t in bin_centres:
                plt.text(t,0.2,t,rotation='vertical',ha='center')


        plt.plot(wvl_solution,star1/star2)

        if alkali:

            plt.axvline(sodium_d1,color='g')
            plt.axvline(sodium_d2,color='g')
            plt.axvline(potassium_d1,color='g')
            plt.axvline(potassium_d2,color='g')

        plt.xticks(visible=False)
        plt.ylabel('Flux ratio')

        if xmin is not None and xmax is not None:
            plt.xlim(xmin,xmax)

        plt.subplot(212)

    if bin_edges is not None:
        for i in bin_edges:
            plt.axvline(i,color='k',ls='--')

    if alkali:
        plt.axvline(sodium_d1,color='g')
        plt.axvline(sodium_d2,color='g')
        plt.axvline(potassium_d1,color='g')
        plt.axvline(potassium_d2,color='g')

    if wvl_solution_2 is None:
        wvl_solution_2 = wvl_solution

    nspectra = len(np.atleast_2d(star1))

    if nspectra > 1: # we're plotting all spectra
        for i in range(nspectra):
            plt.plot(wvl_solution,wc.normalise(star1[i],maximum=True),'b',alpha=0.5)

            if star2 is not None:
                plt.plot(wvl_solution,wc.normalise(star1[i],maximum=True),'r',alpha=0.5)
    else:
        plt.plot(wvl_solution,wc.normalise(star1,maximum=True),'b')
        if star2 is not None:
            plt.plot(wvl_solution_2,wc.normalise(star2,maximum=True),'r')

    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Normalised flux')
    plt.ylim(0,1.1)
    if xmin is not None and xmax is not None:
        plt.xlim(xmin,xmax)

    plt.subplots_adjust(hspace=0)

    if save_fig:
        plt.savefig('wavelength_bins.pdf',bbox_inches='tight')
    plt.show()



def iib_bins(line,flux1,err1,flux2,err2,wvl_solution,xpos,sky,bin_widths=np.arange(10,110,10),weighted=False,n_tukey_points=0,wvl_solution_2=None):

    iib_fluxes = []
    iib_errors = []
    iib_xpos = []
    iib_sky = []
    iib_centres = []
    iib_widths = []

    if line == 'Na':
        line_centre = sodium_centre
    elif line == 'K1':
        line_centre = potassium_d1
    elif line == 'K2':
        line_centre = potassium_d2
    elif line == 'K':
        line_centre = potassium_centre
    elif line == 'Halpha':
        line_centre = Halpha
    else:
        return NameError("line must be one of 'Na', 'K1', 'K2', 'K' or 'Halpha'")

    for i in bin_widths:

        if line == 'K1' or line == 'K2' or line == 'K':
        # left hand boundary is 7645A
            if line_centre - i/2 >= 7645:
                bin_left = line_centre - i/2
                bin_right = line_centre + i/2
                iib_centres.append(line_centre)
                iib_widths.append(i)
            else: # bin expands towards the red
                bin_left = 7645
                bin_right = bin_left + i
                iib_centres.append((bin_left+bin_right)//2)
                iib_widths.append(i)

        else:
            bin_left = line_centre - i/2
            bin_right = line_centre + i/2
            iib_centres.append(line_centre)
            iib_widths.append(i)

        curr_flux,curr_error,_,_,_,_,curr_xpos,curr_sky,_,_,_,_ = wvl_bin_data(flux1,err1,flux2,err2,wvl_solution,np.array([bin_left,bin_right]),n_tukey_points=n_tukey_points,xpos=xpos,sky=sky,weighted=weighted,wvl_solution_2=wvl_solution_2)

        iib_fluxes.append(curr_flux[0])
        iib_errors.append(curr_error[0])
        iib_xpos.append(curr_xpos[0])
        iib_sky.append(curr_sky[0])

    iib_fluxes = np.array(iib_fluxes)
    iib_errors = np.array(iib_errors)
    iib_centres = np.array(iib_centres)
    iib_widths = np.array(iib_widths)
    iib_xpos = np.array(iib_xpos)
    iib_sky = np.array(iib_sky)

    return iib_fluxes,iib_errors,iib_centres,iib_widths,iib_xpos,iib_sky

def bin_down_data(time_bins,time,flux,error,xpos=None,sky=None,errors_from_rms=False):

    nbins = len(np.atleast_1d(flux))
    binned_fluxes = []
    binned_errors = []
    binned_xpos = []
    binned_sky = []

    for i in range(nbins):

        bt, bf, be = rebin(time_bins,time,flux[i],error[i],\
                              weighted=False,errors_from_rms=errors_from_rms)

        binned_fluxes.append(bf)
        binned_errors.append(be)

        if xpos is not None:
            _,bx,_ = rebin(time_bins,time,xpos[i],None,\
                              weighted=False,errors_from_rms=errors_from_rms)
            binned_xpos.append(bx)

        if sky is not None:
            _,bs,_ = rebin(time_bins,time,sky[i],None,\
                              weighted=False,errors_from_rms=errors_from_rms)
            binned_sky.append(bs)

    finite_index = np.isfinite(bt)

    binned_time = np.array(bt)[finite_index]
    binned_fluxes = np.array(binned_fluxes)[:,finite_index]
    binned_errors = np.array(binned_errors)[:,finite_index]

    if xpos is not None:
        binned_xpos = np.array(binned_xpos)[:,finite_index]
    if sky is not None:
        binned_sky = np.array(binned_sky)[:,finite_index]

    return binned_time,binned_fluxes,binned_errors,binned_xpos,binned_sky


def wvl_bin_data_different_wvl_solutions(flux1,err1,flux2,err2,wvl_solutions, bins,xpos,sky,weighted=False,n_tukey_points=0):

    """A function to bin the spectra of the target and comparison to make spectroscopic light curves for each by summing the flux within the defined wavelength bins.
    The target's light curves are divided by the comparison's light curves to correct for telluric extinction.
    Inputs:
    flux1 - ndarray of spectra of the target
    err1 - ndarray of errors of the target
    flux2 - ndarray of spectra of the comparison
    err2 - ndarray of errors of the comparison
    wvl_solutions - the wavelength solution which is used to bin the data for both star1 & star2, expects individual wvl solution for each frame.
                   My tests show that this provides much better light curves than using separate wavelength solutions
    bins - the list of bin edges of the light curves, in Angstroms
    xpos - the ndarray of the average (**non-standarized**) x positions of the target and comparison, so that they can be binned to the same wavelength bins
    sky - the ndarray of the average (**non-standarized**) sky background of the target and comparison, so that they can be binned to the same wavelength bins
    weighted - True/False - define whether we want to perform a weighted (by the flux errors) sum or not. Default=False as I have found this often produce better light curves (less noise)
    n_tukey_points - If wanting to use Tukey bins, set this parameter to the number of points to fall within the Tukey smoothed edges per bin.
                     This downweights those points falling at the edges of bins. I often find this to lead to noisier light curves. Default=0 (no Tukey window is used)
    Returns:
    flux_ratio - the differential (target/comparison) spectroscopic light curves
    err_ratio - the flux errors in the differential light curves
    binned_flux1 - the target's spectroscopic light curves
    binned_err1 - the flux errors in the target's spectroscopic light curves
    binned_flux2 - the comparison's spectroscopic light curves
    binned_err2 - the flux errors in the comparison's spectroscopic light curves
    XPOS_norm - the **standarized**, combined, wavelength-binned x positions
    SKY_norm - the **standarized**, combined, wavelength-binned sky background
    photon_noise_star1 - the photon noise for each bin for the target
    photon_noise_star2 - the photon noise for each bin for the comparison
    """

    nbins = len(bins)
    nframes = len(flux1)

    binned_flux1 = []
    binned_err1 = []

    binned_flux2 = []
    binned_err2 = []

    binned_xpos = []
    binned_sky = []

    photon_noise_star1 = []
    photon_noise_star2 = []

    # outer loop: loop through all frames (1D spectra)
    for i in range(nframes):

        current_flux1 = []
        current_flux2 = []

        current_err1 = []
        current_err2 = []

        current_xpos = []
        current_sky = []

        current_photon_noise_1 = []
        current_photon_noise_2 = []

        current_wvl_solution = wvl_solutions[i]

        # inner loop: loop through all bins
        for j in range(0,nbins-1):

            idx = (current_wvl_solution >= bins[j]) & (current_wvl_solution < bins[j+1])
            nidx = len(np.where(flux1[i][idx])[0])

            if n_tukey_points != 0:

                Tukey1 = float(n_tukey_points)/nidx

                # Create dummy array, 2 points longer than our array so that points weighted as 0 by Tukey window fall outside our desired array and thereofre none of our points are 0 weighted
                dummy_bin1_vals = np.zeros(nidx+2)
                dummy_e1_vals = np.zeros(nidx+2)

                # replace the values between the first and last dummy points with the actual data
                dummy_bin1_vals[1:-1] = flux1[i][idx]
                dummy_e1_vals[1:-1] = err1[i][idx]

                tukey_bin1_vals = dummy_bin1_vals*signal.tukey(nidx+2,Tukey1)
                bin_1_vals = tukey_bin1_vals[1:-1]

                tukey_e1_vals = dummy_e1_vals*signal.tukey(nidx+2,Tukey1)
                bin_e1_vals = tukey_e1_vals[1:-1]

                dummy_bin2_vals = np.zeros(nidx+2)
                dummy_e2_vals = np.zeros(nidx+2)

                dummy_bin2_vals[1:-1] = flux2[i][idx]
                dummy_e2_vals[1:-1] = err2[i][idx]

                tukey_bin2_vals = dummy_bin2_vals*signal.tukey(nidx+2,Tukey1)
                bin_2_vals = tukey_bin2_vals[1:-1]

                tukey_e2_vals = dummy_e2_vals*signal.tukey(nidx+2,Tukey1)
                bin_e2_vals = tukey_e2_vals[1:-1]

                dummy_xpos_vals = np.zeros(nidx+2)
                dummy_sky_vals = np.zeros(nidx+2)

                dummy_xpos_vals[1:-1] = xpos[i][idx]
                dummy_sky_vals[1:-1] = sky[i][idx]

                tukey_xpos_vals = dummy_xpos_vals*signal.tukey(nidx+2,Tukey1)
                tukey_sky_vals = dummy_sky_vals*signal.tukey(nidx+2,Tukey1)

                bin_xpos = tukey_xpos_vals[1:-1]
                bin_sky = tukey_sky_vals[1:-1]


            else:
                bin_1_vals = flux1[i][idx]
                bin_e1_vals = err1[i][idx]

                bin_2_vals = flux2[i][idx]
                bin_e2_vals = err2[i][idx]

                bin_xpos = xpos[i][idx]
                bin_sky = sky[i][idx]

            current_xpos.append(bin_xpos.mean())
            current_sky.append(bin_sky.mean())

            current_photon_noise_1.append(np.mean(1/np.sqrt(flux1[i][idx])))
            current_photon_noise_2.append(np.mean(1/np.sqrt(flux2[i][idx])))

            if weighted:

                weights1 = 1.0/bin_e1_vals**2
                weights2 = 1.0/bin_e2_vals**2

                # replace inf values if using Tukey
                #weights1[weights1 == np.inf] = 0
                #weights2[weights2 == np.inf] = 0

                current_flux1.append( np.sum(weights1*bin_1_vals) / np.sum(weights1) )
                current_flux2.append( np.sum(weights2*bin_2_vals) / np.sum(weights2) )

                current_err1.append( np.sqrt(1.0/np.sum(weights1) ) )
                current_err2.append( np.sqrt(1.0/np.sum(weights2) ) )

            else:
                current_flux1.append(bin_1_vals.sum())
                current_flux2.append(bin_2_vals.sum())

                current_err1.append(np.sqrt(np.sum(bin_e1_vals**2)) )
                current_err2.append(np.sqrt(np.sum(bin_e2_vals**2)) )

        photon_noise_star1.append(current_photon_noise_1)
        photon_noise_star2.append(current_photon_noise_2)

        binned_flux1.append(np.array(current_flux1))
        binned_flux2.append(np.array(current_flux2))

        binned_err1.append(np.array(current_err1))
        binned_err2.append(np.array(current_err2))

        binned_xpos.append(np.array(current_xpos))
        binned_sky.append(np.array(current_sky))


    binned_flux1 = np.array(binned_flux1)
    binned_err1 = np.array(binned_err1)

    binned_flux2 = np.array(binned_flux2)
    binned_err2 = np.array(binned_err2)

    flux_ratio = (binned_flux1/binned_flux2)
    err_ratio = np.sqrt((binned_err1/binned_flux1)**2 + (binned_err2/binned_flux2)**2)*flux_ratio

    XPOS = np.transpose(np.array(binned_xpos))
    XPOS_norm = [(i - i.mean())/i.std() for i in XPOS]

    SKY = np.transpose(np.array(binned_sky))
    SKY_norm = [(i - i.mean())/i.std() for i in SKY]

    return np.transpose(flux_ratio),np.transpose(err_ratio),np.transpose(binned_flux1),\
           np.transpose(binned_err1),np.transpose(binned_flux2),np.transpose(binned_err2),np.array(XPOS_norm),np.array(SKY_norm),\
           np.transpose(np.array(photon_noise_star1)),np.transpose(np.array(photon_noise_star2))



def wvl_bin_data_indivdual_wvl_solutions(flux1,err1,flux2,err2,wvl_solution1, wvl_solution2, bins,xpos,sky,weighted=False,n_tukey_points=0):

    """A function to bin the spectra of the target and comparison to make spectroscopic light curves for each by summing the flux within the defined wavelength bins.
    The target's light curves are divided by the comparison's light curves to correct for telluric extinction.
    Inputs:
    flux1 - ndarray of spectra of the target
    err1 - ndarray of errors of the target
    flux2 - ndarray of spectra of the comparison
    err2 - ndarray of errors of the comparison
    wvl_solutions - the wavelength solution which is used to bin the data for both star1 & star2, expects individual wvl solution for each frame.
    bins - the list of bin edges of the light curves, in Angstroms
    xpos - the ndarray of the average (**non-standarized**) x positions of the target and comparison, so that they can be binned to the same wavelength bins
    sky - the ndarray of the average (**non-standarized**) sky background of the target and comparison, so that they can be binned to the same wavelength bins
    weighted - True/False - define whether we want to perform a weighted (by the flux errors) sum or not. Default=False as I have found this often produce better light curves (less noise)
    n_tukey_points - If wanting to use Tukey bins, set this parameter to the number of points to fall within the Tukey smoothed edges per bin.
                     This downweights those points falling at the edges of bins. I often find this to lead to noisier light curves. Default=0 (no Tukey window is used)
    Returns:
    flux_ratio - the differential (target/comparison) spectroscopic light curves
    err_ratio - the flux errors in the differential light curves
    binned_flux1 - the target's spectroscopic light curves
    binned_err1 - the flux errors in the target's spectroscopic light curves
    binned_flux2 - the comparison's spectroscopic light curves
    binned_err2 - the flux errors in the comparison's spectroscopic light curves
    XPOS_norm - the **standarized**, combined, wavelength-binned x positions
    SKY_norm - the **standarized**, combined, wavelength-binned sky background
    photon_noise_star1 - the photon noise for each bin for the target
    photon_noise_star2 - the photon noise for each bin for the comparison
    """

    nbins = len(bins)
    nframes = len(flux1)

    binned_flux1 = []
    binned_err1 = []

    binned_flux2 = []
    binned_err2 = []

    binned_xpos = []
    binned_sky = []

    photon_noise_star1 = []
    photon_noise_star2 = []

    # outer loop: loop through all frames (1D spectra)
    for i in range(nframes):

        current_flux1 = []
        current_flux2 = []

        current_err1 = []
        current_err2 = []

        current_xpos = []
        current_sky = []

        current_photon_noise_1 = []
        current_photon_noise_2 = []

        current_wvl_solution1 = wvl_solution1[i]
        current_wvl_solution2 = wvl_solution2[i]

        # inner loop: loop through all bins
        for j in range(0,nbins-1):

            idx1 = (current_wvl_solution1 >= bins[j]) & (current_wvl_solution1 < bins[j+1])
            nidx1 = len(np.where(flux1[i][idx1])[0])
            idx2 = (current_wvl_solution2 >= bins[j]) & (current_wvl_solution2 < bins[j+1])
            nidx2 = len(np.where(flux2[i][idx2])[0])
            #print(idx1, idx2)

            if n_tukey_points != 0:

                Tukey1 = float(n_tukey_points)/nidx1

                # Create dummy array, 2 points longer than our array so that points weighted as 0 by Tukey window fall outside our desired array and thereofre none of our points are 0 weighted
                dummy_bin1_vals = np.zeros(nidx1+2)
                dummy_e1_vals = np.zeros(nidx1+2)

                # replace the values between the first and last dummy points with the actual data
                dummy_bin1_vals[1:-1] = flux1[i][idx1]
                dummy_e1_vals[1:-1] = err1[i][idx1]

                tukey_bin1_vals = dummy_bin1_vals*signal.tukey(nidx1+2,Tukey1)
                bin_1_vals = tukey_bin1_vals[1:-1]

                tukey_e1_vals = dummy_e1_vals*signal.tukey(nidx1+2,Tukey1)
                bin_e1_vals = tukey_e1_vals[1:-1]

                dummy_bin2_vals = np.zeros(nidx2+2)
                dummy_e2_vals = np.zeros(nidx2+2)

                dummy_bin2_vals[1:-1] = flux2[i][idx2]
                dummy_e2_vals[1:-1] = err2[i][idx2]

                tukey_bin2_vals = dummy_bin2_vals*signal.tukey(nidx2+2,Tukey1)
                bin_2_vals = tukey_bin2_vals[1:-1]

                tukey_e2_vals = dummy_e2_vals*signal.tukey(nidx2+2,Tukey1)
                bin_e2_vals = tukey_e2_vals[1:-1]

                dummy_xpos_vals = np.zeros(nidx1+2)
                dummy_sky_vals = np.zeros(nidx1+2)

                dummy_xpos_vals[1:-1] = xpos[i][idx1]
                dummy_sky_vals[1:-1] = sky[i][idx1]

                tukey_xpos_vals = dummy_xpos_vals*signal.tukey(nidx1+2,Tukey1)
                tukey_sky_vals = dummy_sky_vals*signal.tukey(nidx1+2,Tukey1)

                bin_xpos = tukey_xpos_vals[1:-1]
                bin_sky = tukey_sky_vals[1:-1]


            else:
                bin_1_vals = flux1[i][idx1]
                bin_e1_vals = err1[i][idx1]

                bin_2_vals = flux2[i][idx2]
                bin_e2_vals = err2[i][idx2]

                bin_xpos = xpos[i][idx1]
                bin_sky = sky[i][idx1]

            current_xpos.append(bin_xpos.mean())
            current_sky.append(bin_sky.mean())

            current_photon_noise_1.append(np.mean(1/np.sqrt(flux1[i][idx1])))
            current_photon_noise_2.append(np.mean(1/np.sqrt(flux2[i][idx2])))

            if weighted:

                weights1 = 1.0/bin_e1_vals**2
                weights2 = 1.0/bin_e2_vals**2

                # replace inf values if using Tukey
                #weights1[weights1 == np.inf] = 0
                #weights2[weights2 == np.inf] = 0

                current_flux1.append( np.sum(weights1*bin_1_vals) / np.sum(weights1) )
                current_flux2.append( np.sum(weights2*bin_2_vals) / np.sum(weights2) )

                current_err1.append( np.sqrt(1.0/np.sum(weights1) ) )
                current_err2.append( np.sqrt(1.0/np.sum(weights2) ) )

            else:
                current_flux1.append(bin_1_vals.sum())
                current_flux2.append(bin_2_vals.sum())

                current_err1.append(np.sqrt(np.sum(bin_e1_vals**2)) )
                current_err2.append(np.sqrt(np.sum(bin_e2_vals**2)) )

        photon_noise_star1.append(current_photon_noise_1)
        photon_noise_star2.append(current_photon_noise_2)

        binned_flux1.append(np.array(current_flux1))
        binned_flux2.append(np.array(current_flux2))

        binned_err1.append(np.array(current_err1))
        binned_err2.append(np.array(current_err2))

        binned_xpos.append(np.array(current_xpos))
        binned_sky.append(np.array(current_sky))


    binned_flux1 = np.array(binned_flux1)
    binned_err1 = np.array(binned_err1)

    binned_flux2 = np.array(binned_flux2)
    binned_err2 = np.array(binned_err2)


    flux_ratio = (binned_flux1/binned_flux2)
    err_ratio = np.sqrt((binned_err1/binned_flux1)**2 + (binned_err2/binned_flux2)**2)*flux_ratio

    XPOS = np.transpose(np.array(binned_xpos))
    XPOS_norm = [(i - i.mean())/i.std() for i in XPOS]

    SKY = np.transpose(np.array(binned_sky))
    SKY_norm = [(i - i.mean())/i.std() for i in SKY]

    return np.transpose(flux_ratio),np.transpose(err_ratio),np.transpose(binned_flux1),\
           np.transpose(binned_err1),np.transpose(binned_flux2),np.transpose(binned_err2),np.array(XPOS_norm),np.array(SKY_norm),\
           np.transpose(np.array(photon_noise_star1)),np.transpose(np.array(photon_noise_star2))


def binning(x, y,  dy=None, binwidth=None, r=None,newx= None, log = False, nan=False):
	"""
    Function written by N. Batalha.

	This contains functionality for binning spectroscopy given an x, y and set of errors.
	This is similar to IDL's regroup but in Python (obviously). Note that y is binned as the
	mean(ordinates) instead of sum(ordinates), as you would want for cmputing flux in a set of
	pixels. User can input a constant resolution, constant binwidth or provide a user defined
	bin. The error is computed as sum(sqrt(sig1^2, sig2^2, sig3^2) )/3, for example
	if there were 3 points to bin.

	Parameters
	----------
	x : array, float
		vector containing abcissae.
	y : array,float
		vector containing ordinates.
	dy : array,float
		(Optional) errors on ordinates, can be float or array
	binwidth : float
		(Optional) constant bin width in same units as x
	r : float
		(Optional) constant resolution to bin to. R is defined as w[1]/(w[2] - w[0])
		to maintain consistency with `pandeia.engine`
	newx : array, float
		(Optional) new x axis to bin to
	log : bool
		(Optional) computes equal bin spacing logarithmically, Default = False
	sort : bool
		(Optional) sort into ascending order of x, default = True
	nan : bool
		(Optional) if true, this returns nan values where no points exist in a given bin
		Otherwise, all nans are dropped

	Returns
	-------
	dict
		bin_y : binned ordinates
		bin_x : binned abcissae
		bin_edge : edges of bins (always contains len(bin_x)+1 elements)
		bin_dy : error on ordinate bin
		bin_n : number of points contained in each bin

	Examples
	--------

	>>> from bintools import binning

	If you want constant resolution (using output dict from PandExo):

	>>> pandexo = result['FinalSpectrum']
	>>> x, y, err = pandexo['wave'], pandexo['spectrum_w_rand'], pandexo['error_w_floor']
	>>> final = binning(x, y, dy = err, r =100)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']

	If you have a x axis that you want PandExo output to be binned to

	>>> newx = np.linspace(1,5,10)
	>>> final = binning(x, y, dy = err, newx =newx)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']

	If you want a constant bin width and want everything to be spaced linearly

	>>> final = binning(x, y, dy = err, binwidth = 0.1)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']

	If you want constant bin width but want everything to spaced logarithmically

	>>> final = binning(x, y, dy = err, binwidth = 0.1, log=True)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']
	"""
	#check x and y are same length
	if len(x) != len(y):
		raise Exception('X and Y are not the same length')

	#check that either newx or binwidth are specified

	if newx is None and binwidth is None and r is None:
		raise Exception('Need to either supply new x axis, resolution, or a binwidth')
	if (binwidth is None) and (log):
		raise Exception("Cannot do logarithmic binning with out a binwidth")

	if newx is not None:
		bin_x = newx
		bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
		bin_edge = astrobin.calculate_bin_edges(bin_x)

		return {'bin_y':bin_y, 'bin_x':bin_x, 'bin_edge':bin_edge, 'bin_dy':bin_dy, 'bin_n':bin_n}

	elif r is not None:
		bin_x = bin_wave_to_R(x, r)
		bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
		bin_edge = astrobin.calculate_bin_edges(bin_x)

		return {'bin_y':bin_y, 'bin_x':bin_x, 'bin_edge':bin_edge, 'bin_dy':bin_dy, 'bin_n':bin_n}


	elif binwidth is not None:
		if (binwidth < 0) and (log):
			warn.warn(UserWarning("Negative binwidth specified. Assuming this is log10(binwidth)"))
			binwidth = 10**binwidth
		if log:
			bin_x = np.arange(np.log10(min(x)),np.log10(max(x)),np.log10(binwidth))
			bin_x = 10**bin_x
		elif not log:
			bin_x = np.arange(min(x),max(x),binwidth)
		bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
		bin_edge = astrobin.calculate_bin_edges(bin_x)

		return {'bin_y':bin_y, 'bin_x':bin_x, 'bin_edge':bin_edge, 'bin_dy':bin_dy, 'bin_n':bin_n}




def uniform_tophat_mean(newx,x, y, dy=None,nan=False):
    """Adapted from Mike R. Line to rebin spectra

    Takes mean of groups of points in certain wave bin

    Parameters
    ----------
    newx : list of float or numpy array of float
    New wavelength grid to rebin to
    x : list of float or numpy array of float
    Old wavelength grid to get rid of
    y : list of float or numpy array of float
    New rebinned y axis

    Returns
    -------
    array of floats
    new wavelength grid

    Examples
    --------

    >>> from pandexo.engine.jwst import uniform_tophat_sum
    >>> oldgrid = np.linspace(1,3,100)
    >>> y = np.zeros(100)+10.0
    >>> newy = uniform_tophat_sum(np.linspace(2,3,3), oldgrid, y)
    >>> newy
    array([ 240.,  250.,  130.])
    """
    newx = np.array(newx)
    szmod=newx.shape[0]
    delta=np.zeros(szmod)
    ynew=np.zeros(szmod)
    bin_dy =np.zeros(szmod)
    bin_n =np.zeros(szmod)

    delta[0:-1]=newx[1:]-newx[:-1]
    delta[szmod-1]=delta[szmod-2]

    for i in range(szmod-1):
        i=i+1
        loc=np.where((x >= newx[i]-0.5*delta[i-1]) & (x < newx[i]+0.5*delta[i]))
        #make sure there are values within the slice
        if len(loc[0]) > 0:
            ynew[i]=np.mean(y[loc])
            if dy is not None:
                bin_dy[i] = np.sqrt(np.sum(dy[loc]**2.0))/len(y[loc])
                bin_n[i] = len(y[loc])
                #if not give empty slice a nan
        elif len(loc[0]) == 0 :
            warn.warn(UserWarning("Empty slice exists within specified new x, replacing value with nan"))
            ynew[i]=np.nan
            bin_n[i] = np.nan

    #fill in zeroth entry
    loc=np.where((x > newx[0]-0.5*delta[0]) & (x < newx[0]+0.5*delta[0]))
    if len(loc[0]) > 0:
        ynew[0]=np.mean(y[loc])
        bin_n[0] = len(y[loc])
        if dy is not None:
            bin_dy[0] = np.sqrt(np.sum(dy[loc]**2.0))/len(y[loc])
    elif len(loc[0]) is 0 :
        ynew[0]=np.nan
        bin_n[0] = np.nan
        if dy is not None:
            bin_dy[0] = np.nan

    #remove nans if requested
    out = pd.DataFrame({'bin_y':ynew, 'bin_x':newx, 'bin_dy':bin_dy, 'bin_n':bin_n})
    if not nan:
        out = out.dropna()

    return out['bin_x'].values,out['bin_y'].values, out['bin_dy'].values, out['bin_n'].values


def bin_wave_to_R(w, R):
	"""Creates new wavelength axis at specified resolution

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

	Examples
	--------

	>>> newwave = bin_wave_to_R(np.linspace(1,2,1000), 10)
	>>> print((len(newwave)))
	11
	"""
	wave = []
	tracker = min(w)
	i = 1
	ind= 0
	firsttime = True
	while(tracker<max(w)):
	    if i <len(w)-1:
	        dlambda = w[i]-w[ind]
	        newR = w[i]/dlambda
	        if (newR < R) & (firsttime):
	            tracker = w[ind]
	            wave += [tracker]
	            ind += 1
	            i += 1
	            firsttime = True
	        elif newR < R:
	            tracker = w[ind]+dlambda/2.0
	            wave +=[tracker]
	            ind = (np.abs(w-tracker)).argmin()
	            i = ind+1
	            firsttime = True
	        else:
	            firsttime = False
	            i+=1
	    else:
	        tracker = max(w)
	        wave += [tracker]
	return wave


def generate_wvls_at_R(starting_wvl,stopping_wvl,R):

    """JK's function to generate a wavelength grid with a desired (constant) resolution.

    Inputs:
    starting_wvl -- the first wavelength to consider
    stopping_wvl -- the final wavelength to consider
    R -- the desired spectral resolution of the resulting wavelength axis

    Returns:
    wvls -- the wavelength grid at the specified resolution"""

    wvls = []
    i = starting_wvl
    while i < stopping_wvl:
        delta_lam = i/R
        wvls.append(i+delta_lam)
        i += delta_lam

    return np.array(wvls)
