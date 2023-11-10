#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import median_abs_deviation as mad
from scipy.ndimage import median_filter as MF
from scipy.ndimage import interpolation
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import pickle
import os
from collections import Counter
from Tiberius.src.global_utils import parseInput
try:
    import astroscrappy
except:
    print("astroscrappy not imported, automatic cosmic ray detection can't be performed with lacosmic")
import copy
from cosmic_removal import interp_bad_pixels
from wavelength_calibration import rebin_spec
from astropy import units as u
from Keck_utils import Keck_order_masking as KO
from astropy.time import Time,TimeDelta


# Prevent matplotlib plotting frames upside down
plt.rcParams['image.origin'] = 'lower'


def gauss(x,amplitude,mean,std,offset):
    """A Gaussian with a fitted flux offset"""
    return amplitude*np.exp(-(x-mean)**2/(std**2))+offset


def BIC(model,data,error,n):
    """Use to calculate the Bayesian Information Criterion."""
    residuals = (model-data)/error
    chi2 = np.sum(residuals*residuals)
    bic = chi2 + n
    return bic

def find_spectral_trace(frame,guess_location,search_width,gaussian_width,trace_poly_order,trace_spline_sf,star=None,verbose=False,co_add_rows=0,instrument=None):
    """The function used to extract the location of a spectral trace either with a Gaussian or the argmax and then
    fits a nth order polynomial to these locations"""


    if "JWST" in instrument: # we need to extract only the first array since the frame is an array of (flux_frame,error_frame)
        frame = frame[0]

    if instrument == "Keck/NIRSPEC":
        search_frame = frame.copy()
        search_frame[search_frame < 0] = 0
    else:
        search_frame = frame

    buffer_pixels = 5 # ignore edges of detector which might have abnormally high counts

    if guess_location-search_width < buffer_pixels:
        search_left_edge = buffer_pixels
    else:
        search_left_edge = guess_location-search_width

    if guess_location+search_width > np.shape(search_frame)[1] - buffer_pixels:
        search_right_edge = np.shape(search_frame)[1] - buffer_pixels
    else:
        search_right_edge = guess_location+search_width

    columns_of_interest = search_frame[:,search_left_edge:search_right_edge]
    nrows,ncols = np.shape(columns_of_interest)

    trace_centre = []
    fwhm = []
    gauss_std = [] # the standard deviation in pixels measured by the Gaussian

    row_array = np.arange(nrows)

    plot_row = nrows//2

    total_errors = [] # number of errors per frame
    force_verbose = False
    delay = verbose

    if delay == -1: # we're overriding force verbose
        override_force_verbose = True
        delay = 0
        verbose = False
    elif delay == -2:
        override_force_verbose = True
    else:
        override_force_verbose = False

    log = open('reduction_output.log','a')

    for i,row in enumerate(columns_of_interest):

        if co_add_rows != 0:
            if i < co_add_rows/2:
                row = np.nanmedian(columns_of_interest[i:i+co_add_rows],axis=0)
            elif i > ncols-co_add_rows/2:
                row = np.nanmedian(columns_of_interest[i-co_add_rows:i],axis=0)
            else:
                row = np.nanmedian(columns_of_interest[i-co_add_rows//2:i+co_add_rows//2],axis=0)

        x = np.arange(ncols)[np.isfinite(row)]+search_left_edge
        row = row[np.isfinite(row)]

        if instrument == "Keck/NIRSPEC":
            # clip out negative frame from A-B
            row_residuals_1 = row - np.median(row)
            keep_index_1 = row_residuals_1 >= -5*mad(row_residuals_1)
            x = x[keep_index_1]
            row = row[keep_index_1]

            # Now use a median filter to clip out cosmic rays which are sharp positive features
            row_median_filter = MF(row,5)
            row_residuals_2 = row - row_median_filter
            keep_index_2 = ((row_residuals_2 >= -5*mad(row_residuals_2)) & (row_residuals_2 <= 5*mad(row_residuals_2)))
            x = x[keep_index_2]
            row = row[keep_index_2]

        nerrors = 0 # running count of errors
        
        centre_guess = peak_counts_location = x[np.argmax(row)]
        amplitude = np.nanmax(row)
        amplitude_offset = np.nanmin(row)

        try:
            popt1,pcov1 = optimize.curve_fit(gauss,x,row,p0=[amplitude,centre_guess,gaussian_width,amplitude_offset])

            # Make sure fitted amplitude (with offset) is not less than 25% of the guess amplitude. - note for ACAM this number was 0.3 (70%).
            # print(search_left_edge,search_right_edge,popt1[1])
            if np.fabs(popt1[0] + popt1[-1] - amplitude) < amplitude * 0.75:
                TC = popt1[1] # trace centre
                trace_centre.append(TC)
                fwhm.append(popt1[2]*2*np.sqrt(2*np.log(2))) ### save width of gaussian as FWHM after applying conversion
                gauss_std.append(abs(popt1[2]))


            else:
                print('--- Unsatisfactory fit, appending argmax at row %d for trace %d'%(i+1,star+1))
                log.write('--- Unsatisfactory fit, appending argmax at row %d for trace %d \n'%(i+1,star+1))
                TC = centre_guess
                trace_centre.append(TC)
                nerrors += 1
                gauss_std.append(0) # append 0, this will be replaced by the mean of surrounding rows in extract_trace_flux
                fwhm.append(np.nan)


        except:
            print('--- Gaussian fit failed at row %d, appending argmax for trace %d'%(i+1,star+1))
            log.write('--- Gaussian fit failed at row %d, appending argmax for trace %d \n'%(i+1,star+1))
            TC = centre_guess
            trace_centre.append(TC)
            nerrors += 1
            gauss_std.append(0) # append 0, this will be replaced by the mean of surrounding rows in extract_trace_flux
            fwhm.append(np.nan)

        total_errors.append(nerrors)

        if nerrors > 0 and i == plot_row and not override_force_verbose:
            force_verbose = True
            delay = 5 # delay in seconds

        if verbose and i == plot_row or force_verbose:
            plt.figure(figsize=(8,6))
            plt.plot(x,row,'bo',ms=5,label='data')
            plt.axvline(centre_guess,label='guessed centre',color='grey',ls='--')
            plt.axvline(TC,label='fitted centre',color='r',ls='--')
            try:
                plt.plot(x,gauss(x,*popt1),'g',label='fit')
            except:
                pass
            plt.title('Trace detection, star %d'%(star+1))
            plt.xlabel('X pixel')
            plt.ylabel('Counts at row %d'%(i+1))
            plt.legend(loc='upper left',numpoints=1)
            if delay == -2:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(delay)
                plt.close()

            force_verbose = False

    log.close()

    # use a running median to smooth the centres, with a running box of 5 data points, before fitting with a polynomial
    trace_median_filter = MF(trace_centre,5)

    if trace_poly_order > 0: # we're using the user-defined polynomial order
        poly = np.poly1d(np.polyfit(row_array,trace_median_filter,trace_poly_order))
    else: # we use a polynomial of fourth order to find the outliers before fitting the spline (which may otherwise fit the outliers)
        poly = np.poly1d(np.polyfit(row_array,trace_median_filter,4))

    fitted_positions = poly(np.arange(nrows))
    old_fitted_positions = fitted_positions.copy() # before sigma clipping
    trace_residuals = np.array(trace_centre)-poly(row_array)

    # Clip 5 sigma outliers and refit
    std_residuals = mad(trace_residuals)
    clipped_trace_idx = (np.fabs(trace_residuals) <= 5*std_residuals)

    if trace_poly_order > 0: # we're using a polynomial for our final trace positions
        if len(row_array[clipped_trace_idx]) > 2:
            fitted_function = np.poly1d(np.polyfit(row_array[clipped_trace_idx],np.array(trace_centre)[clipped_trace_idx],trace_poly_order))
        else:
            fitted_function = poly

    if trace_spline_sf > 0: # we're using a spline for our final trace positions
        spline = UnivariateSpline(row_array,np.array(trace_centre),k=3,s=trace_spline_sf)
        old_fitted_positions = spline(np.arange(nrows))
        fitted_function = UnivariateSpline(row_array[clipped_trace_idx],np.array(trace_centre)[clipped_trace_idx],k=3,s=trace_spline_sf)


    y = np.arange(nrows)

    fitted_positions = fitted_function(np.arange(nrows))

    if sum(total_errors) > 10 and not override_force_verbose:
        force_verbose = True
        delay = 5

    if verbose or force_verbose and not override_force_verbose:
        plt.figure(figsize=(8,6))
        if instrument == "Keck/NIRSPEC":
            vmin,vmax = 0,500
        else:
            vmin,vmax = np.nanpercentile(search_frame,[10,90])
        plt.imshow(search_frame,vmin=vmin,vmax=vmax,aspect="auto")
        plt.plot(trace_centre,row_array,'r',label="row-by-row centre")
        plt.plot(fitted_positions,y,'k',label="fitted centres")
        plt.legend(framealpha=1)
        plt.title('Trace detection, star %d'%(star+1))
        plt.xlim(0,np.shape(frame)[1])
        plt.ylim(0,np.shape(frame)[0])
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.tight_layout()
        if delay == -2:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        if delay >= 5:
            plt.figure(figsize=(8,6))
            plt.plot(y,fitted_positions-old_fitted_positions)
            plt.xlabel('Y pixel')
            plt.ylabel('New poly - old poly')
            plt.title('Difference between trace polynomials before and after sigma clipping, star %d'%(star+1))
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        plt.figure(figsize=(5,8))
        plt.plot(trace_centre,row_array,'bo',ms=4,label='outlier (ignored)')
        plt.plot(np.array(trace_centre)[clipped_trace_idx],row_array[clipped_trace_idx],'ro',ms=4)
        plt.plot(fitted_positions,y,'k')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.ylim(0,nrows)
        plt.title('Trace fitting, star %d'%(star+1))
        plt.legend(numpoints=1)
        # ~ plt.show()
        if delay == -2:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(row_array,np.array(trace_centre)-fitted_function(row_array),'bo',ms=4,label='outlier (ignored)')
        plt.plot(row_array[clipped_trace_idx],np.array(trace_centre)[clipped_trace_idx]-fitted_function(row_array)[clipped_trace_idx],'ro',ms=4)#,markerfacecolor='None')
        plt.title('Residuals of trace fitting, star %d'%(star+1))
        plt.ylabel('Residuals')
        plt.xlabel('X pixel')
        plt.xlim(0,nrows)
        plt.legend(numpoints=1)
        if delay == -2:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        if len(fwhm) > 0:
            plt.figure(figsize=(8,4))
            plt.plot(row_array[clipped_trace_idx],np.array(gauss_std)[clipped_trace_idx],label='Standard deviation of Gaussian')
            plt.plot(row_array[clipped_trace_idx],np.array(fwhm)[clipped_trace_idx],label="FWHM of trace")
            plt.title('Trace width, star %d'%(star+1))
            plt.ylabel('Width in pixels')
            plt.xlabel('X pixel')
            plt.xlim(0,nrows)
            plt.legend(numpoints=1)
            if delay == -2:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(delay)
                plt.close()

    if override_force_verbose and not verbose:
        delay = 0

    fwhm = np.array(fwhm)

    return fitted_positions, delay, np.median(fwhm[np.isfinite(fwhm)]), np.array(gauss_std)


def extract_trace_flux(frame,trace,aperture_width,background_offset,background_width,pre_flat_frame,poly_bg_order,am,exposure_time,verbose,star,mask,instrument,row_min,gauss_std,readout_speed,co_add_rows,rectify_frame,oversampling_factor,gain_file,readnoise_file):
    """The function used to extract the flux of a single spectral trace, using normal extraction"""

    if verbose:
        if verbose == -1:
            verbose = False

    if instrument == 'ACAM':
        D = 420.     # diameter of telescope aperutre in com, 420 = WHT
        h = 2420. # altitude of observatory (La Palma)
        # Recorded units of frame are in counts, so need to convert to electrons (photons)

        if readout_speed.lower() == 'fast':
            gain = 1.86 # electrons/count in fast readout mode with ACAM
            readnoise = 6.5 # electrons
        elif readout_speed.lower() == 'slow':
            gain = 0.92 # electrons/count
            readnoise = 3.7 # electrons
        else:
            raise NameError("readout_speed for ACAM must be defined as either 'fast' or 'slow' in extraction_input")

        buffer_pixels = 20*oversampling_factor # if using all window to estimate background
        dark_current = 4. # electrons per pixel per hour

        # Perform empirical linearity correction: -6.441 x**4 + 41.92 x**3 - 179.5 x**2 + 1.218e+04 x + 75.17 # POSSIBLY INCORRECT - ignore
        #frame = -6.441*frame**4 + 41.92*frame**3 - 179.5*frame**2 + 1.218e+04*frame + 75.17 # POSSIBLY INCORRECT - ignore
        #gain = 1.971 # Empirical from 20170316 data # POSSIBLY INCORRECT - ignore


    elif instrument == 'EFOSC':
        D = 358.
        h = 2377.
        gain = 1.38
        readnoise = 12.6
        buffer_pixels = 65*oversampling_factor # if using whole window to estimate background
        dark_current = 7. # electrons per pixel per hour


    elif instrument == 'Keck/NIRSPEC':
        D = 1000.
        h = 4000
        gain = 3.01
        readnoise = 11.56
        buffer_pixels = 0*oversampling_factor # if using whole window to estimate background
        dark_current = 2520 # electrons per pixel per hour

    elif "JWST" in instrument:
        # see https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-detectors/nirspec-detector-performance
        if gain_file is not None:
            gain = gain_file
        else:
            gain = 1 # the gain_scale step of jwst stage0 has already applied the gain correction

        if readnoise_file is not None:
            readnoise = readnoise_file
        else:
            readnoise = 0 # we're going to take the errors from the jwst .fits extension

        buffer_pixels = 0*oversampling_factor
        dark_current = 0 # dark current correction applied in jwst stage0

        # If we're analysing a JWST frame, the second array is the error array which we will use to calculate the uncertainties in our spectra
        error_frame = frame[1]
        frame = frame[0]

    else:
        # raise Warning('No readnoise or gain being loaded')
        raise NameError('Currently only set up for ACAM, EFOSC, Keck/NIRSPEC and JWST')

    if rectify_frame:
        ## Use the below to spatially rectify a frame (i.e. correct the curvature of a spectrum). Note this works but the difference in the resulting spectra is < negligible.
        frame = rectify_spatial(frame,trace)
        trace = np.ones_like(trace)*np.nanmedian(trace)
        trace = trace.astype(int)
        if verbose:
            plt.figure()
            if instrument == "Keck/NIRSPEC":
                vmin,vmax = 0,500
            else:
                vmin,vmax = np.nanpercentile(frame,[10,90])
            plt.imshow(frame,vmin=vmin,vmax=vmax,aspect="auto")
            plt.axvline(np.nanmedian(trace),color='k',label="New trace")
            plt.xlabel("X pixel")
            plt.ylabel("X pixel")
            plt.legend()
            plt.title("Post-spatial rectification")
            if verbose == -2:
                plt.show()
            if verbose > 0:
                plt.show(block=False)
                plt.pause(verbose)
                plt.close()
    else:
        trace = np.round(trace).astype(int)

    # W is variriable dependent on the line of sight and wind direction
    W3 = 2.0    # parallel to wind
    h0 = 8000. # atmospheric scale height

    if instrument == "ACAM" or instrument == "EFOSC":
        # Scintillation calculation
        scintillation = 0.09*D**(-2./3.)*(am**W3)*np.exp(-h/h0)*(2.*exposure_time)**(-1./2.)
    else:
        scintillation = 0


    # Convert from ADU to electrons (multiply by the gain)
    if "JWST" in instrument: # for JWST, this involves conerting from DN/s to e-
        frame = frame*exposure_time*gain
        error_frame = error_frame*exposure_time*gain
        pre_flat_frame = pre_flat_frame*exposure_time*gain # but we want to preserve a copy of the frame prior to flat field correction for error calculation
    else:
        frame = frame*gain
        pre_flat_frame = pre_flat_frame*gain # but we want to preserve a copy of the frame prior to flat field correction for error calculation

    nrows,ncols = np.shape(frame)
    x = np.arange(ncols)

    flux = []
    error = []
    sky_left = []
    sky_right = []
    sky_avg = []
    sky_poly = []
    raw_star_flux = []
    clipped_frame = [] # storing just the region of the frame within the background apertures

    bkg_poly_orders_used = []

    flux_base_level_left = [] # the background regions after background subtraction, should be close to zero
    flux_base_level_right = []

    max_counts = [] # line by line maximum counts in the aperture

    error_from_readnoise = []
    error_from_scintillation = []
    error_from_source = []

    if background_width == 1: # we're using whole width of the window
        left_bkg_left_hand_edge = buffer_pixels
        right_bkg_right_hand_edge = ncols - buffer_pixels

    if gauss_std is not None: # we're defining the aperture size by the measured width of the trace
        aperture_multiplication_factor = aperture_width
        aperture_width = []
        for i in range(nrows):
            if i <= 10:
                aperture_width.append(np.nanmedian(gauss_std[0:i+10])*aperture_multiplication_factor)
            elif i >= nrows-10:
                aperture_width.append(np.nanmedian(gauss_std[i-10:nrows])*aperture_multiplication_factor)
            else:
                aperture_width.append(np.nanmedian(gauss_std[i-5:i+5])*aperture_multiplication_factor)
        aperture_width = np.round(np.array(aperture_width)).astype(int)
        aperture_width_array = aperture_width.copy()

    if verbose:
        plot_frames = [50,nrows//2,nrows-50]

        # plot the science frame with apertures overlaid
        plt.figure(figsize=(8,6))

        if instrument == "Keck/NIRSPEC":
            vmin,vmax = 0,500
        else:
            vmin,vmax = np.nanpercentile(frame,[50,70])
        plt.imshow(frame,vmin=vmin,vmax=vmax,aspect="auto")

        plt.plot(trace,np.arange(nrows),'k',label="fitted centre")

        plt.plot(trace+aperture_width//2,np.arange(nrows),'r',label="extraction aperture")
        plt.plot(trace-aperture_width//2,np.arange(nrows),'r')

        plt.plot(trace-aperture_width/2-background_offset,np.arange(nrows),'r--',label="background region")
        if background_width != 1: # we're not using whole width of window
            plt.plot(trace-aperture_width//2-background_offset-background_width,np.arange(nrows),'r--')
        else:
            plt.axvline(left_bkg_left_hand_edge,color='r',ls='--')

        plt.plot(trace+aperture_width//2+background_offset,np.arange(nrows),'r--')
        if background_width != 1: # we're not using whole width of window
            plt.plot(trace+aperture_width//2+background_offset+background_width,np.arange(nrows),'r--')
        else:
            plt.axvline(right_bkg_right_hand_edge,color='r',ls='--')


        plt.xlim(0,ncols)
        plt.ylim(0,nrows)

        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')

        plt.title('Aperture locations for star %d'%(star+1))
        plt.legend(framealpha=1)
        plt.tight_layout()
        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()

    # count how many times the background overlaps with the buffer pixels
    lh_overlap = []
    rh_overlap = []

    log = open('reduction_output.log','a')

    for i, row in enumerate(frame):

        if gauss_std is not None:
            aperture_width = aperture_width_array[i]

            aperture_log = open('aperture_log.log','a')
            aperture_log.write("Trace %d, row %d, Gauss std = %f, aperture width = %f \n"%(star+1,i,gauss_std[i],aperture_width))
            aperture_log.close()

        # Using integer (left hand edge) of pixels
        aperture_left_hand_edge = trace[i]-aperture_width//2
        aperture_right_hand_edge = trace[i]+aperture_width//2

        # Check Keck/NIRSPEC's buffer pixels, since the spatial mapping can lead to bright columns at either edge of the image
        # Note: this assumes spectra have been rotated so that they're on the right-hand side of each image
        if instrument == "Keck/NIRSPEC":

            if row[0] > 1000:
                buffer_pixels_left = 8*oversampling_factor
                print("using 8 buffer pixels to the left, row %d=%f"%(i,row[0]))
            else:
                buffer_pixels_left = 0

            if row[-1] > 1000:
                buffer_pixels_right = 8*oversampling_factor
                print("using 8 buffer pixels to the right, row %d=%f"%(i,row[-1]))
            else:
                buffer_pixels_right = 0

        else:
            buffer_pixels_left = buffer_pixels_right = buffer_pixels

        if background_width == 1: # we're using whole width of the window
            # have to update background edges
            left_bkg_left_hand_edge = buffer_pixels_left
            right_bkg_right_hand_edge = ncols - buffer_pixels_right

        if aperture_left_hand_edge < buffer_pixels_left:
            aperture_left_hand_edge = buffer_pixels_left
        if aperture_right_hand_edge > ncols-buffer_pixels_right:
            aperture_right_hand_edge = ncols-buffer_pixels_right

        raw_flux = sum(pre_flat_frame[i][aperture_left_hand_edge:aperture_right_hand_edge])/oversampling_factor # In units of e-

        if "JWST" not in instrument:
            if len(np.shape(gain)) > 1:
                max_counts.append(max(row[aperture_left_hand_edge:aperture_right_hand_edge]/gain[i][aperture_left_hand_edge:aperture_right_hand_edge]/oversampling_factor)) # need to convert back to ADU, hence division by gain
            else:
                max_counts.append(max(row[aperture_left_hand_edge:aperture_right_hand_edge]/gain/oversampling_factor)) # need to convert back to ADU, hence division by gain

            if len(np.shape(readnoise)) > 1:
                error_from_readnoise.append(np.sum(readnoise[i][aperture_left_hand_edge:aperture_right_hand_edge]/oversampling_factor/raw_flux))
            else:
                error_from_readnoise.append(aperture_width*readnoise/raw_flux)

            error_from_scintillation.append(scintillation)

            if raw_flux > 0:
                error_from_source.append(np.sqrt(raw_flux)/raw_flux)
            else:
                error_from_source.append(np.nan)

        if background_width != 1: # we're not using full width of the chip

            # Using left hand edge of pixels defined as integer not np.floor
            left_bkg_left_hand_edge = trace[i]-aperture_width//2-background_offset-background_width

            # Replace left hand edge with hard edge if the chosen location falls too close to the edge of the window
            if left_bkg_left_hand_edge <= buffer_pixels_left:
                # print("WARNING! Background left hand edge overlaps buffer pixels by %d pixels"%(buffer_pixels-left_bkg_left_hand_edge))
                lh_overlap.append(buffer_pixels_left-left_bkg_left_hand_edge)
                left_bkg_left_hand_edge = buffer_pixels_left

        left_bkg_right_hand_edge = aperture_left_hand_edge - background_offset

        if background_width != 1:

            # Using left hand (integer) edge of right hand aperture
            right_bkg_right_hand_edge = trace[i]+aperture_width//2+background_offset+background_width

            # Replace right hand edge with hard edge if the chosen location falls too close to the edge of the window
            if right_bkg_right_hand_edge >= ncols - buffer_pixels_right:
                # print("WARNING! Background right hand edge overlaps buffer pixels by %d pixels"%(right_bkg_right_hand_edge-(ncols-buffer_pixels)))
                rh_overlap.append(right_bkg_right_hand_edge-(ncols-buffer_pixels_right))
                right_bkg_right_hand_edge = ncols - buffer_pixels_right

        right_bkg_left_hand_edge = aperture_right_hand_edge + background_offset

        # check background width is not too narrow (less than 10 pixels on either side)
        if background_width == 1:

            if (left_bkg_right_hand_edge - left_bkg_left_hand_edge) <  10:
                lh_overlap.append(left_bkg_right_hand_edge - left_bkg_left_hand_edge)
                # print("WARNING! Only %d pixels used for left-hand bkg estimate"%(left_bkg_right_hand_edge - left_bkg_left_hand_edge))
                # log.write("WARNING! Only %d pixels used for left-hand bkg estimate \n"%(left_bkg_right_hand_edge - left_bkg_left_hand_edge))

            if (right_bkg_right_hand_edge - right_bkg_left_hand_edge) <  10:
                rh_overlap.append(left_bkg_right_hand_edge - left_bkg_left_hand_edge)
                # print("WARNING! Only %d pixels used for right-hand bkg estimate"%(right_bkg_right_hand_edge - right_bkg_left_hand_edge))
                # log.write("WARNING! Only %d pixels used for right-hand bkg estimate \n"%(right_bkg_right_hand_edge - right_bkg_left_hand_edge))

        # Extract information about the background columns we're interested in
        bkg_cols = list(range(left_bkg_left_hand_edge,left_bkg_right_hand_edge)) + list(range(right_bkg_left_hand_edge,right_bkg_right_hand_edge))
        bkg_cols = np.array(bkg_cols)

        if mask is not None: # Applying mask to stars
            masked_regions = mask + trace[i]
            bkg_cols = np.array(sorted(set(bkg_cols).difference(masked_regions)))

            # If there are masked regions outside our region of interest lets clip them here
            masked_regions = masked_regions[((masked_regions < right_bkg_right_hand_edge) & (masked_regions > left_bkg_left_hand_edge))]

        if co_add_rows > 0:
            if i < co_add_rows/2:
                y = np.nanmedian(np.array([frame[int(r)][bkg_cols] for r in range(i,i+co_add_rows)]),axis=0)
            elif i > nrows-co_add_rows/2:
                y = np.nanmedian(np.array([frame[int(r)][bkg_cols] for r in range(i-co_add_rows,i)]),axis=0)
            else:
                y = np.nanmedian(np.array([frame[int(r)][bkg_cols] for r in range(i-int(co_add_rows/2),i+int(co_add_rows/2))]),axis=0)
        else:
            y = row[bkg_cols]

        # Clip outliers (possible cosmics)
        keep_idx = (y <= np.nanmedian(y)+3*np.nanstd(y)) & (y >= np.nanmedian(y)-3*np.nanstd(y))
        
        # Clip out-of-order background for Keck/NIRSPEC (defined as 0s)
        if instrument == "Keck/NIRSPEC":
            keep_idx_2 = ((np.isfinite(y)) & (abs(y) >= 1e-2))
            keep_idx = keep_idx * keep_idx_2

        y_keep = y[keep_idx]
        bkg_cols_keep = bkg_cols[keep_idx]

        reject_idx = ~keep_idx

        y_reject = y[reject_idx]
        bkg_cols_reject = bkg_cols[reject_idx]

        if poly_bg_order == 0: # don't perform background subtraction
            poly = np.poly1d(0)
            background_fit = poly(np.arange(left_bkg_left_hand_edge,right_bkg_right_hand_edge))

        elif poly_bg_order == -1: # use the median of the background only, don't perform a fit
            poly = np.poly1d(np.median(y_keep)) # set it up as a polynomial object for consistency with following code - this returns the median for evaulated x-arrays
            background_fit = poly(np.arange(left_bkg_left_hand_edge,right_bkg_right_hand_edge))

        elif poly_bg_order == -2: # iterate to select best background - slow
            poly_BIC_dict = {}
            poly_dict = {}

            for order in range(1,5):
                poly = np.poly1d(np.polyfit(bkg_cols_keep,y_keep,order))

                background_residuals = y_keep - poly(bkg_cols_keep)

                poly_BIC_dict[order] = BIC(poly(bkg_cols_keep),y_keep,np.sqrt(y_keep),order*len(y_keep))

                poly_dict[order] = poly(np.arange(left_bkg_left_hand_edge,right_bkg_right_hand_edge))

            # get order associated with minimum BIC
            chosen_order = min(poly_BIC_dict, key=poly_BIC_dict.get)
            bkg_poly_orders_used.append(chosen_order)

            background_fit = poly_dict[chosen_order]

        else: # polynomial with user-defined order
            poly = np.poly1d(np.polyfit(bkg_cols_keep,y_keep,poly_bg_order))
            background_fit = poly(np.arange(left_bkg_left_hand_edge,right_bkg_right_hand_edge))
            bkg_poly_orders_used.append(poly_bg_order)

        # Now saving the clipped sky regions
        if "JWST" not in instrument:
            sky_left.append(np.mean(y_keep[bkg_cols_keep <= left_bkg_right_hand_edge])/oversampling_factor)
            sky_right.append(np.mean(y_keep[bkg_cols_keep >= right_bkg_left_hand_edge])/oversampling_factor)

        sky_avg.append(np.mean(y_keep)/oversampling_factor)
        sky_poly.append(background_fit)
        # sky_poly_full.append(background_fit)

        if verbose and i in plot_frames:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111)
            ax.plot(x[left_bkg_left_hand_edge:right_bkg_right_hand_edge],row[left_bkg_left_hand_edge:right_bkg_right_hand_edge])
            ax.plot(bkg_cols_reject,y_reject,'kx',label='outlier (ignored)') # Ignored pixels

            if mask is not None:
                ax.plot(masked_regions,row[masked_regions],'rx',label='masked points') # masked regions

            ax.plot(x[left_bkg_left_hand_edge:right_bkg_right_hand_edge],background_fit,'g',label="background fit") # Background fit

            ax.axvline(trace[i],color='k',label="trace centre")

            ax.axvline(aperture_left_hand_edge,color='r',label="extraction aperture")
            ax.axvline(aperture_right_hand_edge,color='r')

            ax.axvline(left_bkg_left_hand_edge,color='r',ls='--',label="background regions")
            ax.axvline(left_bkg_right_hand_edge,color='r',ls='--')

            ax.axvline(right_bkg_left_hand_edge,color='r',ls='--')
            ax.axvline(right_bkg_right_hand_edge,color='r',ls='--')

            ax.set_title('Aperture locations before background subtraction, with background fit, star %d'%(star+1))
            ax.set_xlabel('X pixel')
            ax.set_ylabel('Counts at row %d'%(i+1))
            ax.legend(loc='upper left',numpoints=1,framealpha=1)

            ylim_full = ax.get_ylim()

            if verbose == -2:
                plt.show()
            if verbose > 0:
                plt.show(block=False)
                plt.pause(verbose)
                plt.close()

        background_subtracted = row[left_bkg_left_hand_edge:right_bkg_right_hand_edge] - background_fit
        clipped_frame.append(row[left_bkg_left_hand_edge:right_bkg_right_hand_edge])

        flux.append(sum(background_subtracted[aperture_left_hand_edge-left_bkg_left_hand_edge:aperture_right_hand_edge-left_bkg_left_hand_edge])/oversampling_factor)
        if "JWST" not in instrument:
            flux_base_level_left.append(np.median(background_subtracted[:left_bkg_right_hand_edge-left_bkg_left_hand_edge])/oversampling_factor)
            flux_base_level_right.append(np.median(background_subtracted[right_bkg_left_hand_edge-left_bkg_left_hand_edge:])/oversampling_factor)

        # Error calculation used http://www.ucolick.org/~bolte/AY257/s_n.pdf as a reference
        # Note: this neglects the error in the flat field
        if sum(row[aperture_left_hand_edge:aperture_right_hand_edge]) <= 0:
            error.append(np.nan)
        else:
            if instrument == "Keck/NIRSPEC": # we include dark current but not scintillation
                error.append(np.sqrt(sum(row[aperture_left_hand_edge:aperture_right_hand_edge])/oversampling_factor + (aperture_width/oversampling_factor)*readnoise**2 + dark_current*(aperture_width/oversampling_factor)*exposure_time/3600.)) # raw_flux here takes into account noise from the source and noise from the sky, since this is before background subtraction

            elif "JWST" in instrument: # we're using the error frame
                error.append(np.sqrt(np.sum((error_frame[i][aperture_left_hand_edge:aperture_right_hand_edge]/oversampling_factor)**2)))#/(oversampling_factor**2)))

            else: # I'm assuming we're looking at ACAM/EFOSC data and we're including scintillation but not dark current
                error.append(np.sqrt(sum(row[aperture_left_hand_edge:aperture_right_hand_edge])/oversampling_factor + (aperture_width/oversampling_factor)*readnoise**2 + scintillation**2))

        if "JWST" not in instrument:
            raw_star_flux.append(sum(row[aperture_left_hand_edge:aperture_right_hand_edge])/oversampling_factor)

        if verbose and i in plot_frames:

            plt.figure(figsize=(8,6))

            plt.plot(background_subtracted)
            plt.plot(bkg_cols_reject-left_bkg_left_hand_edge,y_reject-poly(bkg_cols_reject),'kx',label='outlier (ignored)') # Ignored pixels
            if mask is not None:
                plt.plot(masked_regions-left_bkg_left_hand_edge,background_subtracted[masked_regions-left_bkg_left_hand_edge],'rx',label='masked points') # masked regions

            plt.axvline(trace[i]-left_bkg_left_hand_edge,color='k',label="trace centre")
            plt.axvline(aperture_left_hand_edge-left_bkg_left_hand_edge,color='r',label="extraction aperture")
            plt.axvline(aperture_right_hand_edge-left_bkg_left_hand_edge,color='r')
            plt.axvline(left_bkg_right_hand_edge-left_bkg_left_hand_edge,color='r',ls='--',label="background regions")
            plt.axvline(right_bkg_left_hand_edge-left_bkg_left_hand_edge,color='r',ls='--')
            plt.axhline(0,color='k')
            plt.title('Aperture locations after background subtraction, star %d'%(star+1))
            plt.xlabel('X pixel')
            plt.ylabel('Background subtracted counts at row %d'%(i+1))
            plt.ylim(-200,200)
            plt.legend(loc='upper left',numpoints=1,framealpha=1)
            if verbose == -2:
                plt.show()
            if verbose > 0:
                plt.show(block=False)
                plt.pause(verbose)
                plt.close()

    if verbose:

        # background subtracted image
        clipped_frame = np.array(clipped_frame)
        plt.figure(figsize=(12,8))
        plt.subplot(121)

        if instrument == "Keck/NIRSPEC":
            vmin,vmax = 0,500
        else:
            vmin,vmax = np.nanpercentile(clipped_frame,[10,50])
        plt.imshow(clipped_frame,vmin=vmin,vmax=vmax,aspect="auto")
        plt.title("Before background subtraction")
        plt.xlabel("Pixel column")
        plt.ylabel("Pixel row")

        plt.subplot(122)

        if instrument == "Keck/NIRSPEC":
            vmin,vmax = -500,1000
        else:
            vmin,vmax = np.nanpercentile(clipped_frame-np.array(sky_poly),[10,50])
        plt.imshow(clipped_frame-np.array(sky_poly),vmin=vmin,vmax=vmax,aspect="auto")
        plt.title("After background subtraction")
        plt.xlabel("Pixel column")
        # ~ ax2.set_ylabel("Pixel row")

        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()


        plt.figure(figsize=(8,6))
        plt.plot(flux)
        plt.show(block=False)
        plt.ylabel('Integrated counts')
        plt.xlabel('X pixel')
        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()

    # Now print how many times the background aperture overlapped with the buffer pixels
    if len(lh_overlap) != 0:
        print("For trace %d..."%(star+1))
        lh_counted = Counter(lh_overlap)
        for k in sorted(lh_counted.keys()):
            print("Left hand edge overlaps buffer pixels by %d pixels for %d rows"%(k,lh_counted[k]))
            log.write("Left hand edge overlaps buffer pixels by %d pixels for %d rows \n"%(k,lh_counted[k]))

    if len(rh_overlap) != 0:
        print("For trace %d..."%(star+1))
        rh_counted = Counter(rh_overlap)
        for k in sorted(rh_counted.keys()):
            print("Right hand edge overlaps buffer pixels by %d pixels for %d rows"%(k,rh_counted[k]))
            log.write("Right hand edge overlaps buffer pixels by %d pixels for %d rows"%(k,rh_counted[k]))

    log.close()

    if "JWST" in instrument: # only return the key arrays since the data files are so large and consume too much memory
        return np.array(flux),np.array(error),np.array(sky_avg)
    else:
        return np.array(flux),np.array(error),np.array(sky_avg),np.array(sky_left),np.array(sky_right),np.array(flux_base_level_left),np.array(flux_base_level_right),np.array(max_counts),np.array(error_from_readnoise),\
               np.array(error_from_scintillation),np.array(error_from_source),np.array(bkg_poly_orders_used),np.array(raw_star_flux)

def extract_all_frame_fluxes(science_list,master_bias,master_flat,trace_dict,window_dict,extraction_dict,verbose=False,bad_pixel_mask=None,cosmic_pixel_mask=None,oversampling_factor=1,gain_file=None,readnoise_file=None):

    """The funtion that loops through all science frames,finding the trace locations, extracting the flux, and saving the final
    output."""

    # if verbose:
    #     if verbose == -1:
    #         verbose = False

    start_time = time.time()

    if master_bias is not None:
        master_bias = fits.open(master_bias)[0].data

    if master_flat is not None:
        master_flat = fits.open(master_flat)[0].data

    if bad_pixel_mask is not None:
        try:
            bad_pixel_mask = np.atleast_2d(fits.open(bad_pixel_mask)[0].data.astype(bool))
        except:
            bad_pixel_mask = np.atleast_2d(pickle.load(open(bad_pixel_mask,"rb")).astype(bool))
        use_mask = True

    if cosmic_pixel_mask is not None:
        cosmic_pixel_mask = pickle.load(open(cosmic_pixel_mask,"rb"))
        use_mask = True

    if bad_pixel_mask is None and cosmic_pixel_mask is None:
        use_mask = False

    rotate_frame = window_dict['rotate_frame']
    row_min = window_dict['row_min']
    row_max = window_dict['row_max']

    if gain_file is not None:
        gain_file = fits.getdata(gain_file)
        if rotate_frame:
            gain_file = np.flip(gain_file.T,axis=1)
        if oversampling_factor > 1:
            gain_file = resample_frame(gain_file,oversampling_factor,verbose=False)
        gain_file = gain_file[row_min:row_max]
        print("Mean gain = %.3f"%(gain_file.mean()))


    if readnoise_file is not None:
        readnoise_file = fits.getdata(readnoise_file)
        if rotate_frame:
            readnoise_file = np.flip(readnoise_file.T,axis=1)
        if oversampling_factor > 1:
            readnoise_file = resample_frame(readnoise_file,oversampling_factor,verbose=False)
        readnoise_file = readnoise_file[row_min:row_max]
        print("Mean readnoise = %.3f"%(readnoise_file.mean()))

    obs_time_array = []
    airmass = []
    exposure_time_array = []

    guess_location = trace_dict['guess_locations']
    search_width = trace_dict['search_width']
    gaussian_width = trace_dict['gaussian_width']
    trace_poly_order = trace_dict['trace_poly_order']
    trace_spline_sf = trace_dict['trace_spline_sf']
    if trace_spline_sf > 0 and trace_poly_order > 0:
        raise ValueError('Cannot use both a spline and polynomial fit to the trace, one of these must be set to zero in extraction input.')

    co_add_rows = trace_dict['co_add_rows']

    instrument = window_dict['instrument']
    readout_speed = window_dict['readout_speed']
    nwindows = window_dict['nwindows']


    aperture_width = extraction_dict['aperture_width']
    background_offset = extraction_dict['background_offset']
    background_width = extraction_dict['background_width']
    poly_bg_order = extraction_dict['poly_bg_order']
    rectify_frame = extraction_dict['rectify_frame']

    nstars = extraction_dict['nstars']
    masks = extraction_dict['masks']
    try:
        NIRSPEC_order = extraction_dict["NIRSPEC_order"]
    except:
        NIRSPEC_order = None
    use_lacosmic = extraction_dict['use_lacosmic']
    ACAM_linearity_correction = extraction_dict['ACAM_linearity_correction']
    gaussian_defined_aperture = extraction_dict['gaussian_defined_aperture']
    if gaussian_defined_aperture:
        aperture_log = open('aperture_log.log','w')
        aperture_log.close()

    stellar_fluxes = []
    stellar_errors = []
    sky_lefts = []
    sky_rights = []
    sky_avgs = []
    sky_polys = []
    base_lefts = []
    base_rights = []
    traces = []
    FWHM = []
    MAX_COUNTS = []
    raw_stellar_fluxes = []
    cosmic_masked_pixels = []

    scintillation_error = []
    readnoise_error = []
    poisson_noise = []

    background_poly_order_used = []

    log = open('reduction_output.log','w')
    log.close()

    if "JWST" in instrument:
        fits_files = [fits.open(s,memmap=False) for s in science_list]
        nints = np.cumsum([f["SCI"].data.shape[0] for f in fits_files])
        total_nints = nints[-1]
        science_list = ["Integration %s"%i for i in range(total_nints)]

    for i,f in enumerate(science_list):

        print(f, '[%.1f%% complete, %d mins since start]'%((i+1)*100./len(science_list),(time.time()-start_time)/60))
        log = open('reduction_output.log','a')
        log.write('%s [%.1f%% complete, %d mins since start] \n'%(f,(i+1)*100./len(science_list),(time.time()-start_time)/60))
        log.close()

        if gaussian_defined_aperture:
            aperture_log = open('aperture_log.log','a')
            aperture_log.write('%s \n'%(f))
            aperture_log.close()

        if "JWST" not in instrument:
            fits_file = fits.open(f,memmap=False)
        else:
            jwst_fits_counter = np.digitize(i,nints)
            if jwst_fits_counter > 0:
                jwst_index_counter = i-nints[jwst_fits_counter]
            else:
                jwst_index_counter = i
            fits_file = fits_files[jwst_fits_counter]

        for window in range(1,nwindows+1):

            if master_bias is None and "JWST" not in instrument:
                if instrument == 'ACAM':
                    master_bias = np.zeros_like(fits_file[window].data)
                else:
                    master_bias = np.zeros_like(fits_file[window-1].data)

            if nwindows > 1:
                bias = master_bias[window-1]
            else:
                bias = master_bias

            if nwindows > 1 and master_flat is not None:
                flat = master_flat[window-1]
            else:
                flat = master_flat

            if window == 1:

                if instrument == "ACAM":
                    obs_time_array.append(fits_file[0].header['MJD-OBS'])
                    exposure_time_array.append(fits_file[0].header['EXPTIME'])
                    am = fits_file[0].header['AIRMASS']
                    airmass.append(am)
                    
                elif instrument == "EFOSC":
                    obs_time_array.append(fits_file[0].header['MJD-OBS'])
                    exposure_time_array.append(fits_file[0].header['EXPTIME'])
                    am = fits_file[0].header['HIERARCH ESO TEL AIRM START']
                    airmass.append(am)
                    
                elif "JWST" in instrument:
                    obs_time_array.append(fits_file["INT_TIMES"].data["int_mid_BJD_TDB"][jwst_index_counter])
                    exposure_time_array.append(fits_file[0].header["EFFINTTM"])
                    am = 0
                    airmass.append(0)
                    
                elif instrument == "Keck/NIRSPEC":
                    exposure_time = fits_file[0].header["ITIME"] / 1e3
                    exposure_time_array.append(exposure_time)
                    obs_date = fits_file[0].header["DATE-OBS"]
                    obs_start = Time(obs_date + "T" + fits_file[0].header["UTSTART"])
                    obs_mid = obs_start + TimeDelta(exposure_time/2,format='sec')
                    obs_time_array.append(obs_mid.mjd)
                    am = fits_file[0].header["AIRMASS"]
                    airmass.append(am)
                    m1temp = fits_file[0].header["SPEC1TMP"]
                    try: # saving m1temp to text file to save propagating through as a numpy array
                        new_tab = open("m1temp.txt","a")
                    except:
                        new_tab = open("m1temp.txt","w")
                    new_tab.write("%f \n"%(m1temp))
                    new_tab.close()
                    
                else:
                    obs_time_array.append(0)
                    exposure_time_array.append(0)
                    am = 0
                    airmass.append(0)

            if instrument == 'ACAM':
                frame = fits_file[window].data - bias
            elif "JWST" in instrument: # we're not performing a bias correction as this is done in jwst stage0
                frame = np.array([fits_file["SCI"].data[jwst_index_counter],fits_file["ERR"].data[jwst_index_counter]])
            else:
                frame = fits_file[window-1].data - bias

            uncorrected_frame = frame.astype(float)

            if master_flat is not None and "JWST" not in instrument: # this doesn't apply for jwst data as this is done in jwst stage0
                if instrument == 'ACAM':
                    frame = (fits_file[window].data - bias) / flat
                else:
                    frame = (fits_file[window-1].data - bias) / flat

            # replace inf with nan
            if "JWST" in instrument:
                if np.any(~np.isfinite(frame[0])):
                    frame[0][~np.isfinite(frame[0])] = np.nan
            else:
                if np.any(~np.isfinite(frame)):
                    frame[~np.isfinite(frame)] = np.nan


            if use_mask:
                if bad_pixel_mask is not None:
                    if len(bad_pixel_mask.shape) > 2:
                        bad_pixel_mask = bad_pixel_mask[i]
                if bad_pixel_mask is not None and cosmic_pixel_mask is None:
                    pixel_mask = bad_pixel_mask
                if bad_pixel_mask is None and cosmic_pixel_mask is not None:
                    pixel_mask = cosmic_pixel_mask[i]
                if bad_pixel_mask is not None and cosmic_pixel_mask is not None:
                    pixel_mask = bad_pixel_mask + cosmic_pixel_mask[i]

                if verbose != -1 and verbose != 0 and i == 0 or verbose != -1 and verbose != 0 and cosmic_pixel_mask is not None:
                    plt.figure()
                    plt.imshow(pixel_mask, interpolation='none',aspect="auto")
                    plt.title("Pixel mask, frame %d"%i)
                    plt.ylabel("Pixel column")
                    plt.xlabel("Pixel row")
                    if verbose == -2:
                        plt.show()
                    if verbose > 0:
                        plt.show(block=False)
                        plt.pause(verbose)
                        plt.close()

                original_frame = frame.copy()
                if "JWST" in instrument:
                    frame = np.array([interp_bad_pixels(frame[0],pixel_mask),interp_bad_pixels(frame[1],pixel_mask)])
                else:
                    frame = interp_bad_pixels(frame,pixel_mask)

                if verbose != -1 and verbose != 0 and i == 0 or verbose != -1 and verbose != 0 and cosmic_pixel_mask is not None:
                    plt.figure()

                    plt.subplot(211)
                    if "JWST" in instrument:
                        vmin,vmax = np.nanpercentile(original_frame[0],[10,70])
                        plt.imshow(original_frame[0],vmin=vmin,vmax=vmax,aspect="auto")
                    # elif instrument == "Keck/NIRSPEC":
                    #     vmin,vmax = 0,500
                    else:
                        vmin,vmax = np.nanpercentile(original_frame,[10,70])
                        plt.imshow(original_frame,vmin=vmin,vmax=vmax,aspect="auto")
                    plt.title("Pre-pixel-masked frame")
                    plt.xticks(visible=False)
                    # ~ plt.xlabel("Pixel column")
                    plt.ylabel("Pixel row")


                    plt.subplot(212)
                    if "JWST" in instrument:
                        vmin,vmax = np.nanpercentile(frame[0],[10,70])
                        plt.imshow(frame[0],vmin=vmin,vmax=vmax,aspect="auto")
                    # elif instrument == "Keck/NIRSPEC":
                    #     vmin,vmax = 0,500
                    else:
                        vmin,vmax = np.nanpercentile(frame,[10,70])
                        plt.imshow(frame,vmin=vmin,vmax=vmax,aspect="auto")

                    plt.title("Post-pixel-masked frame")
                    plt.xlabel("Pixel column")
                    plt.ylabel("Pixel row")

                    if verbose == -2:
                        plt.show()
                    if verbose > 0:
                        plt.show(block=False)
                        plt.pause(verbose)
                        plt.close()

            else:
                pixel_mask=None


            if NIRSPEC_order is not None:
                if i == 0 and verbose:
                    v = verbose
                else:
                    v = False

                frame = KO.mask_NIRSPEC_data(frame,NIRSPEC_order,v)


            #if use_lacosmic and instrument == 'ACAM':
                #frame,_ = lacosmic.lacosmic(frame,0.5,15,15,effective_gain=1.9,readnoise=7)
            if use_lacosmic and instrument == "Keck/NIRSPEC" and cosmic_pixel_mask is None:
                cosmic_search_frame = copy.deepcopy(frame)
                cosmic_search_frame[~np.isfinite(cosmic_search_frame)] = 0
                cosmic_search_frame[cosmic_search_frame < 0] = 0

                # frame[~np.isfinite(frame)] = 0
                # frame[frame < 0] = 0
                cosmic_pixels,_ = astroscrappy.detect_cosmics(cosmic_search_frame[row_min:row_max], gain=3.01,readnoise=11.56, \
                                                          satlevel=np.inf, inmask=pixel_mask[row_min:row_max], sepmed=False, \
                                                          cleantype='medmask', fsmode='median',verbose=True,sigclip=5,objlim=10,niter=8)
                # frame[frame == 0] = np.nan
                frame[row_min:row_max] = interp_bad_pixels(frame[row_min:row_max],cosmic_pixels)

                if verbose != -1 and verbose != 0:
                    plt.figure(figsize=(4,12))
                    plt.imshow(cosmic_pixels,aspect="auto")
                    plt.title("Lacosmic-flagged cosmic pixels")
                    if verbose == -2:
                        plt.show()
                    if verbose > 0:
                        plt.show(block=False)
                        plt.pause(verbose)
                        plt.close()

                cosmic_masked_pixels.append(cosmic_pixels)

            if rotate_frame:
                if "JWST" in instrument:
                    frame = np.array([np.flip(frame[0].T,axis=1),np.flip(frame[1].T,axis=1)])
                    uncorrected_frame = np.array([np.flip(uncorrected_frame[0].T,axis=1),np.flip(uncorrected_frame[1].T,axis=1)])
                else:
                    frame = np.flip(frame.T,axis=1)
                    uncorrected_frame = np.flip(uncorrected_frame.T,axis=1)

            if "JWST" in instrument:
                frame = np.array([frame[0][row_min:row_max].astype(float),frame[1][row_min:row_max].astype(float)])
                uncorrected_frame = np.array([uncorrected_frame[0][row_min:row_max],uncorrected_frame[1][row_min:row_max]])
            else:
                frame = frame[row_min:row_max].astype(float)
                uncorrected_frame = uncorrected_frame[row_min:row_max]

            if oversampling_factor > 1:
                # nrows,ncols = frame.shape
                if "JWST" in instrument:
                    frame = np.array([resample_frame(frame[0],oversampling_factor,verbose=verbose),resample_frame(frame[1],oversampling_factor)])
                    uncorrected_frame = np.array([resample_frame(uncorrected_frame[0],oversampling_factor),resample_frame(uncorrected_frame[1],oversampling_factor)])
                else:
                    frame = resample_frame(frame,oversampling_factor,verbose=verbose)
                    uncorrected_frame = resample_frame(uncorrected_frame,oversampling_factor)
                # oversampling_factor = ((ncols-1)*oversampling+1)/ncols


            if ACAM_linearity_correction and instrument == 'ACAM':
                frame = ((-0.007/65000)*frame + 1)*frame # from ACAM webpages

            if nwindows == 1:
                loop_range = range(nstars)
            else:
                loop_range = range(1)

            for star_number in loop_range:

                if nwindows > 1:
                    star_number += window - 1

                if search_width[star_number] > 0:
                    trace, force_verbose, fwhm, gauss_std = find_spectral_trace(frame,guess_location[star_number],search_width[star_number],gaussian_width,trace_poly_order,trace_spline_sf,star_number,verbose,co_add_rows,instrument)
                else:
                    trace = np.ones(row_max-row_min)*guess_location[star_number]
                    fwhm = gauss_std = np.ones(row_max-row_min)
                    force_verbose = verbose

                if gaussian_defined_aperture:
                    
                    # Smooth the FWHMs with a quadratic polynomial
                    gauss_std_poly = np.poly1d(np.polyfit(np.arange(0,row_max-row_min),gauss_std,trace_poly_order))
                    gauss_std_smooth = gauss_std_poly(np.arange(0,row_max-row_min))
                    
                    # refit with outliers clipped
                    gauss_std_residuals = gauss_std - gauss_std_poly(np.arange(0,row_max-row_min)) 
                    gauss_std_keep_idx = abs(gauss_std_residuals) <= 4*np.std(gauss_std_residuals)
                    
                    gauss_std_poly = np.poly1d(np.polyfit(np.arange(0,row_max-row_min)[gauss_std_keep_idx],gauss_std[gauss_std_keep_idx],trace_poly_order))
                    gauss_std_smooth = gauss_std_poly(np.arange(0,row_max-row_min))
                    
                    if verbose != -1 and verbose != 0:
                        plt.figure()
                        plt.plot(np.arange(row_min,row_max),gauss_std,label="Std dev of trace")
                        plt.plot(np.arange(row_min,row_max)[~gauss_std_keep_idx],gauss_std[~gauss_std_keep_idx],"rx",label="Clipped outlier")
                        plt.plot(np.arange(row_min,row_max),gauss_std_smooth,label="Smoothed with polynomial (order = %d)"%trace_poly_order)
                        plt.xlabel("Pixel number")
                        plt.ylabel("Standard deviation (pixels)")
                        plt.title("Gaussian-defined aperture widths")
                        if verbose == -2:
                            plt.show()
                        if verbose > 0:
                            plt.show(block=False)
                            plt.pause(verbose)
                            plt.close()
                            
                    trace_std = gauss_std_smooth*2*np.sqrt(2*np.log(2))

                else:
                    trace_std = None

                if "JWST" in instrument: # only return the key arrays since the data files are so large and consume too much memory
                    flux,error,sky_avg = extract_trace_flux(frame,trace,aperture_width[star_number],background_offset[star_number],\
                                                                                                    background_width[star_number],uncorrected_frame[0],poly_bg_order[star_number],am,\
                                                                                                    exposure_time,force_verbose,star_number,masks['mask%d'%(star_number+1)],instrument,row_min,trace_std,readout_speed,co_add_rows,rectify_frame,oversampling_factor,\
                                                                                                    gain_file,readnoise_file)

                else:
                    flux,error,sky_avg,sky_left,sky_right,base_left,base_right,max_counts,rn_error,scin_error,pois_error,bkg_poly_order,raw_star_flux = extract_trace_flux(frame,trace,aperture_width[star_number],background_offset[star_number],\
                                                                                background_width[star_number],uncorrected_frame,poly_bg_order[star_number],am,\
                                                                                exposure_time,force_verbose,star_number,masks['mask%d'%(star_number+1)],instrument,row_min,trace_std,readout_speed,co_add_rows,rectify_frame,oversampling_factor,\
                                                                                gain_file,readnoise_file)

                    plt.close("all")

                    sky_lefts.append(sky_left)
                    sky_rights.append(sky_right)
                    # sky_polys.append(sky_poly)
                    base_lefts.append(base_left)
                    base_rights.append(base_right)
                    MAX_COUNTS.append(max_counts)
                    scintillation_error.append(scin_error)
                    readnoise_error.append(rn_error)
                    poisson_noise.append(pois_error)
                    background_poly_order_used.append(bkg_poly_order)
                    raw_stellar_fluxes.append(raw_star_flux)

                stellar_fluxes.append(flux)
                stellar_errors.append(error)
                sky_avgs.append(sky_avg)
                traces.append(trace)
                FWHM.append(fwhm)

        if "JWST" not in instrument:
            fits_file.close()

    try:
        os.mkdir("pickled_objects")
    except:
        pass

    for i in range(nstars):
        pickle.dump(np.array(stellar_fluxes[i::nstars]),open('pickled_objects/star%d_flux.pickle'%(i+1),'wb'))
        pickle.dump(np.array(stellar_errors[i::nstars]),open('pickled_objects/star%d_error.pickle'%(i+1),'wb'))
        pickle.dump(np.array(traces[i::nstars]),open('pickled_objects/x_positions_%d.pickle'%(i+1),'wb'))
        pickle.dump(np.array(FWHM[i::nstars]),open('pickled_objects/fwhm_%d.pickle'%(i+1),'wb'))
        pickle.dump(np.array(sky_avgs[i::nstars]),open('pickled_objects/background_avg_star%d.pickle'%(i+1),'wb'))

        if "JWST" not in instrument:
            pickle.dump(np.array(sky_lefts[i::nstars]),open('pickled_objects/sky_left_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(sky_rights[i::nstars]),open('pickled_objects/sky_right_star%d.pickle'%(i+1),'wb'))
            # pickle.dump(np.array(sky_polys[i::nstars]),open('pickled_objects/sky_poly_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(base_lefts[i::nstars]),open('pickled_objects/flux_base_level_left_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(base_rights[i::nstars]),open('pickled_objects/flux_base_level_right_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(MAX_COUNTS[i::nstars]),open('pickled_objects/max_counts_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(scintillation_error[i::nstars]),open('pickled_objects/scintillation_error_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(readnoise_error[i::nstars]),open('pickled_objects/readnoise_error_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(poisson_noise[i::nstars]),open('pickled_objects/poisson_noise_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(raw_stellar_fluxes[i::nstars]),open('pickled_objects/star%d_raw_flux.pickle'%(i+1),'wb'))

            if 0 in poly_bg_order:
                pickle.dump(np.array(background_poly_order_used[i::nstars]),open('background_poly_orders_used_%d.pickle'%(i+1),'wb'))

    if "JWST" not in instrument:
        pickle.dump(np.array(airmass),open('pickled_objects/airmass.pickle','wb'))
        pickle.dump(np.array(exposure_time_array),open('pickled_objects/exposure_times.pickle','wb'))

    pickle.dump(np.array(obs_time_array),open('pickled_objects/obs_time_array.pickle','wb'))

    if use_lacosmic and instrument == "Keck/NIRSPEC" and cosmic_pixel_mask is None:
        pickle.dump(np.array(cosmic_masked_pixels),open("pickled_objects/cosmic_masked_pixels.pickle","wb"))
        
    if instrument == "Keck/NIRSPEC":
        os.rename("m1temp.txt", "pickled_objects/m1temp.txt")

    return np.array(stellar_fluxes),np.array(stellar_errors),np.array(obs_time_array)


def generate_wl_curve(stellar_fluxes,stellar_errors,time,nstars,overwrite=True):

    """Generate the white light curve and output to table and figure"""

    star1 = stellar_fluxes[::nstars]
    error1 = stellar_errors[::nstars]

    if nstars > 1:
        star2 = stellar_fluxes[1::nstars]
        error2 = stellar_errors[1::nstars]

        ratio = np.sum(star1,axis=1)/np.sum(star2,axis=1)
        err_ratio = np.sqrt((np.sqrt(np.sum(error1**2,axis=1))/np.sum(star1,axis=1))**2 + (np.sqrt(np.sum(error2**2,axis=1))/np.sum(star2,axis=1))**2)*ratio

    else:
        ratio = np.mean(star1,axis=1)
        err_ratio = np.mean(error1,axis=1)

    if overwrite or not os.path.isfile('white_light.txt'):
        tab = open('white_light.txt','w')
        old_time = None
    else:
        tab = open('white_light.txt','a')
        old_time,old_ratio,old_err_ratio = np.loadtxt('white_light.txt',unpack=True)

    for i in range(len(ratio)):
        tab.write("%f %f %f \n"%(time[i],ratio[i],err_ratio[i]))

    tab.close()

    plt.figure(figsize=(8,6))
    if old_time is None:
        plt.plot(time-int(time[0]),ratio,'k.')
        plt.xlabel('Time (MJD/BJD - %d)'%int(time[0]))
    else:
        plt.plot(np.hstack((old_time,time))-int(old_time[0]),np.hstack((old_ratio,ratio)),'k.')
        plt.xlabel('Time (MJD/BJD - %d)'%int(old_time[0]))
    plt.ylabel('Flux')
    plt.savefig('white_light_curve.pdf')
    plt.close()

    try:
        os.mkdir("./initial_WL_fit")
    except:
        pass

    pickle.dump(time-int(time[0]),open("./initial_WL_fit/initial_WL_time.pickle","wb"))
    pickle.dump(ratio,open("./initial_WL_fit/initial_WL_flux.pickle","wb"))
    pickle.dump(err_ratio,open("./initial_WL_fit/initial_WL_err.pickle","wb"))

    return


def main(input_file='extraction_input.txt'):
    input_dict = parseInput(input_file)

    oversampling_factor = input_dict["oversampling_factor"]
    if oversampling_factor is None:
        oversampling_factor = 1
    else:
        oversampling_factor = int(oversampling_factor)


    # order mask for Keck/NIRSPEC data
    if input_dict['instrument'] == 'Keck/NIRSPEC':
        
        NIRSPEC_order = input_dict["NIRSPEC_order"]
        trace_guess_locations,trace_search_widths = KO.get_guess_locations(NIRSPEC_order)
        nstars = 1
        trace_guess_locations *= oversampling_factor
        trace_search_widths *= oversampling_factor

    else:

        NIRSPEC_order = None

        trace_guess_locations = [int(x)*oversampling_factor for x in input_dict['trace_guess_locations'].split(",")]
        nstars = len(trace_guess_locations)

        # Update, added ability to have different extraction parameters for each star
        trace_search_widths = [int(x)*oversampling_factor for x in input_dict['trace_search_width'].split(",")]

        if len(trace_search_widths) == 1:
            trace_search_widths = trace_search_widths*nstars

    polybg_orders = [int(x) for x in input_dict['poly_bg_order'].split(",")]
    if len(polybg_orders) == 1:
        polybg_orders = polybg_orders*nstars


    gaussian_defined_aperture = bool(int(input_dict['gaussian_defined_aperture']))

    if gaussian_defined_aperture:
        aperture_widths = [int(x) for x in input_dict['aperture_width'].split(",")]
    else:
        aperture_widths = [int(x)*oversampling_factor for x in input_dict['aperture_width'].split(",")]
    if len(aperture_widths) == 1:
        aperture_widths = aperture_widths*nstars


    background_offsets = [int(x)*oversampling_factor for x in input_dict['background_offset'].split(",")]
    if len(background_offsets) == 1:
        background_offsets = background_offsets*nstars

    background_widths = []
    for x in input_dict['background_width'].split(","):
        if int(x) > 1:
            background_widths.append(int(x)*oversampling_factor)
        else:
            background_widths.append(1)
    if len(background_widths) == 1:
        background_widths = background_widths*nstars


    mask_input = input_dict['masks']
    mask_width = input_dict['mask_width']
    if mask_input is not None:
        all_masks = mask_input.split(";")
        masked_region_list = []
        for i in range(nstars):
            masked_region_list.append([int(x)*oversampling_factor for x in all_masks[i].split(",") if x != ''])

        masks = create_masks(masked_region_list,nstars,int(mask_width)*oversampling_factor)

    else:
        masks = {}
        for i in range(1,nstars+1):
            masks['mask%d'%i] = None


    if input_dict["instrument"] == "ACAM":
        ACAM_linearity_correction = bool(int(input_dict['ACAM_linearity_correction']))
    else:
        ACAM_linearity_correction = False

    overwrite = bool(int(input_dict['overwrite']))


    trace_location_dict = {'guess_locations':trace_guess_locations,'search_width':trace_search_widths,\
                            'gaussian_width':int(input_dict['trace_gaussian_width'])*oversampling_factor,'trace_poly_order':int(input_dict['trace_poly_order']),\
                            'trace_spline_sf':float(input_dict['trace_spline_sf']),'co_add_rows':int(input_dict['co_add_rows'])}

    window_info_dict = {'instrument':input_dict['instrument'],'nwindows':int(input_dict['nwindows']),'row_min':int(input_dict['row_min']),'row_max':int(input_dict['row_max']),\
                        'readout_speed':input_dict['readout_speed'],"rotate_frame":bool(int(input_dict["rotate_frame"]))}

    extraction_params_dict = {'aperture_width':aperture_widths,'background_offset':background_offsets,\
                               'background_width':background_widths,'poly_bg_order':polybg_orders,\
                               'nstars':nstars,'masks':masks,'ACAM_linearity_correction':ACAM_linearity_correction,'gaussian_defined_aperture':gaussian_defined_aperture,\
                               "NIRSPEC_order":NIRSPEC_order,'use_lacosmic':bool(int(input_dict['use_lacosmic'])),"rectify_frame":bool(int(input_dict["rectify_data"]))}

    v = int(input_dict['verbose'])

    try:
        science_files = np.atleast_1d(np.loadtxt(input_dict['science_list'],str))

    except: # loading in a reference image
        import glob
        if input_dict['instrument'] == 'ACAM':
            file_names = sorted(glob.glob("r*.fit"))
        if input_dict['instrument'] == 'EFOSC':
            file_names = sorted(glob.glob("EFOSC_Spectrum*.fits"))
        ref_frame = input_dict['science_list']
        science_files = file_names[file_names.index(ref_frame):]

    if not overwrite and os.path.isfile('white_light.txt'):
        test = np.loadtxt('white_light.txt')
        n = len(test[:,0])
        print("...loading from frame %d"%n)
        science_files = science_files[n:]


    bias = input_dict['master_bias']
    flat = input_dict['master_flat']
    bad_pixel_mask = input_dict['bad_pixel_mask']
    cosmic_pixel_mask = input_dict['cosmic_pixel_mask']

    if "JWST" in input_dict["instrument"]:
        gain_file = input_dict["gain_file"]
        readnoise_file = input_dict["readnoise_file"]
    else:
        gain_file = readnoise_file = None

    sf,se,time = extract_all_frame_fluxes(science_files,bias,flat,trace_location_dict,window_info_dict,extraction_params_dict,verbose=v,bad_pixel_mask=bad_pixel_mask,cosmic_pixel_mask=cosmic_pixel_mask,oversampling_factor=oversampling_factor,gain_file=gain_file,readnoise_file=readnoise_file)

    if input_dict["instrument"] == "Keck/NIRSPEC":
        f_norm = np.array([f/np.nanmean(f) for f in sf])
        master_spectrum = np.nanmedian(f_norm,axis=0)
        residual_spectra = f_norm-master_spectrum
        print("\n\nStandard deviation of residual spectra = %f\n"%(np.nanstd(residual_spectra)))
        log = open('reduction_output.log','a')
        log.write("\n\nStandard deviation of residual spectra = %f\n"%(np.nanstd(residual_spectra)))

    # ~ if nstars > 1:
    generate_wl_curve(sf,se,time,nstars,overwrite)
    return


def create_masks(masked_region_list,nstars,mask_width):

    masks = {}

    for i in range(nstars):
        if mask_width is None:
            mask = [range(masked_region_list[i][j],masked_region_list[i][j+1]) for j in range(0,len(masked_region_list[i]),2)]
        else:
            mask_width = int(mask_width)
            mask = [range(masked_region_list[i][j]-mask_width//2,masked_region_list[i][j]+mask_width//2) for j in range(len(masked_region_list[i]))]

        flattened_mask = np.array([y for x in mask for y in x])
        if flattened_mask.size == 0: # array is empty
            flattened_mask = None
        masks['mask%d'%(i+1)] = flattened_mask

    return masks


def rectify_spatial(data, curve):
    """
    Shift data, column by column, along y-axis according to curve.
    Returns shifted image.
    Throws IndexError exception if length of curve
    is not equal to number of columns in data.
    """

    # shift curve to be centered at middle of order
    # and change sign so shift is corrective
    curve_p = -1.0 * curve
    curve_p = curve_p - np.median(curve_p)

    rectified = []
    for i in range(0, len(curve_p)):
        keep_index = np.isfinite(data[i])
        row = data[i]
        row[~keep_index] = 0

        rectified_row = interpolation.shift(row, curve_p[i], order=3, mode='nearest', prefilter=True)

        rectified_row[np.abs(rectified_row) <= 1e-30] = np.nan
        rectified.append(rectified_row)

    return((np.array(rectified)))


def resample_frame(data,oversampling=10,xmin=0,verbose=False):
    """A function that resamples all rows within an image to a greater sampling via linear interpolation. This is being tested as a method to deal with partial pixel extraction

    Inputs:
    data - the 2D spectral image
    oversampling - the number of sub-pixels in which to split each larger pixel into. Default=10
    xmin - if the data frame is a cut out of the larger frame, can define xmin as the left hand column for consistent x arrays. Default=0.
    verbose - True/False - do we want to plot the output of the resampling?

    Returns:
    data_resampled - the resampled image data"""

    nrows,ncols = data.shape
    old_x = np.arange(xmin,ncols)
    # new_x = np.arange(xmin,ncols-1+1/oversampling,1/oversampling)
    new_x = np.linspace(xmin,ncols,ncols*oversampling)

    data_resampled = np.array([np.interp(new_x,old_x,y) for y in data])

    if verbose:
        if verbose == -1:
            verbose = False

    if verbose:
        plt.figure()
        plt.plot(old_x,data[nrows//2],'ko',ms=6,mfc="k",label="Pre-oversampling",zorder=10)
        # plt.plot(old_x,data[nrows//2],'ko',ms=10,mfc="none",label="Pre-oversampling")
        plt.plot(new_x,data_resampled[nrows//2],'r.',label='Post-oversampling')
        plt.xlabel("X pixel")
        plt.ylabel("Counts (DN/s)")
        plt.title("Pixel resampling to deal with partial pixels")
        plt.legend()
        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()

    return data_resampled


main()
