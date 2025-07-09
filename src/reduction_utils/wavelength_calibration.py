#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

from astropy.io import fits
from astropy.modeling.models import Moffat1D
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy import stats,optimize,interpolate,conjugate, polyfit
import pickle
from scipy.fftpack import fft, ifft
from scipy.interpolate import UnivariateSpline as US
from scipy.signal import medfilt as MF
from astropy.stats import median_absolute_deviation
import warnings

with warnings.catch_warnings():
    # This is to supress the warnings that are generated upon importing pysynphot and are only becuase we've not downloaded the calibration files - which are not needed by us
    warnings.filterwarnings("ignore", category=UserWarning)
    from pysynphot import observation
    from pysynphot import spectrum


def rebin_spec(wave, specin, wavnew):
    """Using pysynphot which conserves flux during resampling.
    A function that resamples (linearly interpolates) wavelengths and spectra onto a desired wavelength spacing. Formerly this used STSci functions but np.interp performs the same function.

    Inputs:
    wave - the (old) wavelengths to be resampled
    specin - the single 1D spectrum/error corresponding to the old wavelengths
    wavnew - the wavelength array to be resampled onto

    Returns:
    resampled_spectra - the 1D array of the resampled 1D spectrum/errors"""

    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

    return obs.binflux


def resample_spectra(current_pixels,star,error,sampled_grid):

    """The function that iterates through the ndarray of all 1D spectra and errors and resamples them.

    Inputs:
    current_pixels - the ndarray of the current pixel locations (equal to np.arange(0,len(spectrum)))
    star - the ndarray of the 1D spectra
    error - the ndarray of the 1D spectra's errors
    sampled_grid - the new wavelengths/pixels to resample all 1D spectra onto

    Returns:
    np.array(resampled_fluxes) - the ndarray of resampled 1D spectra
    np.array(resampled_errors) - the ndarray of resampled 1D errors
    """

    resampled_flux = []
    resampled_error = []

    clip_idx = sampled_grid > 0
    sampled_grid = sampled_grid[clip_idx]

    for i,s in enumerate(star):

        non_negative = current_pixels[i] > 0

        resampled_flux.append(rebin_spec(current_pixels[i][non_negative],star[i][non_negative],sampled_grid))

        resampled_error.append(rebin_spec(current_pixels[i][non_negative],error[i][non_negative],sampled_grid))

    return np.array(resampled_flux),np.array(resampled_error)

def find_solution(z_pos, phi_ref_check_m):
    '''
    James McCormac's function. Convert the CCF into a shift solution for individual axes
    The location of the peak in the CCF is converted to a shift
    in pixels here. Sub pixel resolution is achieved by solving a
    quadratic for the minimum, using the three pixels around the peak.
    Parameters
    ----------
    z_pos : int
        The location of the peak in the CCF
    phi_ref_check_m: array-like
        The CCF array from which to extract a correction
    Returns
    -------
    solution : float
        The shift in pixels between two images along the
        given axis
    Raises
    ------
    None
    '''
    tst = np.empty(3)
    if z_pos[0][0] <= len(phi_ref_check_m) / 2 and z_pos[0][0] != 0:
        lra = [z_pos[0][0] - 1, z_pos[0][0], z_pos[0][0] + 1]
        tst[0] = phi_ref_check_m[lra[0]].real
        tst[1] = phi_ref_check_m[lra[1]].real
        tst[2] = phi_ref_check_m[lra[2]].real
        coeffs = polyfit(lra, tst, 2)
        solution = -(-coeffs[1] / (2 * coeffs[0]))
    elif z_pos[0][0] > len(phi_ref_check_m) / 2 and z_pos[0][0] != len(phi_ref_check_m) - 1:
        lra = [z_pos[0][0] - 1, z_pos[0][0], z_pos[0][0] + 1]
        tst[0] = phi_ref_check_m[lra[0]].real
        tst[1] = phi_ref_check_m[lra[1]].real
        tst[2] = phi_ref_check_m[lra[2]].real
        coeffs = polyfit(lra, tst, 2)
        solution = len(phi_ref_check_m) + (coeffs[1] / (2 * coeffs[0]))
    elif z_pos[0][0] == len(phi_ref_check_m) - 1:
        lra = [-1, 0, 1]
        tst[0] = phi_ref_check_m[-2].real
        tst[1] = phi_ref_check_m[-1].real
        tst[2] = phi_ref_check_m[0].real
        coeffs = polyfit(lra, tst, 2)
        solution = 1 + (coeffs[1] / (2 * coeffs[0]))
    else:  # if z_pos[0][0] == 0:
        lra = [1, 0, -1]
        tst[0] = phi_ref_check_m[-1].real
        tst[1] = phi_ref_check_m[0].real
        tst[2] = phi_ref_check_m[1].real
        coeffs = polyfit(lra, tst, 2)
        solution = -coeffs[1] / (2 * coeffs[0])
    return solution #* u.pixel

def cross_correlate(frame, reference_image):
    """James McCormac's function that performs the cross-correlation.

    Inputs:
    frame - the 1D spectrum
    reference_image - the spectrum to compare to

    Returns:
    z_pos_x,phi_ref_check_m_x - outputs which can be passed to find_solution"""


    # FFT of the projection spectra
    f_ref_xproj = fft(reference_image)
    f_check_xproj = fft(frame)
     # cross correlate in and look for the maximium correlation
    f_ref_xproj_conj = conjugate(f_ref_xproj)
    complex_sum_x = f_ref_xproj_conj * f_check_xproj

    phi_ref_check_m_x = ifft(complex_sum_x)

    z_x = max(phi_ref_check_m_x)
    z_pos_x = np.where(phi_ref_check_m_x == z_x)


    return z_pos_x, phi_ref_check_m_x



def compute_shifts(frame,reference_frame,line_positions,box_width=10,verbose=False):

    """
    The function that performs the cross correlation for a single frame/1D spectrum.

    Inputs:
    frame - the 1D spectrum
    reference_frame - the reference 1D spectrum
    line_positions - the positions of absorption lines to look into with the cross correlation
    box_width - the width of the region around each absorption line over which the cross correlation is performed. Default=10
    verbose - True/False - plot the output? Default=False

    Returns:
    np.array(shifts) - the measured shifts between the input spectrum and the reference
    """

    shifts = []

    if verbose:
        plt.figure()
        nfeatures = len(line_positions)

    for i,l in enumerate(line_positions):
        if isinstance(box_width,list):
            ref_region = reference_frame[l-box_width[i]:l+box_width[i]]
            frame_region = frame[l-box_width[i]:l+box_width[i]]
        else:
            ref_region = reference_frame[l-box_width:l+box_width]
            frame_region = frame[l-box_width:l+box_width]


        # Now lets normalise
        #ref_region = ref_region/np.median(ref_region)
        #frame_region = frame_region/np.median(frame_region)
        # ref_region = ref_region/min(ref_region)
        # frame_region = frame_region/min(frame_region)
        poly_ref = np.poly1d(np.polyfit(np.array([0,len(ref_region)]),np.array([ref_region[0],ref_region[-1]]),1))
        ref_region = ref_region/poly_ref(np.arange(len(ref_region)))
        ref_region = ref_region/min(ref_region)

        poly_frame = np.poly1d(np.polyfit(np.array([0,len(frame_region)]),np.array([frame_region[0],frame_region[-1]]),1))
        frame_region = frame_region/poly_ref(np.arange(len(frame_region)))
        frame_region = frame_region/min(frame_region)


        cc = cross_correlate(frame_region,ref_region)
        sol = find_solution(cc[0],cc[1])

        if verbose:
            #plt.subplot(nfeatures,1,i+1)
            plt.plot(ref_region-i*0.4,color='r')
            plt.plot(frame_region-i*0.4,color='b',label=str(sol))
            plt.yticks(visible=False)
            plt.legend(loc='upper right')
            plt.xlabel('Pixel position')
            plt.ylabel('Flux')
            plt.title('Input spectrum (blue) vs. reference spectrum (red)')

        shifts.append(sol)

    if verbose:
        plt.show()

    return np.array(shifts)


def polyfit_shifts(line_positions,shifts,npoints,poly_order=3,verbose=False,refit_polynomial=None):

    """
    The function that takes the measured shifts, from either Moffat fitting or cross-correlation, and calculates the 'pixel solution' - fitting the shifts to relate the reference locations to the measured locations.

    Inputs:
    line_positions - the array of reference line positions, as measured in the reference spectrum
    shifts - the ndarray of measured shifts for all 1D spectra
    npoints - the length of the x-axis if this is set to an integer. If the wavelength solution has already been calculated, this can be defined as 1D array of wavelengths in A.
    poly_order - the order of the polynomial used to fit the measured shifts
    verbose - True/False - plot the output or not
    refit_polynomial - Define whether we want to refit the polynomial used to define the pixel solution after the initial fit. This allows us to clip outliers from the first fit.
                       The value given to refit_polynomial is an integer and is interpreted as the number of standard deviations away from the residuals that should be clipped.

    Returns:
    np.array(pixels) - the new x-axis (the fitted polynomial evaluated at the locations of the reference frame)
    """

    poly = np.poly1d(np.polyfit(line_positions,line_positions+shifts,poly_order))

    if type(npoints) is not float and type(npoints) is not int:
        x = npoints
        pixels = poly(x) # the pixels here are actually a wavelength array
    else:
        x = range(npoints)
        pixels = poly(x)

    residuals = line_positions+shifts-poly(line_positions)

    if verbose:
        plt.figure()
        ax1 = plt.subplot(211)

    if refit_polynomial is not None:
        std_residuals = np.std(residuals)
        keep_idx = ((residuals <= refit_polynomial*std_residuals) & (residuals >= -refit_polynomial*std_residuals))

        if verbose:
            plt.plot(line_positions[~keep_idx],line_positions[~keep_idx]+shifts[~keep_idx],'kx',label='Clipped point')
            plt.legend(loc='upper left')

        line_positions = line_positions[keep_idx]
        shifts = shifts[keep_idx]
        poly = np.poly1d(np.polyfit(line_positions,line_positions+shifts,poly_order))
        pixels = poly(range(npoints))
        residuals = line_positions+shifts-poly(line_positions)



    if verbose:

        plt.scatter(line_positions,line_positions+shifts,color='r')
        plt.plot(x,pixels,color='k',label='Poly order = '+str(poly_order))
        plt.xticks(visible=False)
        plt.ylabel('Recorded pixel positions')
        plt.subplot(212,sharex=ax1)
        plt.plot(line_positions,residuals,'ro')
        plt.xlabel('Reference pixel positions')
        plt.ylabel('Residuals')
        plt.show()

    # if refit_polynomial is not None:
    #     return pixels,keep_idx

    return pixels



def compute_all_shifts(ref_frame,all_frames,all_errors,line_positions,search_width=10,poly_order=3,verbose=False,refit_polynomial=None,resample=True,ancillary_data=None):
    """The function that computes the shifts in the spectra for all 1D spectra using the cross-correlation technique.

    Inputs:
    ref_frame - the reference frame which all other spectra are compared with
    all_frames - the ndarray of all 1D spectra (including the reference frame itself)
    all_errors - the ndarray of errors on all 1D spectra (including the reference frame itself)
    line_positions - the list of absorption lines which are used to determine the shifts
    search_width - the seach width (in x data points) around each absorption line to consider. This clips out this many data points and performs the cross correlation for this region. Default=10
    poly_order - the order of the polynomial used for the pixel solution relating the reference pixel locations to the measured pixel locations. Default=3
    verbose - True/False. Use this to plot the measured shifts for each line. Default=False
    refit_polynomial - Define whether we want to refit the polynomial used to define the pixel solution after the initial fit. This allows us to clip outliers from the first fit.
                       The value given to refit_polynomial is an integer and is interpreted as the number of standard deviations away from the residuals that should be clipped.
    resample - True/False - Set whether you want this function to perform the flux and error resampling. Default=True.
    ancillary_data - a dictionary of ancillary data to be resampled (e.g., xpos, sky). Can be left as None, in which case no ancillary data is resampled

    Returns:
    np.array(resampled_flux) - the 1D spectra resampled onto the same x-axis as the reference spectrum
    np.array(resampled_error) - the error on the 1D spectra resampled onto the same x-axis as the reference spectrum
    np.array(all_shifts) - the shifts measured by the cross correlation for all spectra and lines
    np.array(all_polys) - the pixel solutions for all spectra

    """
    resampled_flux = []
    resampled_error = []
    all_shifts = []
    all_polys = []


    npoints = len(ref_frame)
    nfeatures = len(line_positions)

    if ancillary_data is not None:
        resampled_dict = {}

        # populate new dictionary with keys of old dictionary and empty lists of values
        for i in ancillary_data.keys():
            resampled_dict[i] = []

    for i,f in enumerate(all_frames):
        shifts = compute_shifts(f,ref_frame,line_positions,search_width)
        all_shifts.append(shifts)

        pixel_solution = polyfit_shifts(line_positions,shifts,npoints,poly_order,False,refit_polynomial)

        all_polys.append(pixel_solution)

        # Only take the positive indices
        idx = pixel_solution > 0

        if resample:
            resampled_flux.append(rebin_spec(pixel_solution[idx],f[idx],np.arange(1,npoints+1)))
            resampled_error.append(rebin_spec(pixel_solution[idx],all_errors[i][idx],np.arange(1,npoints+1)))

            if ancillary_data is not None:
                for k in resampled_dict.keys():
                    resampled_dict[k].append(rebin_spec(pixel_solution[idx],ancillary_data[k][i][idx],np.arange(1,npoints+1)))


    if verbose:

        s = np.array(all_shifts)
        plt.figure()
        for i in range(nfeatures):
            plt.plot(s[:,i],label=str(i+1))
        plt.ylabel('Pixel shift in feature')
        plt.xlabel('Frame number')
        plt.show()

    if ancillary_data is not None:
        # convert from lists to arrays
        for d in resampled_dict.keys():
            resampled_dict[d] = np.array(resampled_dict[d])
        return np.array(resampled_flux),np.array(resampled_error),resampled_dict,np.array(all_shifts),np.array(all_polys)
    else:
        return np.array(resampled_flux),np.array(resampled_error),np.array(all_shifts),np.array(all_polys)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]




def compute_all_shifts_whole_spectrum(ref_frame,all_frames,all_errors,verbose=False,resample=True,ancillary_data=None,user_shifts=None):
    """The function that computes the shifts in the spectra for all 1D spectra using the cross-correlation technique.
    Unlike compute_all_shifts() which computes the cross-correlation function at different points along the spectrum,
    this function computes one cross-correlation function for the entire function, which is applicable when the spectra are
    only shifting, not also distorting.

    Inputs:
    ref_frame - the reference frame which all other spectra are compared with
    all_frames - the ndarray of all 1D spectra (including the reference frame itself)
    all_errors - the ndarray of errors on all 1D spectra (including the reference frame itself)
    verbose - True/False. Use this to plot the measured shifts for each line. Default=False
    resample - True/False - Set whether you want this function to perform the flux and error resampling. Default=True.
    ancillary_data - a dictionary of ancillary data to be resampled (e.g., xpos, sky). Can be left as None, in which case no ancillary data is resampled
    user_shifts - if wanting to override the shifts calculated for this star with shifts calculated from another star, parse the new user-defined shifts here

    Returns:
    np.array(resampled_flux) - the 1D spectra resampled onto the same x-axis as the reference spectrum
    np.array(resampled_error) - the error on the 1D spectra resampled onto the same x-axis as the reference spectrum
    np.array(all_shifts) - the shifts measured by the cross correlation for all spectra and lines
    """

    resampled_flux = []
    resampled_error = []
    all_shifts = []

    # replace any nans with a linear interpolation for finding shifts only - the end spectra still contain nans
    if np.any(~np.isfinite(ref_frame)):
        yf = ref_frame.copy()
        nans, x= nan_helper(yf)
        yf[nans]= np.interp(x(nans), x(~nans), yf[~nans])
        ref_frame = yf

    if ancillary_data is not None:
        resampled_dict = {}

        # populate new dictionary with keys of old dictionary and empty lists of values
        for i in ancillary_data.keys():
            resampled_dict[i] = []

    x_ref = np.arange(10,len(ref_frame)+10) # add 10 here, this keeps the pixel solution positive and doesn't actually matter since we're not using this as our final solution, only for calculating shifts!

    for i,f in enumerate(all_frames):
        print("cross-correlating frame %d"%i)

        # replace any nans with a linear interpolation for finding shifts only - the end spectra still contain nans
        if np.any(~np.isfinite(f)):
            y = f.copy()
            nans, x= nan_helper(y)
            y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        else:
            y = f

        if user_shifts is None:
            cc = cross_correlate(y[20:-20],ref_frame[20:-20]) # ignore 20 pixels at edge of spectra when doing cross-correlation
            shift = find_solution(cc[0],cc[1])
        else:
            shift = user_shifts[i]

        all_shifts.append(shift)

        x = x_ref + shift

        interp = rebin_spec(x,f,x_ref)

        if resample:
            resampled_flux.append(rebin_spec(x,f,x_ref))
            resampled_error.append(rebin_spec(x,all_errors[i],x_ref))

            if ancillary_data is not None:
                for k in resampled_dict.keys():
                    resampled_dict[k].append(rebin_spec(x,ancillary_data[k][i],x_ref))

    if verbose:

        plt.figure()
        plt.plot(all_shifts)
        plt.ylabel('Pixel shift in spectrum')
        plt.xlabel('Frame number')
        plt.show()

    if ancillary_data is not None:
        # convert from lists to arrays
        for d in resampled_dict.keys():
            resampled_dict[d] = np.array(resampled_dict[d])
        return np.array(resampled_flux),np.array(resampled_error),resampled_dict,np.array(all_shifts)
    else:
        return np.array(resampled_flux),np.array(resampled_error),np.array(all_shifts)




def gauss2(x,a,x0,sigma):
    """A function that returns a Gaussian with a negative amplitude, for fitting absorption lines during the wavelength calibration.

    Inputs:
    x - the x array (of wavelengths/pixels)
    a - the amplitude
    x0 - the mean
    sigma - the standard deviation

    Returns:
    the evaluated Gaussian
    """

    return -a*np.exp(-(x-x0)**2/(2*sigma**2))+1

def normalise(data,maximum=False):
    """A very simple function to 'normalise' 1D spectra by either dividing the 1D spectrum by its median or setting the maximum to 1.

    Inputs:
    data - the 1D spectum to be normalised
    maximum - True/False - choose whether to set the maximum of the normalised spectrum to 1 (True) or divide by the median (False)

    Returns:
    the normalised 1D spectrum
    """

    if maximum:
        return data/np.nanmax(data)

    return data/np.nanmedian(data)

def plot_and_fit_regions(stellar_spectrum,wvl_input,guess_dict,verbose=False,work_in_wavelength=False,absorption=True):
    """
    The function that takes a 1D spectrum, wavelength array and locations of absorption lines, and fits Gaussians to each absorption line. It also returns the position of the minimum flux for each line.

    Note: this could be improved by replacing the Gaussian with a Moffat fit although this is not yet implemented.

    Inputs:
    stellar_spectrum - the 1D spectrum
    wvl_input - the wavelength array
    guess_dict - the array of absorption line centres. This does not have to be 100% accurate, these are just starting guesses for the means of the Gaussians.
    verbose - True/False - plot the output or not. Default=False
    work_in_wavelength - True/False - decide whether to work in wavelength space (A) or pixel space. Default is pixel space.
    absorption - True/False - are we looking at absorption lines (a stellar spectrum) or emission lines (an arc spectrum)?

    Returns:
    np.array(star_centres_gauss) - the means of the Gaussians fitted to each absorption line
    np.array(star_centres_argmin) - the locations of the minimum flux for each absorption line
    """

    nregions = len(guess_dict.keys())

    spectral_lines = sorted(guess_dict.keys())

    star_centres_gauss = []
    star_centres_argmin = []

    plt.close('all')
    for i,l in enumerate(spectral_lines):

        chunk = (wvl_input > guess_dict[l][0]) & (wvl_input < guess_dict[l][-1])

        y = stellar_spectrum[chunk]
        if work_in_wavelength:
            x = wvl_input[chunk]
        else:
            x = np.where(chunk)[0]

        poly = np.poly1d(np.polyfit((x[0],x[-1]),(y[0],y[-1]),1))
        norm_y = y/poly(x)

        if absorption:
            # Find argmin
            minimum = x[np.argmin(norm_y)]
            star_centres_argmin.append(minimum)
            centre_guess = minimum
            amplitude_guess = 1-norm_y.min()

        else:
            # Find argmax
            maximum = x[np.argmax(norm_y)]
            star_centres_argmin.append(maximum)
            centre_guess = maximum
            amplitude_guess = norm_y.max()

        try:
            popt,pcov = optimize.curve_fit(gauss2,x,norm_y,p0=[amplitude_guess,centre_guess,5])
            star_centres_gauss.append(popt[1])
        except:
            print('Gaussian failed for line %d'%(i+1))
            star_centres_gauss.append(centre_guess)


        if verbose:
            plt.figure()
            plt.plot(x,norm_y,'b')
            plt.plot(x,gauss2(x,*popt),'g')
            plt.axvline(centre_guess,color='b',label='Centre guess = %d'%centre_guess)
            plt.axvline(popt[1],color='g',label='Gauss = %.3f'%popt[1])
            if work_in_wavelength:
                plt.xlabel('Wavelength ($\AA$)')
            else:
                plt.xlabel('Pixel number')
            plt.ylabel('Normalised flux')
            plt.legend(loc='lower right')
            plt.title('Actual line = %.3f'%l)
            plt.show()

            fwhm = 2*np.sqrt(2*np.log(2))*popt[2]
            R = l/fwhm

            print("centre guess (argmin/argmax) = %d; Gauss mean = %.3f; Gauss std dev = %.4f; Spectral resolution = %d; Min bin width = %d"%(centre_guess,popt[1],popt[2],np.round(R),np.round(l/R)))

    return np.array(star_centres_gauss),np.array(star_centres_argmin)

def wavelength_solution_multiple_spectra(spectra=None,fitted_centres=None,regions=None,wavelengths=None,absorption=True,\
                                         verbose=False,poly_order=None,refit_clip=None):

    """Fit the wavelength solution for time series spectra with different wavelength solutions"""
    fitted_wavelengths = []
    x = np.arange(len(spectra[0]))

    if fitted_centres is None:
        centres = []
    else:
        centres = fitted_centres

    for i,s in enumerate(spectra):

        if fitted_centres is None:
            fc,_ = plot_and_fit_regions(s,x,regions,verbose=verbose,absorption=absorption)
            centres.append(fc)
        else:
            fc = centres[i]

        w,_,_,_ = calc_wvl_solution(fc,wavelengths,poly_order,s,verbose=verbose,refit_clip=refit_clip)

        fitted_wavelengths.append(w)

    if fitted_centres is None:
        return np.array(centres),np.array(fitted_wavelengths)
    else:
        return np.array(fitted_wavelengths)



def calc_wvl_solution(pixel_values,line_wvls,poly_order,stellar_spectrum,verbose=True,refit_clip=None):
    """
    The function that takes the measured locations of absorption lines, in pixels, and fits a polynomial to these to calculate the wavlength solution (in Angstroms).

    Inputs:
    pixel_values - the location of the absorption lines in pixels
    line_wvls - the wavelengths (in A) that these absorption lines correspond to
    poly_order - the order of the polynomial used to generate the wavelength solution
    stellar_spectrum - the 1D reference stellar spectrum, used for plotting purposes
    refit_clip - refit the polynomial after clipping outliers from the first fit? Set this to the number of standard deviations at which to clip.

    Returns:
    np.array(wvl_solution) - the wavelength solution
    poly - the polynomial used to generate the wavelength solution
    """

    poly = np.poly1d(np.polyfit(pixel_values,line_wvls,poly_order))
    nrows = len(stellar_spectrum)

    if verbose:
        plt.figure()
        plt.subplot(211)
        plt.title('Fit')
        plt.plot(pixel_values,line_wvls,'ro')
        plt.plot(range(nrows),poly(range(nrows)),'b')
        plt.xticks(visible=False)
        plt.ylabel('Wavelength ($\AA$)')

        plt.subplot(212)
        plt.title('Residuals')
        plt.xlabel('Y pixel')
        plt.ylabel('Wavelength ($\AA$)')
        plt.plot(pixel_values,line_wvls-poly(pixel_values),'ro')

    wvl_solution = poly(range(nrows))
    model = poly(pixel_values)
    residuals = line_wvls - model
    npoints = len(pixel_values)

    if refit_clip is not None:

        keep_index = ((residuals >= -refit_clip*median_absolute_deviation(residuals)) & (residuals <= refit_clip*median_absolute_deviation(residuals)))
        print("%d/%d points kept after clipping"%(len(np.where(keep_index)[0]),len(line_wvls)))

        if verbose:
            plt.axhline(refit_clip*median_absolute_deviation(residuals),ls='--')
            plt.axhline(refit_clip*-median_absolute_deviation(residuals),ls='--')

        poly = np.poly1d(np.polyfit(np.array(pixel_values)[keep_index],np.array(line_wvls)[keep_index],poly_order))
        wvl_solution = poly(range(nrows))

        model = poly(np.array(pixel_values)[keep_index])
        residuals = np.array(line_wvls)[keep_index] - model
        npoints = len(np.array(pixel_values)[keep_index])

        if verbose:
            plt.figure()
            plt.subplot(211)
            plt.title('Fit after clipping outliers')
            plt.plot(pixel_values[keep_index],line_wvls[keep_index],'ro')
            plt.plot(range(nrows),poly(range(nrows)),'b')
            plt.xticks(visible=False)
            plt.ylabel('Wavelength ($\AA$)')

            plt.subplot(212)
            plt.title('Residuals')
            plt.xlabel('Y pixel')
            plt.ylabel('Wavelength ($\AA$)')
            plt.plot(pixel_values[keep_index],line_wvls[keep_index]-poly(pixel_values[keep_index]),'ro')


    if verbose:
        plt.show()


    chi2 = np.sum((residuals**2)/model)
    BIC = chi2 + (poly_order+1)*np.log(npoints)
    print("RMS residuals = %f"%(np.sqrt(np.mean(residuals**2))))

    return wvl_solution, poly, chi2, BIC


def resample_smoothly(reference_pixel_locations,measured_shifts,input_arrays,sigma_clip_outliers=3,median=False,poly_order=3,mf_box_width=None,min_good=0.9,spline_smoothing_factor=None,verbose=False,refit_polynomial=None,reference_wvl_array=None,use_pysynphot=True):
    """

    Use this function to resample spectra, following a spline smoothing to the measured shifts from cross-correlation/Gaussian fitting.

    This is recommended to reduce the noise in the resulting light curves.

    Input:
    reference_pixel_locations - the pixel positions of the absorption features in the reference spectrum. As found through Gaussian fitting or defined for the cross-correlation.
    measured_shifts - the measured shifts between each spectrum and the reference spectrum, as obtained through fts.
    input_arrays - dictionary of arrays of to be resampled. Typically {flux,error,xposition,sky}
    sigma_clip_outliers - use this to clip any outliers from the measured shifts. Default is 3 sigma cut. For no sigma cut, set to 0. NOTE: If median==True, this number actually sets the largest shift (in pixels) at which to clip the array. E.g. sigma_clip_outliers = 7 would clip any measured_shifts > 7 pixels.
    median - True/False. Use this to perform the smoothing with a median filter rather than spline. This can be useful for a large number of frames. Default is False.
    poly_order - the order of the polynomial used for the pixel solution relating the reference pixel locations to the measured pixel locations. Default=3
    mf_box_width - the number of data points over which the median filter is calculated. Leave as None if median=False
    min_good - minimum fraction of good line measurements (i.e. not nan values) needed to not ignore a line. Default=0.9.
    spline_smoothing_factor - If using a median, this sets the spline smoothing factor ('s' parameter in scipy's UnivariateSpline) of the spline used to fit the running median.
                              This is necessary as sometimes the running median is still not smooth enough and needs further smoothing
    verbose - True/False. Define whether we want to plot the pixel solution for the first and last frame. This is typically preferable!
    refit_polynomial - Define whether we want to refit the polynomial after the initial fit. This allows us to clip outliers from the first fit.
                       The value given to refit_polynomial is an integer and is interpreted as the number of standard deviations away from the residuals that should be clipped.
    reference_wvl_array: can be set to the wavelength array to resample the data onto if working in wavelength, not pixel, space. If working in pixel space, leave this as None.
    use_pysynphot - True/False. Choose whether to use pysynphot to perform the resampling (via rebin_spec) or use np.interp for 1D linear interpolation. The latter doesn't necessarily conserve flux and should be treated with caution! Default=True.

    Returns:
    resampled_dict - the dictionary of inputs resampled onto the desired x-axis
    len(good_lines) - the number of abosrption lines used in the resampling
    smooth_shifts - the smoothed shifts
    good_lines - the indices of the lines actually used

    """

    smooth_shifts = []

    nfeatures = measured_shifts.shape[1]

    frames = np.arange(len(input_arrays['flux']))

    good_lines = []

    for i in range(nfeatures):

        index = np.isfinite(measured_shifts[:,i])

        # print(frames[index],measured_shifts[:,i][index])

        number_of_good_measurements = len(np.where(index)[0])

        # only use lines with a reasonable number of good measurements
        if number_of_good_measurements < min_good*len(frames):
            continue

        else:
            good_lines.append(i)


        # using a spline to smooth the pixel shifts
        if not median:

            spline = US(frames[index],measured_shifts[:,i][index],s=spline_smoothing_factor)

            plt.figure()

            if sigma_clip_outliers > 0:

                spline_residuals = measured_shifts[:,i][index]-spline(frames[index])
                keep_idx = ((spline_residuals <= sigma_clip_outliers*np.std(spline_residuals)) & (spline_residuals >= -sigma_clip_outliers*np.std(spline_residuals)))
                spline = US(frames[index][keep_idx],measured_shifts[:,i][index][keep_idx],s=spline_smoothing_factor)

                plt.plot(frames,measured_shifts[:,i],'r')
                plt.plot(frames[index][keep_idx],measured_shifts[:,i][index][keep_idx],'k',label='measured shifts')
                plt.plot(frames,spline(frames),label='spline',color='orange')

            else:
                plt.plot(frames[index],measured_shifts[:,i][index],'k',label='measured shifts')
                plt.plot(frames,spline(frames),label='spline',color='orange')

            plt.xlabel('Frame number')
            plt.ylabel('Shift in pixels')
            plt.title('Feature #%d'%(i+1))
            plt.legend(loc='upper right')
            plt.show()

            smooth_shifts.append(spline(frames))

        # using a running median to smooth the pixel shifts
        else:

            if sigma_clip_outliers > 0:
                norm_shifts = measured_shifts[:,i][index] - np.median(measured_shifts[:,i][index])
                # keep_idx = ((measured_shifts[:,i][index] >= -sigma_clip_outliers) & (measured_shifts[:,i][index] <= sigma_clip_outliers)) # cut large outliers
                keep_idx = ((norm_shifts >= -sigma_clip_outliers) & (norm_shifts <= sigma_clip_outliers)) # cut large outliers
                clipping = True
            else:
                keep_idx = np.arange(len(np.where(index)[0])) # keeping all points
                clipping = False

            if mf_box_width is None:
                # make box width 1/10th of array length
                box_width = len(frames[index])/10
                # convert to odd number (needed for median filter)
                box_width = int(np.ceil(box_width) // 2 * 2 + 1)

            else:
                box_width = mf_box_width

            median_filter = MF(measured_shifts[:,i][index][keep_idx],box_width)

            # Use polynomial to remove edge effects of running median
            poly_left = np.poly1d(np.polyfit(frames[index][keep_idx][:box_width//2],measured_shifts[:,i][index][keep_idx][:box_width//2],1))
            poly_right = np.poly1d(np.polyfit(frames[index][keep_idx][-box_width//2:],measured_shifts[:,i][index][keep_idx][-box_width//2:],1))

            median_filter[:box_width//2] = poly_left(frames[index][:box_width//2])
            median_filter[-box_width//2:] = poly_right(frames[index][-box_width//2:])

            if number_of_good_measurements < len(frames) or clipping:
                 # we have to fit a spline to the median filter to then interpolate across all x
                 spline = US(frames[index][keep_idx],median_filter,s=spline_smoothing_factor)
                 median_filter = spline(frames)

            plt.figure()
            plt.plot(frames,measured_shifts[:,i],'r')
            plt.plot(frames[index][keep_idx],measured_shifts[index][:,i][keep_idx],'k',label='measured shifts')
            plt.plot(frames,median_filter,label='median filter',color='orange')
            plt.xlabel('Frame number')
            plt.ylabel('Shift in pixels')
            plt.title('Feature #%d'%(i+1))
            plt.legend(loc='upper right')
            plt.show()

            smooth_shifts.append(median_filter)

    smooth_shifts = np.array(smooth_shifts).T

    print('# lines used = %d'%(len(good_lines)))

    resampled_dict = {}

    # populate new dictionary with keys of old dictionary and empty lists of values
    for i in input_arrays.keys():
        resampled_dict[i] = []

    if reference_wvl_array is None:
        npixels = len(input_arrays['flux'][len(frames)//2])
        if use_pysynphot:
            x_ref = np.arange(1,npixels+1)
        else:
            x_ref = np.arange(npixels)

    else:
        x_ref = npixels = reference_wvl_array

    for i in frames:
        if i == 0 and verbose or i == len(frames)-1 and verbose:
            print('PIXEL SOLUTION PLOT FOR FRAME %d'%i)
            v = True
        else:
            v = False

        # Calculate the pixel solution, using the reference values and shifts.
        pixel_solution_refitted = polyfit_shifts(reference_pixel_locations[good_lines],smooth_shifts[i],npixels,poly_order=poly_order,verbose=v,refit_polynomial=refit_polynomial)

        if use_pysynphot:
            # Only take the positive indices
            idx = pixel_solution_refitted > 0

        for d in input_arrays:

            if use_pysynphot:
                resampled_dict[d].append(rebin_spec(pixel_solution_refitted[idx],input_arrays[d][i][idx],x_ref))

            else:
                resampled_dict[d].append(np.interp(x_ref,pixel_solution_refitted,input_arrays[d][i]))

    # convert from lists to arrays
    for d in input_arrays:
        resampled_dict[d] = np.array(resampled_dict[d])

    return resampled_dict,len(good_lines),smooth_shifts,good_lines



def moffat(a,r):
    """The Moffat profile.

    This is given by (see Wikipedia page):

    f(r,alpha,beta) = (2*(beta-1)/(alpha**2))*(1+(r**2/alpha**2))**(-beta)

    where, in our case, r=pixel number and

    Inputs:
    a - the list of [alpha,beta,location,width,amplitude,continuum flux]
    r - the x array (typically pixels)

    Returns:
    the evaluated Moffat profile
    """

    x = 2. * (a[1]-1.) / (a[0] * a[0]) # this the first factor in the Moffat function
    y = 1. + ((r-a[2])/a[0])**2. # this is the second factor in the Moffat function
    return a[3] * (x * (y ** -a[1])) + a[4] * r + a[5] # a[3:5] are normalisation factors to scale the Moffat function to the stellar spectrum



def moffat_residuals(a,r,y):
    """The function that returns the residuals to the Moffat profile, which is needed to opimize the fit through scipy.

    Inputs:
    a - the list of [alpha,beta,location,width,amplitude,continuum flux] as required by moffat()
    r - the x array (typically pixels)
    y - the y array (flux)

    Returns:
    the residuals of the Moffat profile (y - moffat(a,r)) for use by scipy optimize functions
    """

    mod = moffat(a,r)
    return y - mod




def moffat_fit_lines(flux,error,line_positions,wvl=None,tolerance=10,box_width=60,enforce_negative=False,verbose=False,return_nans=True):
    """A function to fit Moffat profiles at a list of user-defined absorption line locations for a SINGLE stellar spectrum, not an ndarry of multiple spectra. For this use fit_all_moffat_profiles().

    Inputs:
    flux - the 1D stellar spectrum
    error - the error on the 1D stellar spectrum's fluxes
    line_positions - the list of guess locations for absorption lines to be fit. Note: if wanting to fit a single line, this must still be defined as a list but with length=1
    wvl - the wavelengths can be supplied in Angstroms if the wavelength solution has already been calculated. If working in pixel space, this can be left as None. Default=None
    tolerance - the maximum number of wavelength resolution elements at which to accept the solution's location, otherwise reject. For example, if set to 10 and the Moffat returns a mean > 10 pixels away from the inputted location
                it is assumed that the fit was unsuccessful and the mean is ignored. Default=10
    box_width - the width (number of resolution elements) to fit. This needs to include some continuum to get an accurate fit. This can be also set to a list of a number of different box widths, which will all be tested during
               the fitting procedure, with the first box width that corresponds to a successful fit being used. Default=60
    enforce_negative - use this to enforce that the Moffat profile's ampitude is negative (as it should be for an absorption line). HOWEVER: sometimes the other Moffat normalisation functions can prevent this from being
                       negative despite a good fit, so should be used with caution. Default=False
    verbose - True/False. If set to True, plot the results of the fit. Default=False
    return_nans - True/False. If True, any fits which have failed return centres which are at np.nan - so that they can be easily ignored by subsequent analysis while preserving the shape of the input arrays. Default=True

    Returns:
    ref_lines - the ndarray of the fitted absorption line centres
    resulting_box_widths - the ndarray of the box widths used to actually fit the data
    indices_of_good_lines - the indices (line number) of the absorption lines where a successful fit was found

    """

    if type(box_width) is not list and type(box_width) is not np.ndarray:
        box_width = [box_width]

    resulting_box_widths = []

    nframes = len(flux)

    ref_lines = []

    indices_of_good_lines = []

    for i,l in enumerate(line_positions):

        for w in box_width:

            if wvl is not None: # we're working in wavelength (A) space

                index = ((wvl >= l-w//2) & (wvl < l+w//2))
                flux_y = flux[index]
                error_y = error[index]
                x = wvl[index]

            else: # we're working in pixel space

                if (l - w//2) < 0:
                    left_pixel = 0
                else:
                    left_pixel = l - w//2

                if (l + w//2) > len(spectrum):
                    right_pixel = len(spectrum)
                else:
                    right_pixel = l + w//2

                flux_y = flux[left_pixel:right_pixel]

                x = np.arange(left_pixel,right_pixel)

            parms_y = optimize.leastsq(moffat_residuals,np.array([4.0,1.5,l,-150000.,0.,50000.]),args=(x,flux_y))

            xfine = np.arange(x[0],x[-1],0.001)

            if abs(parms_y[0][2] - l) <= tolerance:

                success = True

                if enforce_negative:
                    if parms_y[0][3] < 0:
                        ref_lines.append(parms_y[0][2])
                        resulting_box_widths.append(w)
                        indices_of_good_lines.append(i)
                        break
                    else:
                        success = False

                else:
                    ref_lines.append(parms_y[0][2])
                    resulting_box_widths.append(w)
                    indices_of_good_lines.append(i)
                    break

            else:
                success = False

        if not success and return_nans:
            ref_lines.append(np.nan)


        if verbose and success:
            plt.figure()
            plt.plot(x,flux_y,'.')
            plt.plot(xfine,moffat(parms_y[0],xfine))
            plt.axvline(parms_y[0][2],color='g',label='Fitted centre')
            plt.axvline(l,color='k',label='Guess centre')
            plt.legend()
            plt.title("Fit to line %d"%(i+1))
            plt.ylabel("Counts (ADU)")
            if wvl is not None:
                plt.xlabel("Wavelength ($\AA$)")
            else:
                plt.xlabel("Y pixel")
            plt.show()

    return np.array(ref_lines),np.array(resulting_box_widths),np.array(indices_of_good_lines)


def fit_all_moffat_profiles(flux_array,error_array,ref_lines,wvl_array=None,tolerance=10,box_width=60,enforce_negative=False,verbose=False,return_nans=True):

    """
    A function to fit Moffat profiles to the ndarray of all 1D spectra, by looping through each and running moffat_fit_lines().

    Inputs:
    flux_array - the ndarray of 1D spectra
    error_array - the ndarry of errors in the 1D spectra
    ref_lines - the list of absorption line locations as found in the fitted reference spectrum
    wvl_array - the wavelengths can be supplied in Angstroms if the wavelength solution has already been calculated. If working in pixel space, this can be left as None. Default=None
    tolerance - the maximum number of wavelength resolution elements at which to accept the solution's location, otherwise reject. For example, if set to 10 and the Moffat returns a mean > 10 pixels away from the inputted location
                it is assumed that the fit was unsuccessful and the mean is ignored. Default=10
    box_width - the width (number of resolution elements) to fit. This needs to include some continuum to get an accurate fit. This can be also set to a list of a number of different box widths, which will all be tested during
               the fitting procedure, with the first box width that corresponds to a successful fit being used. Default=60
    enforce_negative - use this to enforce that the Moffat profile's ampitude is negative (as it should be for an absorption line). HOWEVER: sometimes the other Moffat normalisation functions can prevent this from being
                       negative despite a good fit, so should be used with caution. Default=False
    verbose - True/False. If set to True, plot the results of the fit. Default=False
    return_nans - True/False. If True, any fits which have failed return centres which are at np.nan - so that they can be easily ignored by subsequent analysis while preserving the shape of the input arrays. Default=True

    Returns:
    np.array(moffat_line_centres) - the ndarray of absorption line shifts as measured for each 1D spectrum. This is calculated as: reference location - measured location

    """

    moffat_line_centres = []

    if wvl_array is None: # we make an array of pixel numbers equal in shape to flux_array
        wvl_array = np.array(list(range(0,flux_array.shape[1]))*flux_array.shape[0]).reshape(flux_array.shape[0],flux_array.shape[1])


    for spectrum,error,wavelength in zip(flux_array,error_array,wvl_array):

        current_shifts = []

        line_locations,_,_ = moffat_fit_lines(spectrum,error,ref_lines,wavelength,tolerance,box_width,enforce_negative,verbose,return_nans)

        shifts = ref_lines - line_locations

        moffat_line_centres.append(shifts)

    return np.array(moffat_line_centres)



def clip_shift_outliers(moffat_line_shifts,clip_level=2):

    """
    A function that clips outliers from the measured shift arrays, as returned by fit_all_moffat_profiles().

    Inputs:
    moffat_line_shifts - the shifts as measured by the difference in the Moffat centres for each spectrum and the reference, as returned by fit_all_moffat_profiles()
    clip_level - the number of standard deviations at which to clip the outliers. Default=2

    Returns:
    np.array(moffat_line_shifts_clipped) - the Moffat shifts with outliers replaced by np.nan

    """


    moffat_line_shifts_clipped = []

    nframes = moffat_line_shifts.shape[0]

    for i in range(nframes):

        current_frame_shifts = moffat_line_shifts[i]

        std = np.nanstd(current_frame_shifts)

        mean = np.nanmean(current_frame_shifts)

        outliers = np.array(current_frame_shifts > mean+clip_level*std) + np.array(current_frame_shifts < mean-clip_level*std)

        current_frame_shifts[outliers] = np.nan

        moffat_line_shifts_clipped.append(current_frame_shifts)

    return np.array(moffat_line_shifts_clipped)


def wavecal_F444(x):
    """
    Wavelength calibration from Flight Program 1076
    for JWST NIRCam F444 filter.
    Inputs
    ------
    x: float or numpy array
        Zero-based index of the X pixel
    Returns
    -------
    wave: numpy array
        Wavelength in microns
    """
    x0 = 852.0756
    coeff = np.array([ 3.928041104137344, 0.979649332832983])
    xprime = (x - x0)/1000.
    poly = np.polynomial.Polynomial(coeff)
    return poly(xprime)
