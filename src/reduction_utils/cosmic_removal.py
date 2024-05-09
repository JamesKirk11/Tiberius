#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from astropy.stats import median_absolute_deviation
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage import gaussian_filter
import glob
import pickle

def find_cosmic_frames(spectra,ref_frame,clip=3,mad=False,ignore_edges=0,mask=None,verbose=False):
    """A function that uses the standard or median absolute deviation of residuals from each spectrum - a reference spectrum to locate cosmic rays.
    This code first normalizes all inputted spectra for the purposes of accurate comparison, although the normalized
    spectra are only used to locate cosmic rays, they are not returned.

    Inputs:
    spectra - the ndarray of stellar spectra
    clip - the number of stds/mads to clip at. Default=3
    mad - True/False, use the median absolute deviation? Default=False: std is used
    ignore_edges - ignore the edges of the spectra where SNR is low? Define this as the number of data points to ignore. Default = 0
    mask - the masked regions to ignore. Strong absorption lines can be masked, as sometimes these are incorrectly flagged as cosmics. The mask is defined as [range(low_pixel,high_pixel),range(low_pixel,high_pixel)...]
    verbose - True/False - optionally plot example search for first spectrum. Default=False

    Returns:
    the frames and locations of cosmics
    """

    # normalise all spectra
    if ignore_edges == 0:
        spectra = np.array([s/np.nanmedian(s) for s in spectra])
        ref_frame_norm = ref_frame/np.nanmedian(ref_frame)

    else:
        spectra = np.array([s/np.nanmedian(s[ignore_edges:-ignore_edges]) for s in spectra])
        ref_frame_norm = ref_frame/np.nanmedian(ref_frame[ignore_edges:-ignore_edges])


    if not np.any(np.isfinite(spectra)) and ignore_edges == 0:
        return ValueError("all normalised spectra are nans, try increasing the ignore edges parameter")

    if mad:
        if ignore_edges > 0:
            ref_std = median_absolute_deviation(spectra[:,ignore_edges:-ignore_edges]-ref_frame_norm[ignore_edges:-ignore_edges])
        else:
            ref_std = median_absolute_deviation(spectra-ref_frame_norm)
    else:
        if ignore_edges > 0:
            ref_std = np.nanstd(spectra[:,ignore_edges:-ignore_edges]-ref_frame_norm[ignore_edges:-ignore_edges])
        else:
            ref_std = np.nanstd(spectra-ref_frame_norm)

    if mask is not None:
        masked_regions = []
        masked_regions += [i for i in mask]
        masked_regions = sum(masked_regions,[])

    else:
        print('WARNING: Not using a mask, this may throw up absorption lines as cosmics!')

    cosmic_frames = []
    cosmic_pixels = []

    x = np.arange(len(ref_frame_norm))

    for i,f in enumerate(spectra):
        residuals = f-ref_frame_norm
        keep_index = ((residuals >= -clip*ref_std) & (residuals <= clip*ref_std))

        # and now ignore the edges by overwriting keep_index here
        if ignore_edges > 0:
            keep_index[:ignore_edges] = True
            keep_index[-ignore_edges:] = True

        if i == 0 and verbose:
            plt.figure()
            plt.subplot(211)
            plt.plot(f,label="input spectrum")
            plt.plot(ref_frame_norm,label="reference spectrum")
            plt.ylabel("Normalized flux")
            plt.legend()

            plt.subplot(212)
            plt.plot(residuals)
            print("Clipping at %f, %f"%(-clip*ref_std,clip*ref_std))
            plt.axhline(-clip*ref_std,color='gray',ls='--')
            plt.axhline(clip*ref_std,color='gray',ls='--',label="cut off")
            plt.ylabel("Residuals")
            plt.legend()
            plt.xlabel("Pixel number")
            plt.show()

        # Now ignore masked regions from the clip index
        if mask is not None:
            keep_index[masked_regions] = True

        if np.any(~keep_index):
            cosmic_frames.append(i)
            cosmic_pixels.append(np.where(~keep_index)[0])

    return np.array(cosmic_frames),np.array(cosmic_pixels)



def find_cosmic_frames_with_medfilt(data,box_width=7,sigma_clip=5,mask=None,search_region=None,use_mad=True,use_gaussian_filter=False):

    """A function to to find cosmic rays within 1D spectra. This is like find_cosmic_frames but instead of dividing by a reference frame, this function finds cosmics via large differences in the running median.
    This should flag up the locations of comsics as large outliers in the residuals, which are identified via setting the sigma_clip (which removes points that sit away from the mean by a user-defined number of standard deviations).
    Strong absorption lines can be masked, as sometimes these are incorrectly flagged as cosmics. The mask is defined as [range(low_pixel,high_pixel),range(low_pixel,high_pixel)...]

    Inputs:
    data - the ndarray of 1D spectra
    box_width - the number of data points over which to calculate the running median. This must be an odd number and somewhere between 5 and 11 is a good bet. Default=7.
    sigma_clip - the standard deviation at which data points are flagged as cosmic rays. e.g. sigma_clip=5 removes points from the residuals that deviate by > 5 standard deviations. It is better to go with larger values than
    you might think. Default=5.
    mask - the masked regions to ignore (see above)
    search_region - the range (in pixels) over which to search, pixels falling outside this range will be ignored from this search. This is necessary as pixels near the edge can be large outliers and are often ignored when making our wavelength bins later on. This performs better here than find_cosmic_rays
    use_mad - True/False - use the Median Absolute Deviation rather than standard deviation. Default=True but sometimes False is preferable.
    use_gaussian_filter - True/False - instead of using a running median, use a Gaussian filter. Default=False

    Returns:
    np.array(cosmic_frame) - the index (number) of frames/1D spectra which contain cosmic rays
    np.array(cosmic pixels) - the pixel locations corresponding to cosmic rays within each flagged frame/1D spectrum
    """


    cosmic_frames = []
    cosmic_pixels = []

    if mask is not None:
        masked_regions = []
        masked_regions += [i for i in mask]
        masked_regions = sum(masked_regions,[])

    else:
        print('WARNING: Not using a mask, this may throw up absorption lines as cosmics!')

    for i,d in enumerate(data):

        if use_gaussian_filter:
            lowpass = gaussian_filter(d, 3)
            residuals = abs(d - lowpass)
        else:
            MF = medfilt(d,box_width)
            residuals = abs(d - MF)

        if use_mad:
            std = median_absolute_deviation(residuals)

        else:
            std = np.std(residuals)

        possible_cosmics = np.where(residuals > sigma_clip*std)[0]
        if search_region is not None:
            possible_cosmics = sorted(set(possible_cosmics).intersection(search_region))

        if np.any(residuals > sigma_clip*std):

            if mask is not None:
                cosmic = set(possible_cosmics).difference(masked_regions)

            else:
                cosmic = possible_cosmics#.tolist()

            if np.any(cosmic):
                cosmic_frames.append(i)
                cosmic_pixels.append(np.array(sorted(cosmic)))


    return np.array(sorted(set(cosmic_frames))),np.array(cosmic_pixels)


def check_cosmic_frames(spectra,frame_array,cosmic_positions=None,single_plot=True):

    """A sanity check that will check the output of find_cosmic_frames and find_cosmic_frames_with_medfilt. Plots all flagged cosmics overlaid on the 1D spectra.

    Inputs:
    spectra - the array of all 1D spectra with and without cosmic rays
    frame_array - the array of frame indices corresponding to the frames/1D spectra that have been flagged as containing cosmic rays
    cosmic_positions - the ndarray of pixel locations of flagged cosmics
    single_plot - Plot all flagged spectra and cosmics on a single figure, otherwise create new figure for each cosmic. Default=True

    Returns:
    Nothing - only plots figures"""

    if single_plot:
        plt.figure()

    for i,f in enumerate(frame_array):
        if not single_plot:
            plt.figure()
        plt.plot(spectra[f],'k',alpha=0.75)
        if cosmic_positions is not None:
            # ~ for c in cosmic_positions[i]:
                # ~ plt.axvline(c,color='r')
            plt.plot(cosmic_positions[i],spectra[f][cosmic_positions[i]],'rx',ms=10)
        plt.ylabel('Integrated counts')
        plt.xlabel('Y pixel')
        if not single_plot:
            plt.title("Frame "+str(f))
            plt.show()

    if single_plot:
        plt.show()
    return

def consecutive(data, stepsize=1):
    """A function used by replace_cosmics to relace pixels that are flagged more than once."""
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def replace_cosmics(spectra,errors,cosmic_frames,cosmic_positions,replace_with_nans=False):

    """Replaces the cosmic pixels by a linear fit between the 2 nearest, unaffected, pixels.

    Inputs:
    spectra - ndarray of all 1D spectra, with and without cosmics flagged
    errors - ndarray of 1D spectra's errors
    cosmic_frames - the array of frames flagged as containing cosmics
    cosmic_positions - ndarray of locations of pixels flagged as cosmics
    replace_with_nans - if wanting to replace the cosmic rays with nans instead of interpolation use this.

    Returns:
    np.array(clean_flux) - the ndarray of 1D spectra following removal of the cosmics
    np.array(clean_error) - the ndarray of 1D errors following removal of the cosmics"""

    clean_flux = []
    clean_err = []

    nspectra = len(spectra)

    for i in range(nspectra):

        if i not in cosmic_frames:

            clean_flux.append(spectra[i])
            clean_err.append(errors[i])

        else:

            k = cosmic_frames.tolist().index(i)

            cosmic_spectra = spectra[i]
            cosmic_errors = errors[i]
            pixels = cosmic_positions[k]


            if replace_with_nans:
                cosmic_spectra[pixels] = np.nan
                cosmic_errors[pixels] = np.nan

                cleaned_spectra = cosmic_spectra
                cleaned_errors = cosmic_errors

            else:
                unique_cosmics = consecutive(pixels)

                cleaned_spectra = cosmic_spectra.copy()
                cleaned_errors = cosmic_errors.copy()

                for j in unique_cosmics:

                    # ignore cosmic pixels right at the edge of the spectra
                    if max(j)+1 >= len(cleaned_spectra):
                        j = j[:-1]
                        if len(j) == 0:
                            continue

                    poly_flux = np.poly1d(np.polyfit([min(j)-1,max(j)+1],[cleaned_spectra[min(j)-1],cleaned_spectra[max(j)+1]],1))
                    cleaned_spectra[min(j)-1:max(j)+1] = poly_flux(np.arange(min(j)-1,max(j)+1))

                    poly_err = np.poly1d(np.polyfit([min(j)-1,max(j)+1],[cleaned_errors[min(j)-1],cleaned_errors[max(j)+1]],1))
                    cleaned_errors[min(j)-1:max(j)+1] = poly_err(np.arange(min(j)-1,max(j)+1))

            clean_flux.append(cleaned_spectra)
            clean_err.append(cleaned_errors)

    return np.array(clean_flux),np.array(clean_err)


def not_cosmics(cosmic_flagged_frames,cosmic_flagged_pixels,not_cosmic_list):

    """Ignore incorrectly flagged frames and associated pixels.

    Inputs:
    cosmic_flagged_frames - the ndarray of 1D spectra flagged as containing cosmic rays
    cosmic_flagged_pixels - the ndarray of the pixel locations of flagged cosmic rays
    not_cosmic_list - the array of flagged frames which are do not correspond to real cosmic frames

    Returns:
    np.array(cosmic_flagged_frames) - array of 1D spectra that contain cosmic frames after removing incorrectly flagged frames
    np.array(cosmic_flagged_pixels) - array of pixels that contain cosmic frames after removing incorrectly flagged frames"""

    mask = np.in1d(cosmic_flagged_frames,not_cosmic_list)

    #invert the mask
    mask = ~mask

    return cosmic_flagged_frames[mask],cosmic_flagged_pixels[mask]


def interp_bad_pixels(frame,pixel_mask,return_nans=False,replace_with_medians=False):
    """Interpolate over bad pixels in a frame using a 2D interpolation over neighbouring pixels.

    Inputs:
    frame - the frame (2D array) of values
    pixel_mask - the bad pixel mask (same shape as frame)
    returns_nans - True/False - if True return nans at bad pixel locations, otherwise interpolate over them. Default=False.
    replace_with_medians - True/False - if True, replace the bad pixel with the median of the surrounding pixels, as opposed to interpolating with a Gaussian kernel

    Returns:
    frame - with bad pixels interpolated over"""

    nrows,ncols = frame.shape
    for i in range(nrows):
        if replace_with_medians:
            for j in range(ncols):
                if pixel_mask[i][j]:
                    surrounding_pixels = []

                    if j > 0: # lett pixel
                        if not pixel_mask[i][j-1]: # make sure we're not using a bad pixel
                            surrounding_pixels.append(frame[i][j-1])
                        else:
                            if j > 1:
                                if not pixel_mask[i][j-2]:
                                    surrounding_pixels.append(frame[i][j-2])


                    if j < ncols-1: # right pixel
                        if not pixel_mask[i][j+1]: # make sure we're not using a bad pixel
                            surrounding_pixels.append(frame[i][j+1])
                        else:
                            if j < ncols-2:
                                if not pixel_mask[i][j+2]: # make sure we're not using a bad pixel
                                    surrounding_pixels.append(frame[i][j+2])


                    # if i > 0: # lower pixel
                    #     if not pixel_mask[i-1][j]: # make sure we're not using a bad pixel
                    #         surrounding_pixels.append(frame[i-1][j])
                    #
                    # if i < nrows-1: # upper pixel
                    #     if not pixel_mask[i+1][j]: # make sure we're not using a bad pixel
                    #         surrounding_pixels.append(frame[i+1][j])

                    frame[i][j] = np.nanmedian(surrounding_pixels)

        else:
            frame[i][pixel_mask[i]] = np.nan

    if return_nans or replace_with_medians:
        return frame

    # Now use astropy to convolve bad pixels with neighbouring values
    # We smooth with a Gaussian kernel with x_stddev=1 (and y_stddev=1)
    # It is a 9x9 array
    kernel = Gaussian2DKernel(x_stddev=1)

    # create a "fixed" image with NaNs replaced by interpolated values
    frame = interpolate_replace_nans(frame, kernel)
    return frame



def flag_bad_pixels(frame,cut_off=5,max_pixels_per_row=10,plot_rows=None,use_mad=False,verbose=False,mf_box_width=3,left_col=0,right_col=-1,axis=None,std_box_width=0,use_gaussian_filter=False,existing_pixel_mask=None):
    """A function that performs a row-by-row running median to locate bad pixels in a given frame.

    Inputs:
    frame - the (nrows,ncols) frame of image data under consideration
    cut_off - the sigma/mad clip at which outliers are flagged. Default=5 (5 sigma)
    max_pixels_per_row - the maximum number of bad pixels in a row to consider. If the number of flagged pixels is greater than this number, it's assumed something went wrong and no pixels are flagged for this row. Default=10
    plot_rows - give a list of rows at which the running median will be plotted. If not given, then no running median will be plotted. Default=None.
    use_mad - True/False - use the running median rather than the standard deviation of the residuals to flag outliers? Default=False
    verbose - True/False - if True plot the resulting pixel map
    mf_box_width - the number of pixels over which to calculate the running median. Default=3
    left_col - set the leftmost column to consider. Default = 0, i.e. column 0
    right_col - set the rightmost column to consider. Default = -1, i.e. last column
    axis - set whether to calculate one std/mad across the whole frame (axis=None) or calculae a std/mad for every row (axis=1). Default=None
    std_box_width - set this to have a sliding standard deviation across the row, rather than a single standard deviation. Set this number equal to the number of points over which to calculate the sliding standard deviation. Default=0.
    use_gaussian_filter - set this to True to use a Gaussian filter rather than running median to locate outliers. Default=False.
    existing_pixel_mask - use this to pass an existing pixel mask to append newly flagged bad pixels to. Default=None

    Returns:
    pixel_flags - the boolean (nrows,ncols) array of flagged bad pixels"""

    pixel_flags_all = np.zeros_like(frame).astype(bool)
    pixel_flags = np.zeros_like(frame[:,left_col:right_col])

    nrows,ncols = frame[:,left_col:right_col].shape

    residuals = []

    if use_gaussian_filter:
        median = np.array([gaussian_filter(row, 3) for row in frame[:,left_col:right_col]])
        # interpolate over nan values
        for i in range(nrows):
            finite_pixels = np.isfinite(frame[:,left_col:right_col][i])
            infinite_pixels = ~finite_pixels
            finite_medians = np.isfinite(median[i])
            infinite_medians = ~finite_pixels
            if np.any(finite_pixels):
                median[i] = np.interp(np.arange(ncols),np.arange(ncols)[finite_medians],median[i][finite_medians])
            else:
                median[i] = np.nan*finite_medians

        residuals =  median - frame[:,left_col:right_col]

    else:
        median = np.array([medfilt(row,mf_box_width) for row in frame[:,left_col:right_col]])
        residuals = median - frame[:,left_col:right_col]

    if use_mad:
        if std_box_width > 0:
            threshold_array = []
            for i in range(nrows):
                threshold_array.append(np.array([median_absolute_deviation(residuals[i][j:j+std_box_width],ignore_nan=True) for j in range(0,ncols,std_box_width)]))
            threshold = np.zeros_like(residuals)
            for i in range(nrows):
                for j,k in enumerate(range(0,ncols,std_box_width)):
                    threshold[i][k:k+std_box_width] = threshold_array[i][j]
        else:
            threshold = median_absolute_deviation(residuals,ignore_nan=True,axis=axis)
    else:
        if std_box_width > 0:
            threshold_array = []
            for i in range(nrows):
                threshold_array.append(np.array([np.nanstd(residuals[i][j:j+std_box_width]) for j in range(0,ncols,std_box_width)]))
            threshold = np.zeros_like(residuals)

            for i in range(nrows):
                for j,k in enumerate(range(0,ncols,std_box_width)):
                    threshold[i][k:k+std_box_width] = threshold_array[i][j]
        else:
            threshold = np.nanstd(residuals,axis=axis)


    if axis == 1 and std_box_width == 0:
        threshold = np.ones_like(median)*threshold.reshape(nrows,1)
    else:
        threshold = np.ones_like(median)*threshold

    good_pixels = ((residuals <= cut_off*threshold) & (residuals >= -cut_off*threshold))
    bad_pixels = ~good_pixels
    if existing_pixel_mask is not None:
        bad_pixels += existing_pixel_mask[:,left_col:right_col].astype(bool)

    for i,row in enumerate(frame[:,left_col:right_col]):

        bad_pixels[i][~np.isfinite(row)] = True

        # in order to remove rows where lots of pixels are flagged (probably due to the stellar spectrum), ignore these rows
        if len(np.where(bad_pixels[i])[0]) > max_pixels_per_row:
            bad_pixels[i] = np.zeros_like(good_pixels[i]).astype(bool)

        pixel_flags[i] = bad_pixels[i]

        if plot_rows is not None:

            if i in plot_rows:

                plt.figure()
                plt.subplot(211)
                plt.plot(row,label="Data")
                plt.plot(median[i],label="Running median")
                plt.plot(np.arange(ncols)[bad_pixels[i]],row[bad_pixels[i]],"rx",label="Clipped point")
                # plt.yscale("log")
                plt.ylabel("Counts")
                plt.title("Pixel flagging, row %d"%i)
                plt.legend()

                plt.subplot(212)
                plt.plot(residuals[i])
                plt.plot(np.arange(ncols)[bad_pixels[i]],residuals[i][bad_pixels[i]],"rx",label="Clipped point")

                if std_box_width == 0:
                    plt.axhline(cut_off*threshold[i].mean(),ls='--',color='k')
                    plt.axhline(-cut_off*threshold[i].mean(),ls='--',color='k')
                else:
                    plt.plot(cut_off*threshold[i],ls='--',color='k')
                    plt.plot(-cut_off*threshold[i],ls='--',color='k')
                plt.ylabel("Residuals")
                plt.show()

    pixel_flags = pixel_flags.astype(bool)

    if verbose:
        plt.figure()
        plt.imshow(pixel_flags,aspect="auto")
        plt.xlabel("Pixel column")
        plt.ylabel("Pixel row")
        plt.title("Bad pixel map")
        plt.show()

    pixel_flags_all[:,left_col:right_col] = pixel_flags

    return pixel_flags_all.astype(bool)

def gif_init(i,im,data):
    """Function used by animation via the gif() function to generate gifs"""
    im.set_data(data[i])
    print("Saving frame %d to gif"%i)
    return im


def gif(data,filename):
    """
    Function that can save multidimensional frames (nframes,nrows,ncols) as a .gif video.

    Inputs:
    data - the (nframes,nrows,ncols) frames
    filename - the string to which to save the video as (note .gif is added by this function)

    Returns:
    None, just saves the video"""

    import matplotlib.animation as animation

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(data[0])
    nframes = len(data)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    ax.set_title(filename)

    anim = animation.FuncAnimation(fig, gif_init, frames=nframes,fargs=(im,data),repeat=True,save_count=0)

    f = r"%s.gif"%filename
    writervideo = animation.PillowWriter(fps=10)
    anim.save(f, writer=writervideo)


def flag_all_bad_pixels(frames,cut_off=5,max_pixels_per_row=10,use_mad=False,save_gif=False,gif_name=None):
    """A function that runs flag_bad_pixels() to N frames.

    Inputs:
    frames - the (nframes,nrows,ncols) frames of image data under consideration
    cut_off - the sigma/mad clip at which outliers are flagged. Default=5 (5 sigma)
    max_pixels_per_row - the maximum number of bad pixels in a row to consider. If the number of flagged pixels is greater than this number, it's assumed something went wrong and no pixels are flagged for this row. Default=10
    use_mad - True/False - use the running median rather than the standard deviation of the residuals to flag outliers? Default=False
    save_gif - True/False - if True save a video of all pixel maps. Default=False
    gif_name - If save_gif is True, give the name of the file you want to save the gif to here (excluding the .git suffix)

    Returns:
    all_bad_pixels - the (nframes,nrows,ncols) boolean bad pixel arrays
    persistent_pixels - those pixels that have been flagged as bad in >50% of the frames.
    """

    all_bad_pixels = np.zeros_like(frames)
    nframes = len(frames)

    for i,f in enumerate(frames):
        print("Flagging bad pixels for frame %d" %(i+1))
        all_bad_pixels[i] = flag_bad_pixels(f,cut_off=cut_off,use_mad=use_mad)

    master_pixel_map = all_bad_pixels.astype(int).sum(axis=0)
    persistent_pixels = master_pixel_map > nframes/2

    if save_gif:

        gif(all_bad_pixels)

    return all_bad_pixels,persistent_pixels


def combine_masters(seg_masters):
    """A function that combines multiple bad pixel frames (created for each segment) into one global master bad pixel file.

    The final global master bad pixel file is created with any pixels that have been flagged in > 1 of the individual master bad pixel files.

    Inputs:
    seq_masters - a list of the individual master bad pixel files for each segment

    Returns:
    global_master - the single (nrows, ncols) global bad pixel frame as a boolean"""


    summed_masters = np.sum(np.array(seg_masters).astype(int),axis=0)

    # For the global master I only take pixels that have been flagged in more than 1 of the separate masters
    bad_pixels = np.zeros_like(summed_masters)
    bad_pixels[summed_masters > 1] = 1

    global_master = bad_pixels.astype(bool)

    plt.figure()
    plt.xlabel("Pixel column")
    plt.ylabel("Pixel row")
    plt.imshow(global_master)
    plt.title("Global bad pixel master")
    plt.show()

    return global_master



def load_segments(file_type):
    """A function that loads in file types from all segments, assuming you're working in ../reductions_notebooks/ and ls ../ shows seg???/pickled_objects/

    Inputs:
    file_type - star1_flux.pickle / star1_error.pickle / xpos1.pickle etc.

    Returns:
    the concatenated arrays from all separate segments for that particular file type"""

    segment_direcs = sorted(glob.glob("../seg???/pickled_objects/"))

    s1 = pickle.load(open("%s/%s"%(segment_direcs[0],file_type),"rb"))

    combined_arrays = s1

    ndimensions = len(np.shape(s1))

    for i in segment_direcs[1:]:
        if ndimensions > 1:
            combined_arrays = np.vstack((combined_arrays,pickle.load(open("%s/%s"%(i,file_type),"rb"))))
        else:
            combined_arrays = np.hstack((combined_arrays,pickle.load(open("%s/%s"%(i,file_type),"rb"))))

    return combined_arrays


def extract_dq_flags(dq_cube, bits_to_mask=[0, 1, 10, 11]):

    """This function locates given bad pixel flags within the 2D DQ arrays associated with JWST fits files.

    This borrows from Lili Alderson's very helpful implementation of this in the ExoTiC-JEDI reduction pipeline.

    The default behaviour is to mask the following DQ flags:
        Bit = 0, bad pixel, do not use
        Bit = 1, saturated pixel
        Bit = 10, dead pixel
        Bit = 11, hot pixel

    The full set of flags can be found at https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#:~:text=is%20strongly%20discouraged.-,Data%20Quality%20Flags,-Within%20science%20data)

    Inputs:
    dq_cube -- the ndarray of DQ flags from a JWST fits file (equivalent to the 'DQ' fits extentsion in said file)
    bits_to_mask -- the pixel flags that you want to mask. Default = [0,1,10,11]

    Returns:
    new_dq_cube -- the new ndarray of pixel flags that only correspond to the bad pixels you're interested in.
    """

    flags_time = np.where(dq_cube!=0)[0] # finding where the pixels have a data quality flag
    flags_y = np.where(dq_cube!=0)[1]
    flags_x = np.where(dq_cube!=0)[2]

    new_dq_cube = np.zeros_like(dq_cube)

    counter=0

    for i in range(len(flags_time)):

        print("working on integration %d"%(i+1))

        hiti = flags_time[i]
        hity = flags_y[i]
        hitx = flags_x[i]

        binary_sum = dq_cube[flags_time[i] , flags_y[i] , flags_x[i]]

        bit_array = np.flip(np.array(list(np.binary_repr(binary_sum, width=32))).astype(int))

        if np.any(bit_array[bits_to_mask] == 1):

            new_dq_cube[hiti, hity, hitx] = 1

            counter+=1

    return(new_dq_cube)
