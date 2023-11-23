import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse
from scipy.stats import median_abs_deviation as mad
from scipy.ndimage import median_filter as MF
import pickle

parser = argparse.ArgumentParser(description='create master darks, flats and arcs')
parser.add_argument("-dl","--dark_list",help="list of darks",type=str)
parser.add_argument("-fl","--flat_list",help="list of flats",type=str)
parser.add_argument("-al","--arc_list",help="list of arcs",type=str)
parser.add_argument('-c','--clobber',help="""Need this argument to save resulting fits file, default = False""",action='store_true')
parser.add_argument('-v','--verbose',help="""Display the image of each bias frame before combining it.""",action='store_true')
args = parser.parse_args()

verbose = args.verbose

dark_list = np.loadtxt(args.dark_list,dtype='str')
flat_list = np.loadtxt(args.flat_list,dtype='str')
arc_list = np.loadtxt(args.arc_list,dtype='str')

def combine_frames(file_list,verbose,master_dark=None):

    if verbose:
        plt.figure()

    frames = []

    for i,f in enumerate(file_list):

        frame = fits.open(f)

        if i == 0:
            hdr = frame[0].header

        if master_dark is not None:
            frames.append(frame[0].data - master_dark)
        else:
            frames.append(frame[0].data)

        if verbose:
            plt.title('#%d/%d ; %s'%(i,len(file_list)-1,f.split("/")[-1]))
            plt.imshow(frame[0].data,vmin=np.median(frame[0].data)*0.99,vmax=np.median(frame[0].data)*1.01)
            plt.colorbar()
            plt.show(block=False)
            plt.pause(1.0)
            plt.clf()

        frame.close()

    master_frame = np.median(frames,axis=0)

    return master_frame,hdr

def save_fits(data,header,filename,clobber=False):
    hdu = fits.PrimaryHDU(data,header=header)
    hdu.writeto(filename,overwrite=clobber)
    return

def pixel_mask_tight(frame,clobber):
    """A tighter definition of a bad pixel mask using 5 median absolute devitations from the median, computed column by column (along dispersion direction)"""
    medians = np.median(frame,axis=0)
    mads = mad(frame,axis=0)
    good_pixels = []

    for i,row in enumerate(frame):
        good_pixels.append(((row >= medians-5*mads) & (row <= medians+5*mads)))

    good_pixels = np.array(good_pixels)
    bad_pixels = ~good_pixels

    plt.figure()
    plt.imshow(bad_pixels)
    plt.title("Bad pixel mask using medians and MADS")
    plt.show()

    pickle.dump(good_pixels,open("good_pixel_mask_tight.pickle","wb"))
    pickle.dump(bad_pixels,open("bad_pixel_mask_tight.pickle","wb"))
    # ~ save_fits(good_pixels,"good_pixel_mask_tight.fits",clobber)
    # ~ save_fits(bad_pixels,"bad_pixel_mask_tight.fits",clobber)

    return


def pixel_mask_loose(frame,clobber):
    """A looser definition of the bad pixel mask using 5 standard deviations from the global mean"""

    good_pixels = ((frame >= np.mean(frame)-5*np.std(frame)) & (frame <= np.mean(frame)+5*np.std(frame)))
    bad_pixels = ~good_pixels

    plt.figure()
    plt.imshow(bad_pixels)
    plt.title("Bad pixel mask using means and stds")
    plt.show()

    pickle.dump(good_pixels,open("good_pixel_mask_loose.pickle","wb"))
    pickle.dump(bad_pixels,open("bad_pixel_mask_loose.pickle","wb"))
    # ~ save_fits(good_pixels,"good_pixel_mask_loose.fits",clobber)
    # ~ save_fits(bad_pixels,"bad_pixel_mask_loose.fits",clobber)

    return good_pixels,bad_pixels

def normalize_flat(frame):

	"""Not in use yet, needs more testing."""

	nrows,ncols = np.shape(frame)

	polynomial_fits = []

	for i in range(ncols):
		poly = np.poly1d(np.polyfit(np.arange(100,nrows-100),frame[:,i][100:nrows-100],5))
		polynomial_fits.append(poly(np.arange(nrows)))

	normalized_flat = frame/np.array(polynomial_fits).T

	return normalized_flat


def mask_order_gaps(flat):
	"""Replace gaps between orders in flat with np.nan. This replaces zeroes which when divided through give +inf in the science frames."""
	masked_flat = flat.copy()

	for i in range(2048):
		residuals = masked_flat[i] - MF(masked_flat[i],101)
		outliers = (residuals <=  mad(residuals)*-10)
		masked_flat[i][outliers] = np.nan

	return masked_flat


master_dark,dark_header = combine_frames(dark_list,verbose,None)
master_flat,flat_header = combine_frames(flat_list,verbose,None)
masked_flat = mask_order_gaps(master_flat)
master_arc,arc_header = combine_frames(arc_list,verbose,None)

save_fits(master_dark,dark_header,'master_dark.fits',args.clobber)
save_fits(master_flat/np.nanmedian(master_flat),flat_header,'master_flat.fits',args.clobber)
save_fits(masked_flat/np.nanmedian(masked_flat),flat_header,'master_flat_order_gaps_masked.fits',args.clobber)
save_fits(master_arc,arc_header,'master_arc.fits',args.clobber)



pixel_mask_tight(master_dark,args.clobber)
pixel_mask_loose(master_dark,args.clobber)
