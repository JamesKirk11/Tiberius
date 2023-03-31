from astropy.io import fits
import numpy as np

def save_new_fits(data,filename):
    """A function to save a new fits file from a single image for use with 1/f corrections.

    Inputs:
    data - the 2d array of image data
    filename - the string which dictates where the fits file should be saved to

    Returns
    Nothing, just a saved FITS file at the requested filename location"""

    # make dummy header
    new_header = fits.Header()
    new_header["int_mid_BJD_TDB"] = 0
    new_header["TGROUP"] = 0

    # make science extension
    hdu1 = fits.PrimaryHDU([data],header=new_header) # science array
    hdu1.name = "SCI"

    # make dummy error extension
    hdu2 = fits.ImageHDU(np.zeros_like([data]))
    hdu2.name = "ERR"

    # make dummy integration extension
    c3 = fits.Column(name='int_mid_BJD_TDB', array=np.array([0.]), format='K')
    hdu3 = fits.BinTableHDU.from_columns([c3])
    hdu3.name = "INT_TIMES"

    hdu = fits.HDUList([hdu1,hdu2,hdu3])
    print("saving %s"%filename)
    hdu.writeto(filename,overwrite=True)
    return hdu
