from astropy.io import fits
import numpy as np
import argparse
import pickle
from pathlib import Path
from global_utils import parseInput

home = str(Path.home())

parser = argparse.ArgumentParser(description="subtract 1/f noise at the group stage")
parser.add_argument('fits_file',help='the name of the fits file to load in')
parser.add_argument('--pixel_mask',help='the name of the bad pixel mask file to load in')
parser.add_argument("--trace_location",help="optionally parse the locations of the stellar trace to define where the background should be estimated")
parser.add_argument("--extraction_input",help="optionally parse the extraction_input.txt file associated with the trace_location file to determine the width of the background region")
args = parser.parse_args()

f = fits.open(args.fits_file)

try:
    bias_name = f[0].header["R_SUPERB"].split("//")[1]
except:
    bias_name = f[0].header["R_SUPERB"]

instrument = f[0].header["INSTRUME"].lower()
super_bias = fits.open("%s/crds_cache/jwst_pub/references/jwst/%s/%s"%(home,instrument,bias_name))

extension = 1
nints,ngroups,nrows,ncols = f[extension].data.shape
bkg_mask = np.ones((nrows,ncols))

if args.trace_location is not None:
    input_dict = parseInput(args.extraction_input)

    trace_centre = nrows-np.round(pickle.load(open(args.trace_location,"rb"))/int(input_dict["oversampling_factor"])).astype(int)[0]


    col_min = int(input_dict["row_min"])
    col_max = int(input_dict["row_max"])

    # here I'm assuming that any stellar spectra above/below row_min/row_max is too low SNR to throw off the median. In any case, we won't use these rows in the end
    for i,col in enumerate(range(col_min,col_max)):
        ## set non-background rows to zero in our mask
        bkg_mask[:,col][trace_centre[i]: trace_centre[i] + int(input_dict["aperture_width"])//2 + int(input_dict["background_offset"])] = 0
        bkg_mask[:,col][trace_centre[i] - int(input_dict["aperture_width"])//2 - int(input_dict["background_offset"]): trace_centre[i]] = 0

    if input_dict["bad_pixel_mask"] != "":
        pixel_mask = pickle.load(open(input_dict["bad_pixel_mask"],"rb"))
        bkg_mask[pixel_mask] = 0


else:
    bottom_bkg_row = 5
    top_bkg_row = -5

    ## set non-background rows to zero in our mask
    bkg_mask[bottom_bkg_row:top_bkg_row] = 0

if args.pixel_mask is not None:
    pixel_mask = pickle.load(open(args.pixel_mask,"rb"))
    # now exclude bad pixels from our background mask
    bkg_mask[pixel_mask] = 0

one_over_f_noise = []
bias_bkg = []

for i in range(nints):

    group_flux = f[extension].data[i]

    for g in range(ngroups):

        print("working on integration %d, group %d"%(i,g))

        image = group_flux[g]

        bkg = []

        for c in range(ncols):

            bkg_pixels = image[:,c][bkg_mask[:,c].astype(bool)]

            col_median = np.median(bkg_pixels)

            bkg.append(col_median)

            if i == 0 and g == ngroups - 1:

                bias_bkg_pixels = super_bias[1].data[:,c][bkg_mask[:,c].astype(bool)]

                bias_col_median = np.median(bias_bkg_pixels)

                bias_bkg.append(bias_col_median)

        if g == ngroups - 1:
            one_over_f_noise.append(np.array(bkg))

        image = image - np.array(bkg)

        f[extension].data[i][g] = image

pickle.dump(np.array(one_over_f_noise),open("one_over_f_noise.pickle","wb"))
pickle.dump(np.array(bias_bkg),open("bias_level.pickle","wb"))
scale_factor = np.array(bias_bkg)/np.array(one_over_f_noise)
scale_factor[~np.isfinite(scale_factor)] = 0
scale_factor_time_series = scale_factor.mean(axis=1)
scale_factor_time_series_standardized = (scale_factor_time_series-scale_factor_time_series.mean())/scale_factor_time_series.std()
pickle.dump(scale_factor_time_series_standardized,open("bias_scale_factor.pickle","wb"))

# new_name = args.fits_file.split(".")[0]+"_1overf_JK.fits"
new_name = args.fits_file
f.writeto(new_name,overwrite=True)
