# Manually implementing the gain scale step since this is skipped by strun gainscalestep.

from astropy.io import fits
import argparse
import copy

parser = argparse.ArgumentParser(description="apply the gain correction to convert from DN/s to e-")
parser.add_argument('--science_file',help='the name of the science fits file to load in')
parser.add_argument('--gain_file',help='the name of the gain fits file to load in')
args = parser.parse_args()

science_file = fits.open(args.science_file)
gain_file = fits.open(args.gain_file)

print("loading %s"%args.science_file)

gain_factor = gain_file["SCI"].data[gain_file["SCI"].data > 0].mean()

new_science_file = copy.deepcopy(science_file)

new_science_file["SCI"].data *= gain_factor
new_science_file["ERR"].data *= gain_factor
new_science_file["VAR_POISSON"].data *= gain_factor**2
new_science_file["VAR_RNOISE"].data *= gain_factor**2

new_science_file[0].header["S_GANSCL"] = "COMPLETE"
new_science_file[0].header["BUNIT"] = "e/S"

new_science_filename = args.science_file.split("_1")[0] + "_1_gainscalestep.fits"

print("saving to %s"%new_science_filename)

new_science_file.writeto(new_science_filename,overwrite=True)
