from astropy.io import fits
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Override the DQ saturation flags in the FITS headers after the saturation step and before the superbias")
parser.add_argument('fits_file',help='the name of the fits file to load in')
parser.add_argument('--sat_mask',help='optionally load in a master DQ saturation mask, like for W39')
args = parser.parse_args()


f = fits.open(args.fits_file)

extension = "GROUPDQ"

ngroups,nintegrations,nrows,ncols = f[extension].data.shape

if args.sat_mask is not None:
    import pickle
    sat_mask = pickle.load(open(args.sat_mask,"rb"))
else:
    sat_mask = None

for g in range(ngroups):

    group_dq = f[extension].data[g]
    group_flux = f[1].data[g]

    for i in range(nintegrations):

        if sat_mask is None:

            saturated_flags = group_dq[i].sum(axis=0).astype(bool)

            saturated_cols = np.sum(group_flux[i] > 0.9*65535,axis=0).astype(bool)

            saturated_cols = saturated_cols+saturated_flags

            f[extension].data[g][i][:,saturated_cols.astype(bool)] = 2

        else:
            f[extension].data[g][i] = sat_mask[i]

# ~ new_name = args.fits_file.split(".")[0]+"_newDQ.fits"
new_name = args.fits_file

f.writeto(new_name,overwrite=True)
