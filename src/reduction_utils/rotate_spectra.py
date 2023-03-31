#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk 

import numpy as np
from astropy.io import fits
import glob
import os
import copy
import argparse

parser = argparse.ArgumentParser(description='Rotate fits files so that the dispersion direction is along the vertical axis. This will save the rotated .fits files to "./spec_rot/*_rot.fits"')
parser.add_argument('-fl','--file_list',help='Specify which files to read in or optionally (Default) read in all *.fits files within current working directory',nargs='+')
args = parser.parse_args()

if args.file_list is None:
	files_in = sorted(glob.glob("*.fits"))
else:
	files_in = args.file_list

try:
    os.mkdir("spec_rot")
except:
    pass
    # raise SystemExit

for i in files_in:

    print(i)

    f = fits.open(i)

    f_new = copy.deepcopy(f)

    f_new[0].data = np.flip(f[0].data.T,axis=1)

    new_file_name = "spec_rot/"+i[:-5]+"_rot.fits"

    f_new.writeto(new_file_name,overwrite=True)

    f.close()
