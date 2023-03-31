#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

from astropy.io import fits
import os
import crds
import argparse
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

home = str(Path.home())
os.environ["CRDS_PATH"]="%s/crds_cache/jwst_pub"%home
os.environ["CRDS_SERVER_URL"]="https://jwst-crds-pub.stsci.edu"

parser = argparse.ArgumentParser(description='Expand a JWST rateints.fits file into multiple .fits files for each integration, saved to a new directory called science_fits_files/')
parser.add_argument('fits_files', help="the name of the fits file(s) to be expanded. Typical use: $ python expand_JWST_fits.py *rateints.py",nargs="+")
args = parser.parse_args()

try:
    os.mkdir("science_fits_files")
except:
    pass

def load_correct_files(fits_file_names):

    initial_fits_files = np.array(sorted(fits_file_names))
    initial_headers = [fits.getheader(i) for i in sorted(fits_file_names)]
    exp_type = np.array([i["EXP_TYPE"] for i in initial_headers])
    print("\nExposure types = ",exp_type)

    # check that we're only using science images
    if "NIRISS" in initial_headers[0]["INSTRUME"]:
        science_files = np.where(exp_type=="NIS_SOSS")[0]

    if "NIRSPEC" in initial_headers[0]["INSTRUME"]:
        science_files = np.where(exp_type=="NRS_BRIGHTOBJ")[0]

    if "MIRI" in initial_headers[0]["INSTRUME"]:
        science_files = np.where(exp_type=="MIR_LRS-SLITLESS")[0]

    print("\nLoading in ",initial_fits_files[science_files])

    return initial_fits_files[science_files]


print("\nChecking headers for ",sorted(args.fits_files))

fits_files = load_correct_files(args.fits_files)
f_in = fits.open(fits_files[0])
nfiles = len(fits_files)

# extensions: 0 (header), 1 (science), 2 (error), 3 (data quality flag), 4 (integration times), 5 (poisson variance), 6 (readnoise variance), 7 (asdf, parameter input file)

header = f_in[0].header

_,nrows,ncols = f_in["SCI"].data.shape

parameters = {"meta.ref_file.crds.context_used":header["CRDS_CTX"],"meta.ref_file.crds.sw_version":header["CRDS_VER"],\
              "meta.instrument.name":header["INSTRUME"].lower(),"meta.instrument.detector":header["DETECTOR"],"meta.observation.date":header["DATE-OBS"],\
              "meta.observation.time":header["TIME-OBS"],"meta.exposure.type":header["EXP_TYPE"]}

def load_ref_file(parameters,file_type,header):

    short_ref_file_name = crds.getrecommendations(parameters, reftypes=[file_type], context=None, ignore_cache=False, observatory="jwst", fast=False)
    print(short_ref_file_name)

    if "NOT FOUND" in short_ref_file_name[file_type]:
        print("**%s file not found, skipping**"%file_type)
        return

    try:
        long_ref_file_name = crds.getreferences(parameters, reftypes=[file_type], context=None, ignore_cache=False, observatory="jwst", fast=False)
        print("ref file = ",long_ref_file_name)
    except:
        print("**%s file not read, skipping"%file_type)
        return

    if long_ref_file_name[file_type].split(".")[-1] == "fits":
        ref_file = fits.open(long_ref_file_name[file_type])
        ref_file_copy = copy.deepcopy(ref_file)
        ref_file_copy[1].data = ref_file_copy[1].data[header["SUBSTRT2"]-1:header["SUBSTRT2"]-1+header["SUBSIZE2"],header["SUBSTRT1"]-1:header["SUBSTRT1"]-1+header["SUBSIZE1"]]
        ref_file_copy.writeto("science_fits_files/%s"%short_ref_file_name[file_type],overwrite=True)
        return ref_file_copy
    else:
        return


def generate_wavelength_solution(instrument,filt,npixels,first_col,last_col):
    print("using hard-coded wavelengths")

    if instrument == "nirspec" and filt == "CLEAR":
        # the wavelength range for PRISM, from https://jwst-crds.stsci.edu/browse/jwst_nirspec_wavelengthrange_0005.asdf
        w_min = 6.0e-07*1e6 # in microns
        w_max = 5.3e-06*1e6 # in microns

    if instrument == "niriss":
        print("wavelength calibration needs improving!")
        w_min = 0.6
        w_max = 2.8

    wvl_solution = np.linspace(w_min,w_max,npixels)[first_col:last_col]

    return wvl_solution




gain = load_ref_file(parameters,"gain",header)
flat = load_ref_file(parameters,"dflat",header)
bias = load_ref_file(parameters,"superbias",header)
wavelengthrange = load_ref_file(parameters,"wavelengthrange",header)

# if wavelengthrange is None:
#     wvl_solution = generate_wavelength_solution(header["INSTRUME"].lower(),header["FILTER"],2048,header["SUBSTRT1"]-1,header["SUBSTRT1"]-1+header["SUBSIZE1"])
#     pickle.dump(wvl_solution,open("science_fits_files/wavelength_solution.pickle","wb"))

readnoise = load_ref_file(parameters,"readnoise",header)
pickle.dump(f_in[3].data,open("./science_fits_files/pixel_data_quality_flags.pickle","wb"))

count = 0
time = []
mjd_time = []

for i in range(nfiles):

    in_file = fits.open(fits_files[i])

    nint,_,_ = np.shape(in_file["SCI"].data)

    for j in range(nint):

        new_header = hdr = fits.Header()
        new_header["TGROUP"] = header["TGROUP"]

        try:
            new_header["int_mid_BJD_TDB"] = in_file["INT_TIMES"].data["int_mid_BJD_TDB"][j]
            time.append(in_file["INT_TIMES"].data["int_mid_BJD_TDB"][j])
            mjd_time.append(in_file["INT_TIMES"].data["int_mid_MJD_UTC"][j])
            print("%s: Saving frame %d"%(fits_files[i],count+1))

        except:
            new_header["int_mid_BJD_TDB"] = 0
            print("%s: Saving frame %d, without time stamps"%(fits_files[i],count+1))

        hdu1 = fits.PrimaryHDU(in_file["SCI"].data[j],header=new_header)
        hdu2 = fits.ImageHDU(in_file["ERR"].data[j])
        hdu = fits.HDUList([hdu1,hdu2])

        filename = "./science_fits_files/seg%s_frame_%s.fits"%(str(i+1).zfill(3),str(count+1).zfill(5))

        hdu.writeto(filename,overwrite=True)

        count += 1

pickle.dump(np.array(time),open("science_fits_files/bjd_time.pickle","wb"))
pickle.dump(np.array(mjd_time),open("science_fits_files/mjd_time.pickle","wb"))
