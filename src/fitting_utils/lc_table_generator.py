#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import argparse
import pickle
import pandas
from global_utils import parseInput

# try:
#     import fitting_utils.mcmc_utils as mc
# except:
#     from . import mcmc_utils as mc

parser = argparse.ArgumentParser(description='Generate .dat table of light curves and ancillary inputs from .pickle files defined in fitting_input.txt. ')
parser.add_argument("output_file",help="name of files that we're wanting to save output (.csv and .pickle) to")
parser.add_argument("-off",help='mjd offset to be added to time array',type=int)
args = parser.parse_args()


input_dict = parseInput('fitting_input.txt')


white_light_fit = bool(int(input_dict['white_light_fit']))

### Load in various input arrays

time = pickle.load(open(input_dict['time_file'],'rb'))
flux = pickle.load(open(input_dict['flux_file'],'rb'))
flux_error = pickle.load(open(input_dict['error_file'],'rb'))


if args.off is not None:
    time += args.off


column_headings = ["TIME (MJD)"]
data_array = [time]


if input_dict['airmass_file'] is not None:
    am = pickle.load(open(input_dict['airmass_file'],'rb'))
    column_headings += ["AIRMASS"]
    data_array += [am]
else:
    am = None

if input_dict['fwhm_file'] is not None:
    fwhm = pickle.load(open(input_dict['fwhm_file'],'rb'))
    column_headings += ["FWHM"]
    data_array += [fwhm]
else:
    fwhm = None

if input_dict['xpos_file'] is not None:
    xpos = pickle.load(open(input_dict['xpos_file'],'rb'))
    #xpos = np.atleast_1d(xpos)
    if white_light_fit:
        column_headings += ["XPOS"]
        data_array += [xpos]
else:
    xpos = None

if input_dict['ypos_file'] is not None:
    ypos = pickle.load(open(input_dict['ypos_file'],'rb'))
    if white_light_fit:
        column_headings += ["YPOS"]
        data_array += [ypos]
else:
    ypos = None

if input_dict['sky_file'] is not None:
    sky = pickle.load(open(input_dict['sky_file'],'rb'))
    if white_light_fit:
        column_headings += ["SKY"]
        data_array += [sky]
else:
    sky = None




if white_light_fit:
    column_headings += ["FLUX","FLUX_ERR"]
    data_array += [flux,flux_error]

else:

    nbins = flux.shape[0]
    nspectra = flux.shape[1]

    for i in range(nbins):

        if xpos is not None:
            data_array += [xpos[i]]
            column_headings += ["XPOS-%d"%(i+1)]

        if sky is not None:
            data_array += [sky[i]]
            column_headings += ["SKY-%d"%(i+1)]

        data_array += [flux[i],flux_error[i]]
        column_headings += ["FLUX-%d"%(i+1),"FLUX_ERR-%d"%(i+1)]

data_array = np.array(data_array).T

df = pandas.DataFrame(data=data_array,columns=np.array(column_headings))

# save as pandas pickled object and to .csv
df.to_pickle(args.output_file+'.pickle')
df.to_csv(args.output_file+'.csv')
