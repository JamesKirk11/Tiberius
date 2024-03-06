#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from fitting_utils import plotting_utils as pu
from global_utils import parseInput

parser = argparse.ArgumentParser(description='Generate .dat table of time, flux, err and transit and gp models and residuals')
parser.add_argument("-pre",help="the prefix of the table name, to go before model_tab_wbXX.dat, can be left blank")
parser.add_argument("-off",help='mjd offset to be added to time array',type=int)
parser.add_argument("-dt_only",help='Use this if only wanting to save the detrended flux and rescaled uncertainties to table',action="store_true")
args = parser.parse_args()

x,y,e,e_r,m,m_in,w,we,completed_bins,nbins = pu.load_completed_bins(directory=".",start_bin=None,end_bin=None,mask=None,return_index_only=False)

if args.off is None: # automatically load in the time offset
    input_dict = parseInput('fitting_input.txt')
    time = pickle.load(open(input_dict['time_file'],'rb'))
    time_offset = int(time[0])
else:
    time_offset = args.off


for i,n in enumerate(completed_bins):

    print("Working on bin %d..."%(n))

    transit_model = m[i].calc(x[i])

    gp = m[i].GP_used
    poly = m[i].poly_used
    exp = m[i].exp_ramp_used
    WL = m[i].white_light_fit

    if not gp:
        oot = 1
        if poly:
            oot *= m[i].red_noise_poly(x[i])
        if exp:
            oot *= m[i].exponential_ramp(x[i])
        residuals = y[i] - transit_model
        full_model = transit_model.copy()
        transit_model /= oot

    if gp:
        oot,_ = m[i].calc_gp_component(x[i],y[i],e[i],deconstruct_gp=False)
        full_model = transit_model + oot
        residuals = y[i] - transit_model - oot

    npoints = len(x[i])

    x[i] += time_offset

    if args.pre is not None:
        table_name = "%s_model_tab_"%args.pre
    else:
        table_name = "model_tab_"

    if args.dt_only:
        table_name += "detrended_"

    if WL:
        table_name += "WL.txt"
    else:
        table_name += "wb%s.txt"%(str(n).zfill(4))

    new_tab = open(table_name,'w')

    if not args.dt_only:
        new_tab.write("# Time | Flux | Flux err | Rescaled flux err | Full model | Transit model | Systematics model | Residuals \n")
        for j in range(npoints):
            new_tab.write("%f %f %f %f %f %f %f %f \n"%(x[i][j],y[i][j],e[i][j],e_r[i][j],full_model[j],transit_model[j],oot[j],residuals[j]))
        new_tab.close()

    else:
        new_tab.write("# Time | Detrended Flux | Flux err \n")
        for j in range(npoints):
            new_tab.write("%f %f %f \n"%(x[i][j],y[i][j]-oot[j],e_r[i][j]))
        new_tab.close()
