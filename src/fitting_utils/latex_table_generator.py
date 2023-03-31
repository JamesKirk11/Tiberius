#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
import argparse
from global_utils import parseInput

# try:
#     import fitting_utils.mcmc_utils as mc
# except:
#     from . import mcmc_utils as mc

parser = argparse.ArgumentParser(description='Generate latex table from .dat table of results. Note: not tested with updated classes.')
parser.add_argument("tab_name",help="which table are we using?")
args = parser.parse_args()

w,we,u1,u1e,u2,u2e = np.loadtxt('LD_coefficients.dat',unpack=True)

input = parseInput('fitting_input.txt')

wl = bool(int(input['white_light_fit']))
if wl:
    nbins = 1
else:
    nbins = len(w)

fix_u1 = bool(int(input['fix_u1']))
fix_u2 = bool(int(input['fix_u2']))


if wl:

    blob = np.loadtxt(args.tab_name,dtype=str,delimiter='\n')
    input_dict = {}

    for i,line in enumerate(blob):

        param_name,values = line.split('=')
        param_name = param_name.split("_")[0]


        values.split()

        v = str(values.split()[0])

        if 'k' not in param_name and 't0' not in param_name:
            v = str(float('%.2f'%np.round(float(v),2)))
            v_up = '%s'%float('%.2f'%np.round(float(values.split()[2]),2))
            v_low = '%s'%float('%.2f'%np.round(float(values.split()[4]),2))
        else:
            v_up = values.split()[2]
            v_low = values.split()[4]


        if v_up == v_low:
            print('%s & $ %s\pm{%s} $ \ \\ '%(param_name,v,v_up))
        else:
            print('%s & $ %s^{+%s}_{-%s} $ \ \\'%(param_name,v,v_up,v_low))

        if 'u' in param_name and fix_u2:
            if wl:
                print("u2 (fixed) & $ %s $ \ \\"%(u2))
            else:
                print("u2 (fixed) & $ %s $ \ \\"%(u2[i]))

        if 'u' in param_name and fix_u1:
            if wl:
                print("u1 (fixed) & $ %s $ \ \\"%(u1))
            else:
                print("u1 (fixed) & $ %s $ \ \\"%(u1[i]))

else: # we're tabulating the transmission spectrum:

    table = np.loadtxt(args.tab_name)

    print("Bin centre  & Bin width & $R_P/R_*$ & $u1$ & $u2$ \\\\")
    print("(\AA) & (\AA) & & & \\\\ \hline")

    for i in range(nbins):

        k = str("%.5f" % np.round(table[:,2][i],5))
        k_up = str("%.5f" % np.round(table[:,3][i],5))
        k_low = str("%.5f" % np.round(table[:,4][i],5))

        if k_up == k_low:
            k_err = " \pm %s "%k_up
        else:
            k_err = " ^{+%s} _{-%s} "%(k_up,k_low)

        nbins = len(k)

        idx = 5

        if not fix_u1:
            u1_fit = str("%.2f" % np.round(table[:,5][i],2))
            u1_up_fit = str("%.2f" % np.round(table[:,6][i],2))
            u1_low_fit = str("%.2f" % np.round(table[:,7][i],2))

            if u1_up_fit == u1_low_fit:
                u1_err = " \pm %s "%u1_low_fit
            else:
                u1_err = " ^{+%s} _{-%s} "%(u1_up_fit,u1_low_fit)

            idx = 8

        else:
            u1_fit = str("%.2f" % np.round(u1[i],2))
            u1_err = ""

        if not fix_u2:
            u2_fit = str("%.2f" % np.round(table[:,idx][i],2))
            u2_up_fit = str("%.2f" % np.round(table[:,idx+1][i],2))
            u2_low_fit = str("%.2f" % np.round(table[:,idx+2][i],2))

            if u2_up_fit == u2_low_fit:
                u2_err = " \pm %s "%u2_low_fit
            else:
                u2_err = " ^{+%s} _{-%s} "%(u2_up_fit,u2_low_fit)

        else:
            u2_fit = str("%.2f" % u2[i])
            u2_err = ""

        print("%d   &   %d  &  $ %s %s $ & $ %s %s $ & $ %s %s $ \\\\"%(w[i],we[i],k,k_err,u1_fit,u1_err,u2_fit,u2_err))
print("\hline")
