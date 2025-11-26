#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

from Tiberius.src.fitting_utils import plotting_utils as pu
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Use this code to give you mean-weighted uncertainties from > 1 white light curve fit')
parser.add_argument('best_fit_tabs',help="the paths to the WL best_fit_parameters.txt tables you want to pull from",nargs='+')
parser.add_argument('-t0_offsets',help="the offsets to be added to the best-fitting t0 values, to convert back into MJD/BJD",nargs='+',type=float)
parser.add_argument('-P','--period',help="the planet's period so that the weighted mean t0 can be computed from multiple visits",type=float)
parser.add_argument('-l','--labels',help="The x-axis labels if wanting to overwrite the default (Table 1, Table 2,...)",nargs="+")
parser.add_argument('-sp_only','--sys_params_only',help="Use this if wanting to plot only the system parameters without systematics parameters. Default is that all parameters are plotted.",action="store_true")
parser.add_argument('-no_mean','--no_mean',help="Use this if wanting to plot only parameters without calculating and plotting the weighted mean",action="store_true")
parser.add_argument('-s','--save_fig',help="Use this if wanting to save the figure",action="store_true")
parser.add_argument('-t','--title',help='use this to define the filename of the saved figure, overwriting the default')
parser.add_argument('-k','--k',help='use this to plot a horizontal line at a desired value of k (Rp/Rs)',type=float)
parser.add_argument('-aRs','--aRs',help='use this to plot a horizontal line at a desired value of aRs (a/Rs)',type=float)
parser.add_argument('-t0','--t0',help='use this to plot a horizontal line at a desired value of t0 (time of mid-transit)',type=float)
parser.add_argument('-t0_lit','--t0_lit',help='use this to define a literature value of t0 (time of mid-transit), for which all t0s are compared with',type=float)
parser.add_argument('-inc','--inc',help='use this to plot a horizontal line at a desired value of inc (inclination)',type=float)
args = parser.parse_args()

best_fit_dict = {}

if args.t0_offsets is not None:
    norbits = []

for i,t in enumerate(args.best_fit_tabs):

    keys,med,up,lo = np.genfromtxt(t,unpack=True,usecols=[0,2,4,6],dtype=str)

    nkeys = len(keys)

    for j in range(nkeys):

        new_key = keys[j].split("_")[0]
        value = float(med[j])
        value_up = float(up[j])
        value_lo = float(lo[j])

        if args.sys_params_only:
            if new_key not in ["t0","inc","k","u1","u2","aRs","ecc","omega"]:
                continue

        if i == 0:
            best_fit_dict[new_key] = []
            best_fit_dict["%s_up"%new_key] = []
            best_fit_dict["%s_lo"%new_key] = []

        if args.t0_offsets is not None and new_key == "t0":
            value += args.t0_offsets[i]
            if args.t0_lit is not None:
                n = np.round((value - args.t0_lit)/args.period)
            else:
                n = np.round((args.t0_offsets[i] - args.t0_offsets[0])/args.period)
            norbits.append(int(n))
            value -= int(n)*args.period

        try:
            if value == 0:
                value = value_up = value_lo = np.nan
            best_fit_dict[new_key].append(value)
            best_fit_dict["%s_up"%new_key].append(value_up)
            best_fit_dict["%s_lo"%new_key].append(value_lo)
        except:
            pass

    nkeys = len(best_fit_dict.keys())//3

if not args.no_mean:
    print("\n***Weighted mean parameters***\n")
    weighted_mean_dict = {}
    for k in best_fit_dict.keys():
        if "up" in k or "lo" in k:
            continue

        weighted_mean,weighted_mean_error = pu.weighted_mean_uneven_errors(best_fit_dict[k],best_fit_dict["%s_up"%k],best_fit_dict["%s_lo"%k])

        weighted_mean_dict[k] = weighted_mean
        weighted_mean_dict["%s_err"%k] = weighted_mean_error

        print("%s = %f +/- %f"%(k,weighted_mean,weighted_mean_error))
    print("\n*******************************\n")

    if args.t0_offsets is not None:
        for i,t in enumerate(args.t0_offsets):
            print("t0, offset %d = %f"%(i+1,weighted_mean_dict["t0"]+norbits[i]*args.period))

ntables = len(args.best_fit_tabs)
# if args.no_mean:
#     n_xticks = ntables
# else:
#     n_ticks = ntables+1

fig = plt.figure(figsize=(12*(np.ceil(ntables/4)),10))
subplot_counter = 1

for k in best_fit_dict.keys():

    if "err" in k or "_lo" in k or "_up" in k or len(best_fit_dict[k]) != ntables:
        continue

    ax = fig.add_subplot(nkeys//2,2,subplot_counter)
    ax.errorbar(np.arange(ntables)+1,best_fit_dict[k],yerr=((best_fit_dict["%s_lo"%k],best_fit_dict["%s_up"%k])),fmt='o',mec='k',capsize=3,lw=2)

    if not args.no_mean:
        ax.errorbar(ntables+1,weighted_mean_dict[k],yerr=weighted_mean_dict["%s_err"%k],color='r',fmt='o',mec='k',capsize=3,lw=2)
        ax.axhline(weighted_mean_dict[k],ls='--',color='k',zorder=0)

    if k == "k" and args.k is not None:
        ax.axhline(args.k,ls='--',color='r',lw=2,zorder=0)

    if k == "aRs" and args.aRs is not None:
        ax.axhline(args.aRs,ls='--',color='r',lw=2,zorder=0)

    if k == "inc" and args.inc is not None:
        ax.axhline(args.inc,ls='--',color='r',lw=2,zorder=0)

    if k == "t0" and args.t0 is not None:
        if args.t0_lit is not None:
            n = np.round((args.t0 - args.t0_lit)/args.period)
            ax.axhline(args.t0-int(n)*args.period,ls='--',color='r',lw=2,zorder=0)
        else:
            ax.axhline(args.t0,ls='--',color='r',lw=2,zorder=0)

    ax.set_ylabel("%s"%(k),fontsize=12)
    subplot_counter += 1

    if args.no_mean:
        ax.set_xticks(np.arange(1,ntables+1))
    else:
        ax.set_xticks(np.arange(1,ntables+2))

    if args.labels is None:
        if args.no_mean:
            ax.set_xticklabels(["Table %s"%(i+1) for i in range(ntables)],fontsize=12)
        else:
            ax.set_xticklabels(["Table %s"%(i+1) for i in range(ntables)]+["Weighted\nmean"],fontsize=12)
    else:
        if args.no_mean:
            ax.set_xticklabels(["%s"%(i).replace(" ","\n") for i in args.labels],fontsize=12)
        else:
            ax.set_xticklabels(["%s"%(i).replace(" ","\n") for i in args.labels]+["Weighted\nmean"],fontsize=12)

    ax.tick_params(which='minor',bottom=False,top=False,left=True,right=True)#,direction="inout",length=2,width=1.)

if args.save_fig:
    if args.title is None:
        fig_title = "system_parameters.pdf"
    else:
        fig_title = args.title
    fig.savefig(fig_title,bbox_inches="tight",dpi=260)

plt.show()
