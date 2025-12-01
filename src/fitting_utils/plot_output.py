#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import pickle
import argparse
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from global_utils import parseInput
from fitting_utils import plotting_utils as pu
from fitting_utils import mcmc_utils as mc

### Define command line arguments with help

parser = argparse.ArgumentParser(description="Plot the output of gp_fit.py, pm_fit.py or gppm_fit.py. This produces the model fits to the wavelength-binned light curves, transmission spectrum and a plot of the calculated quadratic limb-darkening coefficients vs. expected limb-darkening coefficients.")
parser.add_argument('-s','--save_fig',help='Use if wanting to save the outputted figures, note overwrites current figures in cwd.',action='store_true')
parser.add_argument('-mask_bins','--mask_bins',help='Use if wanting to mask out particular bins (e.g. telluric O2) from plot. This is given as a list and indexed from 0, e.g. to mask bins 3,6 and 8: --mask_bins 2 5 7',type=int,nargs='+')
parser.add_argument('-st','--save_table',help='Use if wanting to save the outputted transmission spectrum to a table, note overwrites current table in cwd.',action='store_true')
parser.add_argument('-iib','--iib',help='If using iib for Na or K',action='store_true')
parser.add_argument('-sb','--start_bin',help='Select first bin to include in fitted model plot, indexed from 0. Note: using this option will not plot a transmission spectrum or the LDCs. Useful when wanting to split many fitted model into several plots.',type=int)
parser.add_argument('-eb','--end_bin',help='Select final bin to include in fitted model plot, indexed from 0. Note: using this option will not plot a transmission spectrum or the LDCs. Useful when wanting to split many fitted model into several plots.',type=int)
parser.add_argument('-pn','--photon_noise',help="Use this to not use rescaled errors but to use original errors, so that the RMS/photon noise can be accurately printed. Note: using this option means that no outputs will be saved.",action="store_true")
parser.add_argument('-wlc','--white_light_curve',help="Are we plotting a white light fit? If so, skip plotting the transmission spectrum",action="store_true")
parser.add_argument('-cp','--close_plots',help="If wanting to not show the plots (i.e. only wanting to save them), use this option",action="store_true")
parser.add_argument('-rebin','--rebin_data',help="If wanting to rebin the data for light curve plotting, specify how many bins here",type=int)
args = parser.parse_args()


### Load in data
x,y,e,e_r,m,m_in,w,we,completed_bins,nbins = pu.load_completed_bins(start_bin=args.start_bin,end_bin=args.end_bin,mask=args.mask_bins)
if not args.photon_noise: # if we're not using photon noise uncertainties, then we are using the rescaled error bars as our photometric uncertainties
    e = e_r

# raise SystemExit

### Print median RMS of all bins from LM_statistics.dat/prod_statistics.dat
try:
    rms_tab = "LM_statistics.txt"
    r = open(rms_tab,"r")
except:
    rms_tab = "prod_statistics.txt"
    r = open(rms_tab,"r")

all_RMS = []
for line in r:
    if "RMS" in line:
        all_RMS.append(float(line.split(" ")[-2]))

print("\n******Median RMS all bins = %d ppm*****\n"%np.median(all_RMS))
r.close()
if args.save_table:
    rn = open(rms_tab,"a")
    rn.write("\n******Median RMS all bins = %d ppm*****\n"%np.median(all_RMS))
    rn.close()

directory = os.getcwd()

if not args.white_light_curve and not args.photon_noise and args.start_bin is None and args.end_bin is None:
    ### Plot the transmission spectrum & the Rp/Rs error divided by photon noise
    trans_fig = pu.recover_transmission_spectrum(directory,save_fig=args.save_fig,plot_fig=True,bin_mask=args.mask_bins,print_RpErr_over_RMS=True,save_to_tab=args.save_table,iib=args.iib)
    if args.close_plots:
        plt.close()
    else:
        trans_fig.show()

    ### Plot the expected vs calculated limb darkening coefficients
    pu.expected_vs_calculated_ldcs(".",args.save_fig,bin_mask=args.mask_bins)
    if args.close_plots:
        plt.close()

### Plot the fitted models
# if we're only plotting bins between a certain range we don't want to plot the transmission spectrum
if not args.white_light_curve:
    if args.start_bin is not None or args.end_bin is not None:
        fig = pu.plot_models(m,x,y,e,w,save_fig=False,rebin_data=args.rebin_data)
        if args.save_fig and not args.photon_noise:
            if args.rebin_data is None:
                fig.savefig('fitted_models_wb%d-%d.pdf'%(args.start_bin+1,args.end_bin+1),bbox_inches='tight')
                fig.savefig('fitted_models_wb%d-%d.png'%(args.start_bin+1,args.end_bin+1),bbox_inches='tight')
            else:
                fig.savefig('fitted_models_wb%d-%d_rebin_%d.pdf'%(args.start_bin+1,args.end_bin+1,args.rebin_data),bbox_inches='tight')
                fig.savefig('fitted_models_wb%d-%d_rebin_%d.png'%(args.start_bin+1,args.end_bin+1,args.rebin_data),bbox_inches='tight')

        fig.show()
        raise SystemExit
    else:
        if nbins > 20:
            pass
            # ~ for i in range(0,nbins,10):
                # ~ fig = pu.plot_models(m[i:i+10],x[i:i+10],y[i:i+10],e[i:i+10],w[i:i+10],save_fig=False,rebin_data=args.rebin_data)
                # ~ if args.save_fig and not args.photon_noise:
                    # ~ if args.rebin_data is None:
                        # ~ fig.savefig('fitted_models_wb%s-%s.png'%(str(i).zfill(4),str(i+10).zfill(4)),bbox_inches='tight',dpi=200)
                    # ~ else:
                        # ~ fig.savefig('fitted_models_wb%s-%s_rebin_%d.png'%(str(i).zfill(4),str(i+10).zfill(4),args.rebin_data),bbox_inches='tight',dpi=200)
                    # ~ plt.close()
        else:
            pu.plot_models(m,x,y,e,w,save_fig=args.save_fig,rebin_data=args.rebin_data)
        if args.close_plots:
            plt.close()
else:
    fig = pu.plot_single_model(m[0],x[0],y[0],e[0],rebin_data=args.rebin_data,save_fig=args.save_fig,wavelength_bin=0,deconstruct=True,plot_residual_std=0)
    if not args.close_plots:
        fig.show()

if args.rebin_data is not None:
    raise SystemExit

### RMS vs bins (for comparison with photon noise)
print("Plotting RMS vs bins...")
residuals = []

input_dict = parseInput("fitting_input.txt")

for i,model in enumerate(m):
    # calculate transit model
    model_y = model.calc(x[i])
    if model.GP_used:
        mu,std = model.calc_gp_component(x[i],y[i],e[i])
        residuals.append(y[i] - model_y - mu)
    else:
        residuals.append(y[i]-model_y)

    # calculate ingress duration which sets the upper limit on the number of bins
    if i == 0:
        # first turn off limb-darkening
        if model.fix_u1:
            model.pars["u1"] = 0
        else:
            model.pars["u1"].currVal = 0
        if model.ld_law == "linear":
            continue
        if model.fix_u2:
            model.pars["u2"] = 0
        else:
            model.pars["u2"].currVal = 0
        if model.ld_law == "nonlinear":
            if model.fix_u3:
                model.pars["u3"] = 0
            else:
                model.pars["u3"].currVal = 0
            if model.fix_u4:
                model.pars["u4"] = 0
            else:
                model.pars["u4"].currVal = 0

        # now get transit only model
        red_noise_model = 1
        if model.poly_used:
            red_noise_model *= model.red_noise_poly(x[i])
        if model.exp_ramp_used:
             red_noise_model *= model.exponential_ramp(x[i])
        if model.step_func_used:
             red_noise_model *= model.step_function(x[i])

        tm = model.calc(x[i])/red_noise_model

        # now determine where the transit depth first and last reaches maximum - these are contact points 2 and 3
        # calculate the depth
        depth = model.pars["k"].currVal**2
        # ~ full_transit = np.where(tm==(1-depth))[0]
        full_transit = np.where(tm==tm.min())[0]
        contact2 = full_transit.min()
        contact3 = full_transit.max()

        # use these to refine contact1 and contact4
        contact1 = np.where(tm[:contact2]==tm.max())[0].max()
        contact4 = np.where(tm[contact3:]==tm.max())[0].min()+contact3

        ingress_duration = 24*60*(x[i][contact2]-x[i][contact1])
        print("Ingress duration = %d mins = %d frames"%(ingress_duration,contact2-contact1))

        transit_duration = 24*60*(x[i][contact4]-x[i][contact1])
        print("Transit duration = %d mins = %d frames"%(transit_duration,contact4-contact1))

        print("Contact 1 = %d; Contact 2 = %d; Contact 3 = %d; Contact 4 = %d"%(contact1,contact2,contact3,contact4))

residuals = np.atleast_1d(np.array(residuals))

if nbins == 1:
    ncols = 1
    nrows = 1
    nplots = 1
if nbins == 2:
    ncols = 2
    nrows = 1
    nplots = 1
if nbins > 2:
    ncols = 3
    nrows = int(np.ceil(float(nbins)/ncols))
    if nrows > 10:
        nrows = 10
        nplots = int(np.ceil(nbins/(nrows*ncols)))
    else:
        nplots = 1

if args.save_table:
    beta_factor_tab = open("red_noise_beta_factors.txt","w")
    beta_factor_tab.write("# Wavelength bin, Beta factor \n")

beta_factors = []
bin_counter = 0

for plot_no in range(nplots):
    fig = plt.figure(figsize=(ncols*5,nrows*2.5))
    subplot_counter = 1
    for i in range(bin_counter,nbins):
        if subplot_counter > nrows*ncols:
            continue

        ax = fig.add_subplot(nrows,ncols,subplot_counter)
        rms = []
        bin_size = []
        time_steps = []
        time_diff = np.diff(x[i]).min()*24*60 # in mins
        max_points = int(np.round(60/time_diff)) # go up to maximum of 30 minute bins
        # max_points = contact2 - contact1 # go up to ingress duration as max points per bin
        # max_points = 200

        npoints_per_bin = np.hstack((np.linspace(1,max_points,1000),max_points))

        for j in npoints_per_bin:
            if j == 1:
                rms.append(np.sqrt(np.nanmean(np.square(residuals[i]))))
                time_steps.append(np.diff(x[i]).mean()*24*60)
                bin_size.append(1)
            else:
                bins = np.linspace(x[i][0],x[i][-1],int(len(x[i])/j))
                time_steps.append(np.diff(bins).mean()*24*60)
                binned_x,binned_y,binned_e = pu.rebin(bins,x[i],residuals[i],e[i],weighted=False)
                rms.append(np.sqrt(np.nanmean(np.square(binned_y))))
                N = float(len(residuals[i]))/float((len(bins)))
                bin_size.append(N)

        gaussian_white_noise = np.array([1/np.sqrt(n) for n in npoints_per_bin])
        offset = max(gaussian_white_noise)/rms[0]
        gaussian_white_noise = gaussian_white_noise/offset
        # beta_factor = np.max(rms/gaussian_white_noise) # average ratio of measured dispersion to theortical value, https://iopscience.iop.org/article/10.1086/589737/pdf
        #
        # beta_factors.append(beta_factor)
        beta_factor = (rms[-1]/rms[0])/(gaussian_white_noise[-1]/gaussian_white_noise[0])
        beta_factors.append(beta_factor)

        if args.save_table:
            beta_factor_tab.write('%f %f \n'%(w[i],beta_factor))

        if i == 0:
            # ax.plot(npoints_per_bin,rms/rms[0],'r',lw=2,label='measured noise',zorder=1)
            # ax.plot(npoints_per_bin,gaussian_white_noise/gaussian_white_noise[0],color='k',lw=2,label='white noise',zorder=0)
            ax.plot(npoints_per_bin,1e6*np.array(rms),'r',lw=2,label='measured noise',zorder=1)
            ax.plot(npoints_per_bin,(gaussian_white_noise/gaussian_white_noise[0])*(1e6*rms[0]),color='k',lw=2,label='white noise',zorder=0)
            ax.legend()
        else:
            # ax.plot(npoints_per_bin,rms/rms[0],'r',lw=2,zorder=1)
            # ax.plot(npoints_per_bin,gaussian_white_noise/gaussian_white_noise[0],color='k',lw=2,zorder=0)
            ax.plot(npoints_per_bin,1e6*np.array(rms),'r',lw=2,zorder=1)
            ax.plot(npoints_per_bin,(gaussian_white_noise/gaussian_white_noise[0])*(1e6*rms[0]),color='k',lw=2,zorder=0)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(r'%.3f%s'%(w[i],pu.determine_wvl_units(w)),fontsize=12)

        if nrows == 1 and i == 0:
            ax.set_xlabel("Points per bin",fontsize=14)
            ax.set_ylabel("RMS (ppm)",fontsize=14)
        if i < nrows*ncols-ncols:
            ax.set_xticklabels([])
        if i%3 != 0:
            ax.set_yticklabels([])

        ax.tick_params(bottom=True,top=True,left=True,right=True,direction="inout",labelsize=12,length=8,width=1.,pad=1)
        ax.tick_params(which='minor',bottom=True,top=True,left=True,right=True,direction="inout",labelsize=12,length=4,width=1.)

        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)

        if nrows > 1:
            fig.text(0.5,0.08,'Points per bin',fontsize=14,ha='center',va='center')
            # fig.text(0.08,0.5,'Normalized RMS',fontsize=14,ha='center',va='center',rotation=90)
            fig.text(0.08,0.5,'RMS (ppm)',fontsize=14,ha='center',va='center',rotation=90)

        subplot_counter += 1
        bin_counter += 1

    if args.save_fig:
        if nbins > 10:
            plt.savefig('rms_vs_bins_%s.png'%(str(plot_no+1).zfill(4)),bbox_inches='tight',dpi=200)
        else:
            plt.savefig('rms_vs_bins_%s.pdf'%(str(plot_no+1).zfill(4)),bbox_inches='tight')
    if not args.close_plots:
        plt.show()
    plt.close()

print("\n******Median red noise beta factor = %.3f*****\n"%np.median(beta_factors))

if args.save_table:
    beta_factor_tab.write("\n******Median red noise beta factor = %.3f*****\n"%np.median(beta_factors))
    beta_factor_tab.close()
    if args.white_light_curve:
        mc.beta_rescale_uncertainties(np.array(beta_factors),"best_fit_parameters.txt",trans_spec_tab=None)
    else:
        mc.beta_rescale_uncertainties(np.array(beta_factors),"best_fit_parameters.txt","transmission_spectrum.txt")
