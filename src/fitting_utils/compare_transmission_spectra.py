#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from astropy import constants
from matplotlib.ticker import AutoMinorLocator,ScalarFormatter

from Tiberius.src.fitting_utils import plotting_utils as pu
from Tiberius.src.reduction_utils import wavelength_binning as wb
from global_utils import parseInput

parser = argparse.ArgumentParser(description='Convert ACCESS pipeline outputs to input pickle files needed by LRG-BEASTS fitting_utils')
parser.add_argument('-d','--directory_list',help="the directories containing the transmission spectra to compare",nargs='+')
parser.add_argument('-ts','--trans_spec',help="Define the transmission spectra tables if not defining the directory list",nargs='+')
parser.add_argument('-s','--save_fig',help='Save resulting figure?',action='store_true')
parser.add_argument('-o','--auto_offset',help='Apply automatic offset to each transmission spectrum so that the median Rp/Rs is equal?',action='store_true')
parser.add_argument('-mo','--manual_offset',help='Apply manual offset to each transmission spectrum using user-defined offsets',nargs='+',type=float)
parser.add_argument('-anchor_wvl','--anchor_wvl',help="Use this to anchor all transmission spectra to a single common bin. Define the wavelength here and it'll use the nearest bin",type=float)
parser.add_argument('-c','--combine',help='Take the weighted mean of the transmission spectra and plot on separate panel?',action='store_true')
parser.add_argument('-t','--title',help='use this to define the suffix of the plot title (after trans_spec_comparison), otherwise is determined automatically')
parser.add_argument('-l','--labels',help='use this to define the labels on the plot, otherwise are determined automatically',nargs='+')
parser.add_argument('-iib','--iib',help="use this if we're plotting iib transmission spectra (wvl width on x-axis)",action="store_true")
parser.add_argument("-x_offset","--plot_x_offset",help="add offset in x for overlapping wavelength bins. Default is to not apply offset",action="store_true")
parser.add_argument("-plot_xerr","--plot_xerr",help="Plot the wavelength bin widths on the plot? Default is not to.",action="store_true")
parser.add_argument("-std_wmean","--std_wmean",help="Perform a standard weighted mean using the average of error bars, not taking into account the fact they are assymetric",action="store_true")
parser.add_argument("-alkali","--alkali",help="use this if wanting to plot the alkali lines on the transmission spectrum",action="store_true")
parser.add_argument("-halpha","--halpha",help="use this if wanting to plot H-alpha on the transmission spectrum",action="store_true")
parser.add_argument("-flat_line","--flat_line",help="use this if wanting to add a flat line to the transmission spectrum",action="store_true")
parser.add_argument("-microns","--microns",help="use this if the input units are microns, not Angstroms",action="store_true")
parser.add_argument("-log","--log_scale",help="use this if wanting to make the x axis (wavelength) log-scale",action="store_true")
parser.add_argument("-H_rs","--H_rs",help="use this if wanting to plot the scale height as the right hand axis. This is defined as the ratio of the scale height to the stellar radius",type=float)
parser.add_argument("-input","--input",help="second method to plot the scale height as the right hand axis. To use this define the path to the fitting_input.txt file")
parser.add_argument("-pickle","--pickle",help="use this if wanting to pickle the figure object",action="store_true")
parser.add_argument("-colours","--colours",help="use this to define the plot colours",nargs='+')
parser.add_argument("-symbols","--symbols",help="use this to define the plot symbols",nargs='+')
parser.add_argument("-R","--resolution",help="use this to rebin the spectrum to a lower spectral resolution",type=int,default=0)
args = parser.parse_args()




def average_transmission_spectra(w_array,we_array,k_array,k_up_array,k_low_array,std_wmean=False):
    """Takes an array of arrays and calculates the weighted mean taking into account uneven error bars."""

    # first make a wavelength array that contains the full range of bins
    wavelengths = np.atleast_1d(sorted(set(np.hstack(w_array))))
    nbins = len(wavelengths)

    w_all = []
    we_all = []
    rp_all = []
    rp_up_all = []
    rp_low_all = []

    # the number of transmission spectra is given by the length of any of these arrays
    n_trans_spec = len(w_array)

    # Now loop through w_all and nightly transmission spectra:

    for n in range(n_trans_spec):

        current_w = []
        current_we = []
        current_rp = []
        current_rp_up = []
        current_rp_low = []


        for i,w in enumerate(wavelengths):

            if w in np.atleast_1d(w_array[n]):
                # print(w)
                matching_idx = np.atleast_1d(w_array[n]).tolist().index(w)
                # print(we_array[n][matching_idx],matching_idx)
                current_w.append(np.atleast_1d(w_array[n])[matching_idx])
                current_we.append(np.atleast_1d(we_array[n])[matching_idx])
                current_rp.append(np.atleast_1d(rp_array[n])[matching_idx])
                current_rp_up.append(np.atleast_1d(rp_up_array[n])[matching_idx])
                current_rp_low.append(np.atleast_1d(rp_low_array[n])[matching_idx])
            else:
                current_w.append(w)
                current_we.append(np.nan)
                current_rp.append(np.nan)
                current_rp_up.append(np.nan)
                current_rp_low.append(np.nan)

        w_all.append(current_w)
        we_all.append(current_we)
        rp_all.append(current_rp)
        rp_up_all.append(current_rp_up)
        rp_low_all.append(current_rp_low)

    w_all = np.array(w_all)
    we_all = np.array(we_all)
    rp_all = np.array(rp_all)
    rp_up_all = np.array(rp_up_all)
    rp_low_all = np.array(rp_low_all)

    # Now have to loop through k array to caculate mean of only those entries without nans
    rp_mean = []
    rpe_mean = []
    for i in range(nbins):
        index = np.isfinite(rp_all.T[i])

        current_rp = rp_all.T[i][index]
        current_rp_up = rp_up_all.T[i][index]
        current_rp_low = rp_low_all.T[i][index]

        if std_wmean:
            # print(current_rp_up,current_rp_low)
            # print(np.mean((current_rp_up,current_rp_low),axis=1))
            current_rp_mean,current_rpe_mean = np.average(current_rp,weights=1/np.mean((current_rp_up,current_rp_low),axis=0)**2,returned=True)
            current_rpe_mean = np.sqrt(1/current_rpe_mean)
        else:
            current_rp_mean,current_rpe_mean = pu.weighted_mean_uneven_errors(current_rp,current_rp_up,current_rp_low)

        rp_mean.append(current_rp_mean)
        rpe_mean.append(current_rpe_mean)

    rp_mean = np.array(rp_mean)
    rpe_mean = np.array(rpe_mean)

    return rp_mean,rpe_mean,w_all,we_all,rp_all,rp_up_all,rp_low_all

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_all_transmission_spectra(direc_list,auto_offset,manual_offset,anchor_wvl,resolution):

    """If providing a list of directories"""

    medians = []
    w_all = []
    we_all = []
    rp_all = []
    rp_up_all = []
    rp_low_all = []

    for d in direc_list:

        print("\n## Statistics for %s..."%d)

        rp,rp_up,rp_low,w,we,_ = pu.recover_transmission_spectrum(d,save_fig=False,plot_fig=False,bin_mask=None,save_to_tab=False,print_RpErr_over_RMS=False)

        if resolution > 0:
            binned_wvl = pu.bin_wave_to_R(w,resolution)
            binned_spec = pu.bin_trans_spec(binned_wvl,w,rp,rp_up,rp_low)
            rp,rp_up,rp_low,w,we = binned_spec["bin_y"],binned_spec["bin_dy"],binned_spec["bin_dy"],binned_spec["bin_x"],binned_spec["bin_dx"]

        w_all.append(w)
        we_all.append(we)
        rp_all.append(rp)
        rp_up_all.append(rp_up)
        rp_low_all.append(rp_low)

        medians.append(np.nanmedian(rp))

    global_median = np.nanmedian(medians)

    rp_all_offset = []

    ndirecs = len(direc_list)

    if auto_offset:
        for i in rp_all:
            rp_all_offset.append(i-np.nanmedian(i)+global_median)


    elif manual_offset is not None:
        for i,j in enumerate(rp_all):
            rp_all_offset.append(rp_all[i]+manual_offset[i])


    elif anchor_wvl is not None:
        anchor_median = []
        for i in range(ndirecs):
            index = find_nearest(w_all[i],anchor_wvl)
            anchor_median.append(rp_all[i][index])
        anchor_median = np.nanmedian(anchor_median)
        for i in range(ndirecs):
            index = find_nearest(w_all[i],anchor_wvl)
            rp_all_offset.append(rp_all[i]-rp_all[i][index]+anchor_median)

    else:
        rp_all_offset = rp_all

    return w_all,we_all,rp_all_offset,rp_up_all,rp_low_all



def load_all_transmission_spectra(tspec_list,auto_offset,manual_offset,anchor_wvl,resolution):

    """If providing a list of trans spec files"""

    medians = []
    w_all = []
    we_all = []
    rp_all = []
    rp_up_all = []
    rp_low_all = []

    for d in tspec_list:

        print("\n## Statistics for %s..."%d)

        w,we,rp,rp_up,rp_low = np.loadtxt(d,unpack=True,usecols=[0,1,2,3,4])

        if resolution > 0:
            binned_wvl = pu.bin_wave_to_R(w,resolution)
            binned_spec = pu.bin_trans_spec(binned_wvl,w,rp,rp_up,rp_low)
            rp,rp_up,rp_low,w,we = binned_spec["bin_y"],binned_spec["bin_dy"],binned_spec["bin_dy"],binned_spec["bin_x"],binned_spec["bin_dx"]

        print("\nMedian Rp/Rs = %f ;  Median Rp/Rs +ve error (ppm) = %d ; Median Rp/Rs -ve error (ppm) = %d ; Median R/Rs error (ppm) = %d "%(np.median(rp),np.nanmedian(rp_up)*1e6,np.nanmedian(rp_low)*1e6,np.nanmedian(np.hstack((rp_up,rp_low)))*1e6))

        w_all.append(w)
        we_all.append(we)
        rp_all.append(rp)
        rp_up_all.append(rp_up)
        rp_low_all.append(rp_low)

        medians.append(np.nanmedian(rp))

    global_median = np.nanmedian(medians)

    rp_all_offset = []

    ndirecs = len(tspec_list)

    if auto_offset:
        for i in rp_all:
            rp_all_offset.append(i-np.nanmedian(i)+global_median)

    elif manual_offset is not None:
        for i,j in enumerate(rp_all):
            rp_all_offset.append(rp_all[i]+manual_offset[i])


    elif anchor_wvl is not None:
        anchor_median = []
        for i in range(ndirecs):
            index = find_nearest(w_all[i],anchor_wvl)
            anchor_median.append(rp_all[i][index])
        anchor_median = np.nanmedian(anchor_median)
        for i in range(ndirecs):
            index = find_nearest(w_all[i],anchor_wvl)
            rp_all_offset.append(rp_all[i]-rp_all[i][index]+anchor_median)

    else:
        rp_all_offset = rp_all

    return w_all,we_all,rp_all_offset,rp_up_all,rp_low_all


if args.directory_list is not None:
    w_all,we_all,rp_all,rp_up_all,rp_low_all = get_all_transmission_spectra(args.directory_list,args.auto_offset,args.manual_offset,args.anchor_wvl,args.resolution)
    in_list = args.directory_list

if args.trans_spec is not None:
    w_all,we_all,rp_all,rp_up_all,rp_low_all = load_all_transmission_spectra(args.trans_spec,args.auto_offset,args.manual_offset,args.anchor_wvl,args.resolution)
    in_list = args.trans_spec


if args.combine:
    if not args.iib:
        rp_mean,rpe_mean,w_concat,we_concat,rp_concat,rp_up_concat,rp_low_concat = average_transmission_spectra(w_all,we_all,rp_all,rp_up_all,rp_low_all,args.std_wmean)
    else:
        rp_mean,rpe_mean,w_concat,we_concat,rp_concat,rp_up_concat,rp_low_concat = average_transmission_spectra(we_all,we_all,rp_all,rp_up_all,rp_low_all,args.std_wmean)

    w_mean = np.nanmean(w_concat,axis=0)
    we_mean = np.nanmean(we_concat,axis=0)

    print('## Average error bar in combined spectrum = %d ppm'%(np.round(1e6*np.nanmean(rpe_mean))))

    subplots = 2

else:
    subplots = 1


# Calculate scale height
if args.H_rs is not None:
    H_Rs = args.H_rs # scale height divided by stellar radius
else:
    if args.input is not None:
        input_dict = parseInput(args.input)
        try:
            H = (constants.k_B.value*float(input_dict['Teq']))/(2.3*constants.m_p.value*10**float(input_dict['logg'])/100.)
        except:
            H = (constants.k_B.value*float(input_dict['Teq']))/(2.3*constants.m_p.value*float(input_dict['g']))
        Rs = float(input_dict['rs'])*constants.R_sun.value
        H_Rs = H/Rs
        transit_depth_1H = (2*H*constants.R_jup.value*float(input_dict['rp']))/(float(input_dict['rs'])*constants.R_sun.value)**2
        print("\n** Atmospheric scale height: H/Rs = %d ppm **"%(H_Rs*1e6))
        print("\n** Transit depth per scale height = %d ppm **\n"%(transit_depth_1H*1e6))
    else:
        H_Rs = None

# Make figure

if args.symbols is None:
    markers = ['o','^','s','p','D','*','8','P','X','.','o','^','s','p','D','*']
else:
    markers = args.symbols


if args.combine:
    ms = 2
    alpha = 0.7
else:
    ms = 4
    alpha = 1

if args.microns:
    wav_units = 1e-4
else:
    wav_units = 1

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(subplots,1,1)

if args.title is None:
    title = ''
else:
    title = args.title

if args.plot_x_offset:
    # add 10A offsets in x
    # offsets = np.arange(0-20*len(in_list),0+20*len(in_list),20)*wav_units
    offsets = np.arange(0-0.5*np.mean(we_all)*len(in_list),0+0.5*np.mean(we_all)*len(in_list),0.5*np.mean(we_all))
else:
    offsets = np.zeros(len(in_list))

for i,d in enumerate(in_list):

    if args.labels is None:
        handle = d.split('/')[-2]
    else:
        handle = args.labels[i]

    if args.title is None:
        title += '_%s'%handle

    w = w_all[i]*wav_units
    we = we_all[i]*wav_units
    rp = rp_all[i]
    rp_up = rp_up_all[i]
    rp_low = rp_low_all[i]

    if not args.iib:
        if args.plot_xerr:
            if args.colours is not None:
                ax.errorbar(w+offsets[i],rp,xerr=we/2,yerr=(np.atleast_1d(rp_low),np.atleast_1d(rp_up)),fmt=markers[i],label=handle,alpha=alpha,capsize=2,mec='k',ms=ms,\
                mfc=args.colours[i],color=args.colours[i])
            else:
                ax.errorbar(w+offsets[i],rp,xerr=we/2,yerr=(np.atleast_1d(rp_low),np.atleast_1d(rp_up)),fmt=markers[i],label=handle,alpha=alpha,capsize=2,mec='k',ms=ms)
        else:
            if args.colours is not None:
                ax.errorbar(w+offsets[i],rp,xerr=None,yerr=(np.atleast_1d(rp_low),np.atleast_1d(rp_up)),fmt=markers[i],label=handle,alpha=alpha,capsize=2,mec='k',ms=ms,\
                mfc=args.colours[i],color=args.colours[i])
            else:
                ax.errorbar(w+offsets[i],rp,xerr=None,yerr=(np.atleast_1d(rp_low),np.atleast_1d(rp_up)),fmt=markers[i],label=handle,alpha=alpha,capsize=2,mec='k',ms=ms)
    else:
        ax.errorbar(we,rp,yerr=(rp_low,rp_up),fmt='o',label=handle,alpha=0.7,capsize=5)

# ax.set_ylabel('$R_P/R_*$',fontsize=14)
ax.legend(framealpha=1,ncol=4)

xlims = ax.get_xlim()
ylims = ax.get_ylim()

if xlims[0] < wb.sodium_centre*wav_units < xlims[1] and args.alkali:
    ax.axvline(wb.sodium_centre*wav_units,ls='--',color='grey',zorder=0)
    # ax.text(wb.sodium_centre*wav_units+20*wav_units,ylims[1]*0.98,'Na',color='grey',fontsize=14,ha='left',zorder=10)

if xlims[0] < wb.potassium_centre*wav_units < xlims[1] and args.alkali:
    ax.axvline(wb.potassium_d1*wav_units,ls='--',color='grey',zorder=0)
    ax.axvline(wb.potassium_d2*wav_units,ls='--',color='grey',zorder=0)
    # ax.text(wb.potassium_centre*wav_units+30*wav_units,ylims[1]*0.98,'K',color='grey',fontsize=14,ha='left',zorder=10)

    # ax.axvline(wb.potassium_d1,ls='--',color='grey')
    # ax.text(wb.potassium_d1+5,ylims[1]*0.95,'K1',color='grey',fontsize=14,ha='left')
    #
    # ax.axvline(wb.potassium_d2,ls='--',color='grey')
    # ax.text(wb.potassium_d2+5,ylims[1]*0.95,'K2',color='grey',fontsize=14,ha='left')


if xlims[0] < wb.Halpha*wav_units < xlims[1] and args.halpha:
    ax.axvline(wb.Halpha*wav_units,ls='--',color='grey',zorder=0)
    ax.text(wb.Halpha*wav_units+5*wav_units,ylims[1]*0.98,'H-alpha',color='grey',fontsize=14,ha='left',zorder=10)



if args.flat_line:
    ax.axhline(np.median([np.median(i) for i in rp_all]),ls='--',color='k',lw=2,zorder=0)

minor_locator = AutoMinorLocator(10)
ax.xaxis.set_minor_locator(minor_locator)
ax.tick_params(bottom=True,top=True,left=True,right=True,direction="inout",labelsize=12,length=6,width=1.,pad=3)
ax.tick_params(which='minor',bottom=True,top=True,left=True,right=True,direction="inout",labelsize=8,\
                   length=4,width=1.)

if args.log_scale:
    ax.set_xscale('log')
    ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1,2,3,4,5])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())


if H_Rs is not None: # plot scale axis as right hand axis
    yticks = ax.get_yticks()
    ax2 = ax.twinx()
    scale_height_tick_range = (yticks[-1]-yticks[0])/H_Rs
    ax2.set_ylim(-scale_height_tick_range/2.,scale_height_tick_range/2.)
    # ax2.set_ylabel('Atmospheric Scale Heights (H)',fontsize=12)

if args.combine:

    ax2 = fig.add_subplot(212)
    if not args.iib:
        ax2.errorbar(w_mean,rp_mean,xerr=we_mean/2,yerr=rpe_mean,fmt='o',label='Weighted mean',ecolor='k',color='k',capsize=2,mfc='grey')
    else:
        ax2.errorbar(we_mean,rp_mean,yerr=rpe_mean,fmt='o',label='Weighted mean',ecolor='k',color='k',capsize=2,mfc='grey')
    # ax2.set_ylabel('$R_P/R_*$',fontsize=14)
    ax2.set_xlabel('Wavelength (%s)'%pu.determine_wvl_units(w_mean),fontsize=14)
    ax2.legend()
    ylims = ax2.get_ylim()

    if xlims[0] < wb.sodium_centre < xlims[1] and args.alkali:
        ax2.axvline(wb.sodium_centre,ls='--',color='grey',zorder=0)
        ax2.text(wb.sodium_centre+20,ylims[1]*0.98,'Na',color='grey',fontsize=14,ha='left',zorder=10)

    if xlims[0] < wb.potassium_centre < xlims[1] and args.alkali:
        ax2.axvline(wb.potassium_d1,ls='--',color='grey',zorder=0)
        ax2.axvline(wb.potassium_d2,ls='--',color='grey',zorder=0)
        ax2.text(wb.potassium_centre+30,ylims[1]*0.98,'K',color='grey',fontsize=14,ha='left',zorder=10)

        # ax2.axvline(wb.potassium_d1,ls='--',color='grey')
        # ax2.text(wb.potassium_d1+5,ylims[1]*0.95,'K1',color='grey',fontsize=14,ha='left')
        #
        # ax2.axvline(wb.potassium_d2,ls='--',color='grey')
        # ax2.text(wb.potassium_d2+5,ylims[1]*0.95,'K2',color='grey',fontsize=14,ha='left')

    if xlims[0] < wb.Halpha < xlims[1] and args.halpha:
        ax2.axvline(wb.Halpha,ls='--',color='grey',zorder=0)
        ax2.text(wb.Halpha+5,ylims[1]*0.98,'H-alpha',color='grey',fontsize=14,ha='left',zorder=10)

    ax2.set_xlim(xlims)
    # ax2.set_ylim(ylims)

    ax2.xaxis.set_minor_locator(minor_locator)

    ax2.tick_params(bottom=True,top=True,left=True,right=True,direction="inout",labelsize=12,length=6,width=1.,pad=3)
    ax2.tick_params(which='minor',bottom=True,top=True,left=True,right=True,direction="inout",labelsize=8,\
                   length=4,width=1.)

    if H_Rs is not None: # plot scale axis as right hand axis
        yticks2 = ax2.get_yticks()
        ax2a = ax2.twinx()
        scale_height_tick_range_2 = (yticks2[-1]-yticks2[0])/H_Rs
        ax2a.set_ylim(-scale_height_tick_range_2/2.,scale_height_tick_range_2/2.)
        # ax2a.set_ylabel('Atmospheric Scale Heights (H)',fontsize=12)


else:
    # if args.microns:
    #     ax.set_xlabel('Wavelength ($\mu$m)',fontsize=14)
    # else:
        # ax.set_xlabel('Wavelength ($\AA$)',fontsize=14)
    ax.set_xlabel('Wavelength (%s)'%pu.determine_wvl_units(w),fontsize=14)

if args.combine:
    plt.tight_layout()
    if args.auto_offset or args.manual_offset is not None:
        fig.text(0.001, 0.5, '$R_P/R_*$ + offset', va='center', rotation='vertical',fontsize=14)
    else:
        fig.text(0.001, 0.5, '$R_P/R_*$', va='center', rotation='vertical',fontsize=14)
else:
    if args.auto_offset or args.manual_offset is not None:
        fig.text(0.03, 0.5, '$R_P/R_*$ + offset', va='center', rotation='vertical',fontsize=14)
    else:
        fig.text(0.03, 0.5, '$R_P/R_*$', va='center', rotation='vertical',fontsize=14)

if H_Rs is not None:
    fig.text(1-0.04, 0.5, 'Atmospheric scale heights (H)', va='center', rotation='vertical',fontsize=14)

if args.save_fig:
    fig.savefig('%s.pdf'%title,bbox_inches='tight')
    fig.savefig('%s.png'%title,bbox_inches='tight',dpi=360)

    if args.combine:

        # save the combined transmission spectrum to a table
        new_tab = open('%s.txt'%title,'w')
        new_tab.write('# Wvl_centre Wvl_error Rp/Rs Rp/Rs_err Rp/Rs_err \n')

        # save the combined transmission spectrum to a table
        depths_tab = open('%s_depths.txt'%title,'w')
        depths_tab.write('# Wvl centre (%s), Wvl error (%s), Transit depth (ppm), Transit depth err (ppm) \n'%(pu.determine_wvl_units(w_mean),pu.determine_wvl_units(w_mean)))

        # Make table ready for PLATON input
        retrieval_tab = open('%s_PLATON.txt'%title,'w')
        retrieval_tab.write('# Wlow (%s) Wup (%s) Transit_Depth (ppm) Transit_Depth_ErrUp (ppm) Transit_Depth_ErrLow (ppm) \n'%(pu.determine_wvl_units(w_mean),pu.determine_wvl_units(w_mean)))

        Wlow = w_mean-we_mean/2
        Wup = w_mean+we_mean/2
        depths = 1e6*(rp_mean**2)
        ErrUp = ErrLow = 1e6*2*(rpe_mean*rp_mean)

        nbins = len(w_mean)
        for i in range(nbins):
            new_tab.write('%f %f %f %f %f \n'%(w_mean[i],we_mean[i],rp_mean[i],rpe_mean[i],rpe_mean[i]))
            depths_tab.write('%f %f %f %f \n'%(w_mean[i],we_mean[i],depths[i],ErrUp[i]))
            retrieval_tab.write('%f %f %f %f %f \n'%(Wlow[i],Wup[i],depths[i],ErrUp[i],ErrLow[i]))

        new_tab.close()
        retrieval_tab.close()
        depths_tab.close()

if args.pickle:
    pickle.dump(fig, open('FigureObject_%s.fig.pickle'%title, 'wb'))
plt.show()
