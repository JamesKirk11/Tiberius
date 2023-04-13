#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-x1','--x1',help='Use this argument to give the name of the file containing the x pixel shifts of the 1st trace if wanting to plot')
parser.add_argument('-x2','--x2',help='Use this argument to give the name of the file containing the x pixel shifts of the 2nd trace if wanting to plot')
parser.add_argument('-y1','--y1',help='Use this argument to give the name of the file containing the y pixel shifts of the 1st trace if wanting to plot')
parser.add_argument('-y2','--y2',help='Use this argument to give the name of the file containing the y pixel shifts of the 2nd trace if wanting to plot')
parser.add_argument('-no_fwhm','--fwhm',help='Use this argument to turn off plotting of FWHM, default is on',action='store_true')
parser.add_argument('-no_max','--max',help='Use this argument if turn off plotting of maximum counts for both stars, default is on',action='store_true')
parser.add_argument('-m1','--model1',help='Use this argument to give the name of the file containing the the first white light curve model if wanting to plot')
parser.add_argument('-m2','--model2',help='Use this argument to give the name of the file containing the the second white light curve model if wanting to plot')
parser.add_argument('-wlx','--wlx',help='Use this argument to give the name of the file containing the time data for the white light fit')
parser.add_argument('-wly','--wly',help='Use this argument to give the name of the file containing the flux data for the white light fit')
parser.add_argument('-wle','--wle',help='Use this argument to give the name of the file containing the error data for the white light fit')
parser.add_argument('-raw_lcs','--raw_lcs',help='Use to *NOT* plot the raw light curves of star 1 and star 2. Default is True',action='store_false')
parser.add_argument('-am_limit','--am_limit',help='Can set an airmass limit here such that only points below the limit are plotted, default=3',type=float,default=3.0)
parser.add_argument('-single','--single',help='Use this argument to plot only the overall sky and FWHM for target, not comparison which is less confusing for paper plots',action='store_true')
parser.add_argument('-s','--save_figure',help='Use if wanting to save the outputted figure, default is false',action='store_true')
args = parser.parse_args()


# Keep a running tally of how many subplots we want
nsubplots = 0

try:
    am = pickle.load(open('airmass.pickle','rb'))
    nsubplots += 1

except:
    am = np.loadtxt('airmass.txt')
    nsubplots += 1

am_cut = np.max(np.where(am <= args.am_limit)[0])
am = am[:am_cut]
print('Nframes = ',len(am))

if args.x1 is not None:

    x1 = pickle.load(open(args.x1,'rb'))[:am_cut]
    x2 = pickle.load(open(args.x2,'rb'))[:am_cut]
    nsubplots += 1

if args.y1 is not None:
    y1 = pickle.load(open(args.y1,'rb'))[:am_cut]
    y2 = pickle.load(open(args.y2,'rb'))[:am_cut]

    # Pixel shift
    nsubplots += 1

else:

    y1 = y2 = None

    # Using trace to plot x shift
    nsubplots += 1
    print('Not plotting pixel shift in y')


if not args.fwhm:
    fwhm = pickle.load(open('fwhm_1.pickle','rb'))[:am_cut]
    fwhm2 = pickle.load(open('fwhm_2.pickle','rb'))[:am_cut]
    nsubplots += 1
else:
    fwhm = None
    fwhm2 = None


if not args.max:
    max1 = pickle.load(open('max_counts_1.pickle','rb')).max(axis=1)[:am_cut]
    max2 = pickle.load(open('max_counts_2.pickle','rb')).max(axis=1)[:am_cut]
    nsubplots += 1
else:
    max1 = None
    max2 = None

try:
	mjd = pickle.load(open('mjd_time.pickle','rb'))[:am_cut]
	time = mjd - int(mjd[0])
except:
	time = pickle.load(open('time.pickle','rb'))[:am_cut]
	time = time - int(time[0])

s1 = pickle.load(open('star1_flux.pickle','rb'))[:am_cut]
s2 = pickle.load(open('star2_flux.pickle','rb'))[:am_cut]

if args.raw_lcs:
    print('Plotting raw light curves')
    nsubplots += 1
    try:
        exposure_times = np.loadtxt('exposure_times.txt')[:am_cut]
    except:
        exposure_times = pickle.load(open('exposure_times.pickle','rb'))[:am_cut]
        #print 'No exposure times loaded, may generate problems when plotting raw light curves'

#nsubplots += 1

try:
    sky1 = pickle.load(open('sky1.pickle','rb'))[:am_cut]
    sky2 = pickle.load(open('sky2.pickle','rb'))[:am_cut]

    single_sky = True

    nsubplots += 1

except:
    sky1 = pickle.load(open('sky_avg_star1.pickle','rb'))[:am_cut]
    sky2 = pickle.load(open('sky_avg_star2.pickle','rb'))[:am_cut]

    sky1_left = pickle.load(open('sky_left_star1.pickle','rb'))[:am_cut]
    sky1_right = pickle.load(open('sky_right_star1.pickle','rb'))[:am_cut]

    sky2_left = pickle.load(open('sky_left_star2.pickle','rb'))[:am_cut]
    sky2_right = pickle.load(open('sky_right_star2.pickle','rb'))[:am_cut]

    single_sky = False

    nsubplots += 3


trace1 = pickle.load(open('x_positions_1.pickle','rb'))[:am_cut]
trace2 = pickle.load(open('x_positions_2.pickle','rb'))[:am_cut]

if args.x1 is None:
    x1 = trace1.mean(axis=1) - trace1.mean(axis=1)[0]
    x2 = trace2.mean(axis=1) - trace2.mean(axis=1)[0]

    nsubplots += 1

rotation1 = np.array([x[100] - x[-100] for x in trace1])
rotation2 = np.array([x[100] - x[-100] for x in trace2])

nsubplots += 1





if args.model1 is None and args.model2 is None:
    print('Not plotting white light model fit')
    plot_model = False
    white_light = np.loadtxt('white_light.txt')
    wl_flux = white_light[:,1][:am_cut]
    wl_error = white_light[:,2][:am_cut]

    nsubplots += 1

else:
    wl_flux = pickle.load(open(args.wly,'r'))[:am_cut]
    wl_error = pickle.load(open(args.wle,'r'))[:am_cut]
    wl_time = pickle.load(open(args.wlx,'r'))[:am_cut]
    plot_model = True
    nsubplots += 2

    print('Need to add functionality to plot best fitting models')

heights = [1]*(nsubplots-1)+[2]

fig = plt.figure(figsize=(12,22))
gs = gridspec.GridSpec(nsubplots, 1, height_ratios=heights)
panel = 0 # Keep running tally of what panel we're in
lower_x = time[0] - time[0]/100.
upper_x = time[-1] + time[-1]/100.

# Airmass
if am is not None:
    ax1 = plt.subplot(gs[panel])
    ax1.plot(time,am,'k.')
    plt.xticks(visible=False)
    ax1.set_ylabel('Airmass')
    lower_y = ax1.get_ylim()[0] - ax1.get_ylim()[0]/10.
    upper_y = ax1.get_ylim()[1] + ax1.get_ylim()[1]/10.
    #ax1.set_ylim(lower_y,upper_y)
    ax1.set_ylim(1.01,upper_y)
    ax1.set_xlim(lower_x,upper_x)
    panel += 1

# Pixel shifts

# Pixel shift in x
ax2 = plt.subplot(gs[panel])
ax2.plot(time,x1,'bx')
ax2.plot(time,x2,'r+')
ax2.set_ylabel('Pixel shift \nin X')
plt.xticks(visible=False)
lower_y2 = ax2.get_ylim()[0] - ax2.get_ylim()[0]/10.
upper_y2 = ax2.get_ylim()[1] + ax2.get_ylim()[1]/10.

if lower_y2 < -5:
    lower_y2 = -5
if upper_y2 > 5:
    upper_y2 = 5

# ax2.set_ylim(lower_y2,upper_y2)
ax2.set_ylim(-4.9,4.9)
#ax2.set_ylim(-0.9,4.9)
ax2.set_xlim(lower_x,upper_x)

panel += 1

if y1 is not None:

    # Pixel shift in y
    ax3 = plt.subplot(gs[panel])
    ax3.plot(time,y1,'bx')
    ax3.plot(time,y2,'r+')
    ax3.set_ylabel('Pixel shift \nin Y')
    lower_y3 = ax3.get_ylim()[0] - ax3.get_ylim()[0]/100.
    upper_y3 = ax3.get_ylim()[1] + ax3.get_ylim()[1]/100.

    if lower_y3 < -5:
        lower_y3 = -5
    if upper_y3 > 5:
        upper_y3 = 5

    # ax3.set_ylim(lower_y3,upper_y3)
    ax3.set_ylim(-4.9,4.9)
    plt.xticks(visible=False)
    ax3.set_xlim(lower_x,upper_x)

    panel += 1

# Rotation
ax4 = plt.subplot(gs[panel])
ax4.plot(mjd-int(mjd[0]),rotation1,'bx')
ax4.plot(mjd-int(mjd[0]),rotation2,'r+')
ax4.set_ylabel('$X_{4600A} - X_{8900A}$\n (pixels)')
#ax4.set_ylabel('$X(100) - X(-100)$\n (pixels)')
lower_y4 = ax4.get_ylim()[0] - ax4.get_ylim()[0]/10.
upper_y4 = ax4.get_ylim()[1] + ax4.get_ylim()[1]/10.
ax4.set_ylim(lower_y4,upper_y4)
plt.xticks(visible=False)
ax4.set_xlim(lower_x,upper_x)

panel += 1

# FWHM
if fwhm is not None:
    axfwhm = plt.subplot(gs[panel])
    if not args.single:
        axfwhm.plot(time,fwhm*0.253,'bx')
    else:
        axfwhm.plot(time,fwhm*0.253,'k.')
    if fwhm2 is not None and not args.single:
        axfwhm.plot(time,fwhm2*0.253,'r+')


    axfwhm.set_ylabel('FWHM \n(arcsec)')
    plt.xticks(visible=False)
    lower_yaxfwhm = axfwhm.get_ylim()[0] - 6*axfwhm.get_ylim()[0]/100.
    upper_yaxfwhm = axfwhm.get_ylim()[1] + 6*axfwhm.get_ylim()[1]/100.

    axfwhm.set_ylim(lower_yaxfwhm,upper_yaxfwhm)
    #start, end = axfwhm.get_ylim()
    #axfwhm.yaxis.set_ticks(np.arange(round(start,1), round(end,1), 0.2))

    axfwhm.set_xlim(lower_x,upper_x)

    panel += 1

# Max counts
if max1 is not None:
    ax_max = plt.subplot(gs[panel])
    ax_max.plot(time,max1,'bx')
    ax_max.plot(time,max2,'r+')
    ax_max.set_ylabel('Max counts \n(ADU)')
    plt.xticks(visible=False)
    lower_yax_max = ax_max.get_ylim()[0] - ax_max.get_ylim()[0]/100.
    upper_yax_max = ax_max.get_ylim()[1] + ax_max.get_ylim()[1]/100.
    ax_max.set_ylim(lower_yax_max,upper_yax_max)
    ax_max.set_xlim(lower_x,upper_x)

    panel += 1

# Raw light curves
if args.raw_lcs:
    ax_raw = plt.subplot(gs[panel])

    if exposure_times is not None:
         raw_lc_1 = s1.sum(axis=1)/exposure_times
         raw_lc_2 = s2.sum(axis=1)/exposure_times
         ax_raw.set_ylabel('Normalised \nflux')

    else:
         raw_lc_1 = s1.sum(axis=1)
         raw_lc_2 = s2.sum(axis=1)
         ax_raw.set_ylabel('Normalised \nflux')

    ax_raw.plot(time,raw_lc_1/np.median(raw_lc_1),'bx')
    ax_raw.plot(time,raw_lc_2/np.median(raw_lc_2),'r+')
    plt.xticks(visible=False)

    lower_yr = ax_raw.get_ylim()[0] - 8*ax_raw.get_ylim()[0]/100.
    upper_yr = ax_raw.get_ylim()[1] + 8*ax_raw.get_ylim()[1]/100.
    ax_raw.set_ylim(lower_yr,upper_yr)
    #ax_raw.set_ylim(0.51,1.09)
    ax_raw.set_xlim(lower_x,upper_x)

    panel += 1

# Sky background, average
ax_sky = plt.subplot(gs[panel])
if exposure_times is not None:
    sky_plot1 = sky1.sum(axis=1)/exposure_times
    if not args.single:
        sky_plot2 = sky2.sum(axis=1)/exposure_times
    ax_sky.set_ylabel('Normalised \nsky background')
else:
    sky_plot1 = sky1.sum(axis=1)
    if not args.single:
        sky_plot2 = sky2.sum(axis=1)
    ax_sky.set_ylabel('sky \ncounts')

if not args.single:
    ax_sky.plot(time,sky_plot1,'bx')#/np.median(sky1.sum(axis=1)),'bx')
    ax_sky.plot(time,sky_plot2,'r+')#/np.median(sky2.sum(axis=1)),'r+')
else:
    ax_sky.plot(time,sky_plot1/np.median(sky_plot1),'k.')
plt.xticks(visible=False)

lower_ysky = ax_sky.get_ylim()[0] - 8*ax_sky.get_ylim()[0]/100.
upper_ysky = ax_sky.get_ylim()[1] + 8*ax_sky.get_ylim()[1]/100.
ax_sky.set_ylim(lower_ysky,upper_ysky)
#ax_sky.set_ylim(0.0051,0.034)
ax_sky.set_xlim(lower_x,upper_x)

panel += 1




if not single_sky and not args.single:
    # Sky background, left hand background regions
    ax_sky_left = plt.subplot(gs[panel])
    if exposure_times is not None:
        sky_plot_left_s1 = sky1_left.sum(axis=1)/exposure_times
        sky_plot_left_s2 = sky2_left.sum(axis=1)/exposure_times
        ax_sky_left.set_ylabel('Left \nsky counts $s^{-1}$')
    else:
        sky_plot_left_s1 = sky1_left.sum(axis=1)
        sky_plot_left_s2 = sky2_left.sum(axis=1)
        ax_sky_left.set_ylabel('Left \nsky counts')

    ax_sky_left.plot(time,sky_plot_left_s1,'bx')#/np.median(sky1_left.sum(axis=1)),'bx')
    ax_sky_left.plot(time,sky_plot_left_s2,'r+')#/np.median(sky1_right.sum(axis=1)),'r+')
    plt.xticks(visible=False)

    lower_ysky_left = ax_sky_left.get_ylim()[0] - ax_sky_left.get_ylim()[0]/100.
    upper_ysky_left = ax_sky_left.get_ylim()[1] + ax_sky_left.get_ylim()[1]/100.
    ax_sky_left.set_ylim(lower_ysky_left,upper_ysky_left)
    #ax_sky.set_ylim(0.0051,0.034)
    ax_sky_left.set_xlim(lower_x,upper_x)

    panel += 1

    # Sky background, right hand background regions
    ax_sky_right = plt.subplot(gs[panel])
    if exposure_times is not None:
        sky_plot_right_s1 = sky1_right.sum(axis=1)/exposure_times
        sky_plot_right_s2 = sky2_right.sum(axis=1)/exposure_times
        ax_sky_right.set_ylabel('Right \nsky counts $s^{-1}$')
    else:
        sky_plot_right_s1 = sky1_right.sum(axis=1)
        sky_plot_right_s2 = sky2_right.sum(axis=1)
        ax_sky_right.set_ylabel('Right \nsky counts')

    #ax_sky.plot(time,sky_plot_left_s2/np.median(sky2_left.sum(axis=1)),'bx')
    #ax_sky.plot(time,sky_plot_right_s2/np.median(sky2_right.sum(axis=1)),'r+')
    ax_sky_right.plot(time,sky_plot_right_s1,'bx')
    ax_sky_right.plot(time,sky_plot_right_s2,'r+')
    plt.xticks(visible=False)

    lower_ysky_right = ax_sky_right.get_ylim()[0] - ax_sky_right.get_ylim()[0]/100.
    upper_ysky_right = ax_sky_right.get_ylim()[1] + ax_sky_right.get_ylim()[1]/100.
    ax_sky_right.set_ylim(lower_ysky_right,upper_ysky_right)
    #ax_sky.set_ylim(0.0051,0.034)
    ax_sky_right.set_xlim(lower_x,upper_x)

    panel += 1




# White light curve
ax5 = plt.subplot(gs[panel])
plt.ylabel('Differential \nflux')
panel += 1

if plot_model:
    prod_model_gp = pickle.load(open(args.model1,'r'))[0]
    mu,std = prod_model_gp.calc_gp_component(wl_time,wl_flux,wl_error)
    ma_comp = prod_model_gp.calc(wl_time)

    if prod_model_gp.wn_kernel == 'scaling':
        combined_error_1sig = np.sqrt(std**2 + (wl_error*prod_model_gp.sigma.currVal)**2)
        combined_error_3sig = 3*np.sqrt((std)**2 + (wl_error*prod_model_gp.sigma.currVal)**2)
    if prod_model_gp.wn_kernel == 'white_noise':
        raise SystemExit('white_noise kernel plotting not yet implemented')



    plt.errorbar(wl_time,wl_flux,yerr=wl_error,fmt='None',ecolor='k',capsize=0)
    plt.xticks(visible=False)
    plt.plot(wl_time,ma_comp+mu,color='g')
    lower_y5 = ax5.get_ylim()[0] - ax5.get_ylim()[0]/100.
    upper_y5 = ax5.get_ylim()[1] + ax5.get_ylim()[1]/100.
    ax5.set_ylim(lower_y5,upper_y5)
    ax5.set_xlim(lower_x,upper_x)


    # Plot of residuals to white light fit
    ax6 = plt.subplot(gs[panel])
    ax6.plot(wl_time,wl_flux-ma_comp-mu,'ko',fillstyle='none')
    plt.fill_between(wl_time,combined_error_3sig,-combined_error_3sig, where=None, color='lightgrey',alpha=1)
    plt.fill_between(wl_time,combined_error_1sig,-combined_error_1sig, where=None, color='grey',alpha=1)
    #lower_y6 = ax6.get_ylim()[0] - ax6.get_ylim()[0]/100.
    #upper_y6 = ax6.get_ylim()[1] + ax6.get_ylim()[1]/100.
    #ax6.set_ylim(lower_y6,upper_y6)
    ax6.set_ylim(-0.0039,0.0039)
    plt.axhline(0,ls='--',color='k')
    ax6.set_xlim(lower_x,upper_x)


else:
    #plt.errorbar(time,wl_flux,yerr=wl_error,fmt='None',ecolor='k',capsize=0)
    plt.plot(time,wl_flux,'k.')
    lower_y5 = ax5.get_ylim()[0] - ax5.get_ylim()[0]/100.
    upper_y5 = ax5.get_ylim()[1] + ax5.get_ylim()[1]/100.
    ax5.set_ylim(lower_y5,upper_y5)
    #ax5.set_ylim(lower_y5,1.019)
    ax5.set_xlim(lower_x,upper_x)


plt.subplots_adjust(hspace=0)
plt.xlabel('Time (MJD - %d)'%int(mjd[0]))

if args.save_figure:
    plt.savefig('ancillary_plots.pdf',bbox_inches='tight')

plt.show()
