#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk 

from astropy.io import fits
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import numpy as np

# Prevent matplotlib plotting frames upside down
plt.rcParams['image.origin'] = 'lower'


parser = argparse.ArgumentParser(description='Plot a through slit image along with traces and frames for an extraction. NOTE: currently only set up for ACAM.')
parser.add_argument('-x1','--x1',help="parse the pickled x positions of the first trace. Note can parse 2 files if a separate extraction was run in the red end.",nargs="+")
parser.add_argument('-x2','--x2',help="parse the pickled x positions of the second trace. Note can parse 2 files if a separate extraction was run in the red end.",nargs="+")
parser.add_argument('-apw','--aperture_widths',help="Define the widths of the target apertures used. Can parse > 1 aperture width.",nargs="+",type=int)
parser.add_argument('-bkw','--background_widths',help="Define the widths of the background apertures used. Can parse > 1 aperture width.",nargs="+",type=int)
parser.add_argument('-bko','--background_offset',help="Define the offset between the background and target apertures used. Can parse > 1 aperture width.",nargs="+",type=int)
parser.add_argument("-thru",'--thru_slit',help="parse the fits through slit image if wanting to plot this in addition to a science frame")
parser.add_argument("-sci","--science",help="parse the science frame for plotting")
parser.add_argument("-bias","--bias",help="parse the bias frame")
parser.add_argument("-flat","--flat",help="parse the flat frame")
parser.add_argument("-sci_lims","--science_limits",help="parse a minimum and maximum scaling for science plotting if wanting to override default. IMPORTANT: must use the first science frame of the night as the first trace locations of the night are used.",nargs="+",type=float)
parser.add_argument("-thru_lims","--thru_slit_limits",help="parse a minimum and maximum scaling for thru slit plotting if wanting to override default.",nargs="+",type=float)
parser.add_argument("-rw","--row_min",help="parse minimum row at which trace was extracted (this also cuts the plot to this minimum row).",type=int)
parser.add_argument("-rm","--row_max",help="parse maximum row at which trace was extracted (this also cuts the plot to this maximum row).",type=int)
parser.add_argument("-s","--save_figure",help="use to save the resulting plot to a png file.",action="store_true")
args = parser.parse_args()

science = fits.open(args.science)

if args.bias is not None:
    bias = fits.open(args.bias)[0].data
if args.flat is not None:
    flat = fits.open(args.flat)[0].data
else:
    flat = np.ones_like(science[1].data)

nwindows = len(science) - 1

if args.thru_slit is not None:
    thru_slit = fits.open(args.thru_slit)
    nplot_rows = 2
else:
    thru_slit = None
    nplot_rows = 1

if nwindows == 2:
    nplot_cols = 2
else:
    nplot_cols = 1


# Plots limits
if args.science_limits is None:
    vmin_sci = 1e2
    vmax_sci = 5e3
else:
    vmin_sci = args.science_limits[0]
    vmax_sci = args.science_limits[1]

if args.thru_slit_limits is None:
    vmin_thru = 1e2
    vmax_thru = 5e3
else:
    vmin_thru = args.thru_slit_limits[0]
    vmax_thru = args.thru_slit_limits[1]

nextractions = len(args.x1)

aperture_widths = args.aperture_widths
background_widths = args.background_widths
background_offsets = args.background_offset

if len(aperture_widths) == 1:
    aperture_widths = aperture_widths*2

if len(background_widths) == 1:
    background_widths = background_widths*2

if len(background_offsets) == 1:
    background_offsets = background_offsets*2

# Load in trace locations
if nextractions > 1:
    x1_b = pickle.load(open(args.x1[0],'rb'))
    x1_r = pickle.load(open(args.x1[1],'rb'))

    x1 = np.hstack((x1_b[0],x1_r[0]))

    x2_b = pickle.load(open(args.x2[0],'rb'))
    x2_r = pickle.load(open(args.x2[1],'rb'))

    x2 = np.hstack((x2_b[0],x2_r[0]))
else:
    x1 = pickle.load(open(args.x1[0],'rb'))[0]
    x2 = pickle.load(open(args.x2[0],'rb'))[0]

rows = np.arange(args.row_min,args.row_max)

### Plot figure
# fig = plt.figure(figsize=(6,10))

# Make figure aspect ratio match science frame.
nrows,ncols = science[1].data.shape
fig_height = 8
fig_width = (len(rows)/ncols)*fig_height

if nwindows > 1:
    fig_width *= 2

plt.figure(figsize=(fig_width,fig_height))

# set up gridspec
if thru_slit is not None:
    height_ratio = [100/len(rows),1]
else:
    height_ratio = None

gs1 = gridspec.GridSpec(nplot_rows,nplot_cols,height_ratios=height_ratio)
gs1.update(wspace=0.02, hspace=0.02) # set the spacing between axes.

# record running number of plots
# plot_index = 1
plot_index = 0

ax1 = plt.subplot(gs1[0])

# Through slit image
if thru_slit is not None:
    ax1.set_ylim(840-50,840+50)
    ax1.imshow(thru_slit[1].data,vmin=vmin_thru,vmax=vmax_thru)

    # plot vertical line at star location
    # ax1.axvline(x1.mean())

    if nwindows > 1:
        plot_index += 1
        ax1a = plt.subplot(gs1[plot_index])
        ax1a.imshow(thru_slit[2].data,vmin=vmin_thru,vmax=vmax_thru)

        # vertical line at star locations
        ax1a.axvline(x2.mean())
        ax1a.set_ylim(840-50,840+50)
        ax1a.axis("off")
        ax1a.set_aspect('auto')

    else:
        # ax1.axvline(x2.mean())
        ax1.axis("off")
    ax1.set_aspect('auto')

    plot_index += 1


# Science image
if plot_index >= 1:
    ax2 = plt.subplot(gs1[plot_index])
else:
    ax2 = ax1

ax2.imshow((science[1].data-bias[0])/flat[0],vmin=vmin_sci,vmax=vmax_sci)
ax2.set_aspect('auto')

# target aperture
ax2.plot(np.ceil(x1+aperture_widths[0]//2),rows,color='b',lw=1)
ax2.plot(np.floor(x1-aperture_widths[0]//2),rows,color='b',lw=1)

# check whether buffer pixels have been used
# buffer_pixels = np.where(x1-aperture_widths[0]//2-background_offsets[0]-background_widths[0] <= 20)[0]

# background aperture
ax2.plot(np.ceil(x1+aperture_widths[0]//2)+background_offsets[0],rows,color='b',ls='--',lw=1)
ax2.plot(np.ceil(x1+aperture_widths[0]//2)+background_offsets[0]+background_widths[0],rows,color='b',ls='--',lw=1)
ax2.plot(np.floor(x1-aperture_widths[0]//2)-background_offsets[0],rows,color='b',ls='--',lw=1)
ax2.plot(np.floor(x1-aperture_widths[0]//2)-background_offsets[0]-background_widths[0],rows,color='b',ls='--',lw=1)

ax2.set_ylim(args.row_min,args.row_max)
ax2.set_ylabel('Y pixel')
ax2.set_xlabel('X pixel')

if nwindows > 1:
    plot_index += 1
    ax2a = plt.subplot(gs1[plot_index])
    ax2a.imshow((science[2].data-bias[1])/flat[1],vmin=vmin_sci,vmax=vmax_sci)
    # ax2a.plot(x2,np.arange(nrows),color='b',lw=1)

    # target aperture
    ax2a.plot(np.ceil(x2+aperture_widths[1]//2),rows,color='b',lw=1)
    ax2a.plot(np.floor(x2-aperture_widths[1]//2),rows,color='b',lw=1)

    # background aperture
    ax2a.plot(np.ceil(x2+aperture_widths[1]//2)+background_offsets[1],rows,color='b',ls='--',lw=1)
    ax2a.plot(np.ceil(x2+aperture_widths[1]//2)+background_offsets[1]+background_widths[1],rows,color='b',ls='--',lw=1)
    ax2a.plot(np.floor(x2-aperture_widths[1]//2)-background_offsets[1],rows,color='b',ls='--',lw=1)
    ax2a.plot(np.floor(x2-aperture_widths[1]//2)-background_offsets[1]-background_widths[1],rows,color='b',ls='--',lw=1)

    ax2a.set_ylim(args.row_min,args.row_max)
    ax2a.set_aspect('auto')
    ax2a.set_xlabel('X pixel')
else:
    # ax2.plot(x2,rows,color='b',lw=1)

    # target aperture
    ax2.plot(np.ceil(x2+aperture_widths[1]//2),rows,color='b',lw=1)
    ax2.plot(np.floor(x2-aperture_widths[1]//2),rows,color='b',lw=1)

    # background aperture
    ax2.plot(np.ceil(x2+aperture_widths[1]//2)+background_offsets[1],rows,color='b',ls='--',lw=1)
    ax2.plot(np.ceil(x2+aperture_widths[1]//2)+background_offsets[1]+background_widths[1],rows,color='b',ls='--',lw=1)
    ax2.plot(np.floor(x2-aperture_widths[1]//2)-background_offsets[1],rows,color='b',ls='--',lw=1)
    ax2.plot(np.floor(x2-aperture_widths[1]//2)-background_offsets[1]-background_widths[1],rows,color='b',ls='--',lw=1)


if args.save_figure:
    plt.savefig('extraction_frame.pdf',bbox_inches='tight')
plt.show()
