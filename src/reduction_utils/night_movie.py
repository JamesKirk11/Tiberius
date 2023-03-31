#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import matplotlib.animation as animation
import numpy as np
from pylab import *
import argparse
from astropy.io import fits
from matplotlib import gridspec

# Prevent matplotlib plotting frames upside down
plt.rcParams['image.origin'] = 'lower'


parser = argparse.ArgumentParser(description='')
parser.add_argument("science_list",help="Where is the location of the science files?")
parser.add_argument("-inst","--instrument",help="ACAM or EFOSC?")
parser.add_argument('-nwin','--nwindows',help='How many windows were used? (For ACAM only)',type=int,default=1)
parser.add_argument('-skip','--skip',help='Use to skip each N files, if dealing with many images',type=int)
args = parser.parse_args()

file_list = np.loadtxt(args.science_list,str)[::args.skip]


time, flux, flux_err = np.loadtxt('white_light.dat',unpack=True)
time = time[::args.skip]
flux = flux[::args.skip]
flux_err = flux_err[::args.skip]

nframes = len(time)
x = np.arange(nframes)

first_frame = fits.open(file_list[0])

if args.instrument == "EFOSC":
    nrows,ncols = np.shape(first_frame[0].data)
else:
    nrows,ncols = np.shape(first_frame[args.nwindows].data.T)

plt.figure()
plt.imshow(first_frame[1].data,vmin=args.vmin,vmax=args.vmax)
plt.show()

cont = input("Scaling ok? [vmin = %d, vmax = %d] (Y/n) : "%(args.vmin,args.vmax))
if cont == 'Y':
	pass
else:
	raise SystemExit

fig = plt.figure()

if args.nwindows == 1:
    ax = fig.add_subplot(211)
else:
    ax = fig.add_subplot(311)

if args.instrument == "EFOSC":
   im = ax.imshow(first_frame[0].data,vmin=args.vmin,vmax=args.vmax)
   plt.xticks(visible=False)
else:
   im = ax.imshow(first_frame[1].data.T,vmin=args.vmin,vmax=args.vmax)
   if args.nwindows > 1:
        ax1 = fig.add_subplot(312)
        im1 = ax1.imshow(first_frame[2].data.T,vmin=args.vmin,vmax=args.vmax)


first_frame.close()


if args.nwindows == 1:
    ax2 = fig.add_subplot(212)
else:
    ax2 = fig.add_subplot(313)

ax2.plot(x,flux,'ko',markerfacecolor='None')
ax2.set_xlabel('Frame number')
ax2.set_ylabel('Differential flux')


def init(i):

    f = file_list[i]

    print(f, i)
    frame = fits.open(f)

    if args.instrument == 'EFOSC':
        im.set_data(frame[0].data)
    else:
        im.set_data(frame[1].data.T)

        if args.nwindows > 1:
            vmin,vmax = np.nanpercentile(frame[2].data.T,[10,90])
            im1 = ax1.imshow(frame[2].data.T,vmin=vmin,vmax=vmax)

    frame.close()
    ax2.plot(x[i],flux[i],'bo')

    return im

anim = matplotlib.animation.FuncAnimation(fig, init, frames=nframes,repeat=False,save_count=0)
anim.save('night_movie.gif', writer='imagemagick', fps=10)
