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

first_frame = fits.open(file_list[0])

if args.instrument == "EFOSC":
    nrows,ncols = np.shape(first_frame[0].data)
else:
    nrows,ncols = np.shape(first_frame[args.nwindows].data.T)

plt.figure()
for i,f in enumerate(file_list):
    data_frame = fits.open(f)
    if args.nwindows > 1:
        plt.subplot(311)
        plt.imshow(data_frame[1].data.T,vmin=data_frame[1].data.mean()*0.5,vmax=data_frame[1].data.mean()*2.0)#vmin=0,vmax=1000)
        plt.subplot(312)
        plt.imshow(data_frame[2].data.T,vmin=data_frame[2].data.mean()*0.5,vmax=data_frame[2].data.mean()*2.0)#vmin=0,vmax=1000)
        total_plots = 3
    elif args.nwindows == 1 and args.instrument == 'ACAM':
        plt.subplot(211)
        plt.imshow(data_frame[1].data,vmin=data_frame[1].data.mean()*0.5,vmax=data_frame[1].data.mean()*2.0)#vmin=1e3,vmax=5e3)
        total_plots = 2
    else:
        plt.subplot(211)
        plt.imshow(data_frame[0].data,vmin=data_frame[0].data.mean()*0.5,vmax=data_frame[0].data.mean()*2.0)#vmin=1e3,vmax=5e3)
        total_plots = 2

    plt.subplot(total_plots,1,total_plots)
    plt.plot(time[:i],flux[:i],'ko')
    plt.xlim(time[0],time[-1])
    plt.ylim(flux.min(),flux.max())
    plt.show(block=False)
    plt.pause(5e-6)
    plt.clf()
    data_frame.close()
