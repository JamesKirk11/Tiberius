#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk 

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import argparse
from scipy.signal import medfilt
from scipy.ndimage import filters
from scipy.stats import median_abs_deviation as mad
import pickle

# Prevent matplotlib plotting frames upside down
plt.rcParams['image.origin'] = 'lower'


parser = argparse.ArgumentParser()
parser.add_argument('flatslist', help="""Enter list of flats file names, created through ls > flat.lis in the command line""")
parser.add_argument('-v','--verbose',help="""Display the image of each frame before combining it.""",action='store_true')
parser.add_argument('-b','--bias_frame',help="""Define the bias frame.""")
parser.add_argument('-inst','--instrument',help="""Define the instrument used, either EFOSC or ACAM""")
parser.add_argument('-c','--clobber',help="""Need this argument to save resulting fits file, default = False""",action='store_true')
parser.add_argument('-s','--saturation_limit',help="""Use this to exclude frames with counts above a satruation threshold. Default is 55000""",type=int,default=55000)
args = parser.parse_args()

# ~ if args.verbose:
    # ~ plt.ion()

if args.instrument != 'EFOSC' and args.instrument != 'ACAM':
    raise NameError('Currently only set up to deal with ACAM or EFOSC data')

flats_files = np.loadtxt(args.flatslist,str)

master_bias_data = fits.open(args.bias_frame)[0].data

# Find how many windows we're dealing with
if args.instrument == 'EFOSC':
    nwin = 1

if args.instrument == 'ACAM':
    test = fits.open(flats_files[0])
    nwin = len(test) - 1

def bias_subtraction(master_bias,data):
    return data - master_bias

def combine_flats_1window(flats_list,master_bias,instrument,sat_limit,verbose=False):

    flat_data = []

    if verbose:
        plt.figure()

    for f in flats_list:

        i = fits.open(f)

        if instrument == 'ACAM':
            data_frame = i[1].data
        if instrument == 'EFOSC':
            data_frame = i[0].data

        print('File = ',f,'; Mean = ',np.mean(data_frame),'; Shape = ',np.shape(data_frame), 'Max count = ',np.max(data_frame[:,20:-20])) # Need to clip extreme edges which can have very high counts
        if np.max(data_frame[:,20:-20]) <= sat_limit:
            flat_data.append(data_frame-master_bias)
        else:
            print('--- ingoring potentially saturated frame')

        if verbose:

            vmin,vmax = np.nanpercentile(data_frame,[10,90])
            plt.imshow(data_frame,vmin=vmin,vmax=vmax,cmap='hot')
            plt.xlabel("X pixel")
            plt.ylabel("Y pixel")
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.5)
            plt.clf()
        i.close()

    flat_data = np.array(flat_data)

    mean_combine = np.mean(flat_data,axis=0)
    return mean_combine

def combine_flats_2windows(flats_list,master_bias,sat_limit,verbose=False):

    flat_data = [[],[]]

    if verbose:
        plt.figure()

    for f in flats_list:
        i = fits.open(f)

        for level in range(1,len(i)):
            data_frame = i[level].data

            flat_data[level-1].append(data_frame-master_bias[level-1])


            if verbose:
                plt.subplot(1,2,level)
                plt.imshow(data_frame)
                plt.colorbar()

            print('File = %s ; Mean (window %d) = %f ; Shape (window %d) = %s ; Max count = %d '%(f,level,np.mean(data_frame),level,np.shape(data_frame),np.max(data_frame[:,20:-20])))
            if np.max(data_frame[:,20:-20]) <= sat_limit:
                flat_data[level-1].append(data_frame-master_bias[level-1])
            else:
                print('--- ingoring potentially saturated frame')

            if verbose:
                plt.subplot(1,2,level)
                vmin,vmax = np.nanpercentile(data_frame,[10,90])
                plt.imshow(data_frame,vmin=vmin,vmax=vmax,cmap='hot')
                if level == 1:
                    plt.xlabel("X pixel")
                    plt.ylabel("Y pixel")
                else:
                    plt.yticks(visible=False)

        if verbose:
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.5)
            plt.clf()

        i.close()

    flat_data = np.array(flat_data)

    mean_combine = [np.mean(flat_data[0],axis=0),np.mean(flat_data[1],axis=0)]

    return mean_combine

def save_fits(data,filename,clobber=False):
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename,overwrite=clobber)
    return


def test_smoothing_widths(flat_data,nwindows):

    """Run a series of median filters with differing box widths to find optimal width."""

    if nwindows > 1:
        nrows,ncols1 = np.shape(flat_data[0])
        _,ncols2 = np.shape(flat_data[1])
        std1 = []
        std2 = []
    else:
        nrows,ncols = np.shape(flat_data)
        std = []

    for i in range(1,nrows//10,2):

        box_width = i

        if nwindows == 1:
            MF = medfilt(flat_data[:,ncols//2],box_width)
            residuals = flat_data[:,ncols//2]/MF
            # rms.append(np.sqrt(np.mean(residuals**2)))
            std.append(np.std(residuals))

        else:
            MF1 = medfilt(flat_data[0][:,ncols1//2],box_width) # just use single column for plotting
            MF2 = medfilt(flat_data[1][:,ncols2//2],box_width)

            residuals1 = flat_data[0][:,ncols1//2]/MF1
            residuals2 = flat_data[1][:,ncols2//2]/MF2

            std1.append(np.std(residuals1))
            std2.append(np.std(residuals2))

    print("Plotting bin width vs. standard deviation - MAKE NOTE OF DESIRED BIN WIDTH!")
    plt.figure()
    plt.xlabel('Bin width')
    plt.ylabel('Standard deviation')
    if nwindows == 1:
        plt.plot(np.arange(1,nrows//10,2),std,'ko')
    else:
        plt.plot(np.arange(1,nrows//10,2),std1,'bo',label='Window 1')
        plt.plot(np.arange(1,nrows//10,2),std2,'ro',label='Window 2')
        plt.legend(loc='upper left')

    plt.show()
    return

def median_smooth(flat_data,name,nwindows,box_width,clobber):

    """Smooth the flat using a running median, and evaluated at each column individually"""

    if nwindows > 1:
        nrows1,ncols1 = np.shape(flat_data[0])
        nrows2,ncols2 = np.shape(flat_data[1])

        MF1 = medfilt(flat_data[0][:,ncols1//2],box_width) # just use single column for plotting
        MF2 = medfilt(flat_data[1][:,ncols2//2],box_width)
    else:
        nrows,ncols = np.shape(flat_data)
        MF = medfilt(flat_data[:,ncols//2],box_width)

    plt.figure(figsize=(10,8))
    plt.subplot(211)

    if nwindows == 1:
        plt.plot(flat_data[:,ncols//2],lw=4)
        plt.plot(MF,'r',lw=2)
    else:
        plt.plot(flat_data[0][:,ncols1//2],lw=4,color='b',label='window 1')
        plt.plot(MF1,'r',lw=2)
        plt.plot(flat_data[1][:,ncols2//2],lw=4,color='k',label='window 2')
        plt.plot(MF2,'r',lw=2)
        plt.legend(loc='upper left')


    plt.xticks(visible=False)
    plt.ylabel('Counts')

    plt.subplot(212)
    if nwindows == 1:
        plt.plot(flat_data[:,ncols//2]/MF)
    else:
        plt.plot(flat_data[0][:,ncols1//2]/MF1,color='b',label='window 1')
        plt.plot(flat_data[1][:,ncols2//2]/MF2,color='k',label='window 2')
        plt.legend(loc='upper left')

    plt.xlabel('Y pixel')
    plt.ylabel('Residuals')
    plt.ylim(0.951,1.049)
    plt.subplots_adjust(hspace=0)
    plt.suptitle('Median filter, column by column')

    plt.show(block=False)
    plt.pause(5)
    plt.close()

    if nwindows == 1:
        # Now find running median for each column individually
        MF_reshaped =  np.array([medfilt(flat_data[:,i],box_width) for i in range(ncols)]).transpose()

        MF_reshaped[MF_reshaped == 0] = 1 # have to replace 0s which occur near the edges, the bias over subtracts to give negative values which mess up this line

        divided_sky_flat = flat_data/MF_reshaped
        normalised_sky_flat = divided_sky_flat/divided_sky_flat.mean()

    else:
        MF1_reshaped = np.array([medfilt(flat_data[0][:,i],box_width) for i in range(ncols1)]).transpose()
        divided_sky_flat1 = flat_data[0]/MF1_reshaped
        normalised_sky_flat1 = divided_sky_flat1/divided_sky_flat1.mean()

        MF2_reshaped = np.array([medfilt(flat_data[1][:,i],box_width) for i in range(ncols2)]).transpose()
        divided_sky_flat2 = flat_data[1]/MF2_reshaped
        normalised_sky_flat2 = divided_sky_flat2/divided_sky_flat2.mean()

        normalised_sky_flat = np.array([normalised_sky_flat1,normalised_sky_flat2])


    plt.figure(figsize=(10,10))
    if nwindows == 1:
        plt.imshow(normalised_sky_flat,vmin=0.95,vmax=1.05)
    else:
        plt.subplot(121)
        plt.imshow(normalised_sky_flat[0],vmin=0.95,vmax=1.05)
        plt.subplot(122)
        plt.imshow(normalised_sky_flat[1],vmin=0.95,vmax=1.05)

    plt.suptitle('Median filter, column by column')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.colorbar()
    plt.show(block=False)

    plt.pause(5)
    plt.close()

    hdu = fits.PrimaryHDU(normalised_sky_flat)
    hdu.writeto('master_flat_'+name+'_median_filter_column_by_column_box_width_%d_FITTED.fits'%box_width,overwrite=clobber)


def gaussian_smooth(flat_data,name,nwindows,inst,clobber):

    """Smooth the flat using a Gaussian filter"""

    if inst == 'ACAM':
        sigma = 1
    if inst == 'EFOSC':
        sigma = 0.4 # this is found through inspection by eye

    if nwindows > 1:
        nrows1,ncols1 = np.shape(flat_data[0])
        nrows2,ncols2 = np.shape(flat_data[1])

        GF1 = filters.gaussian_filter(flat_data[0],sigma)
        GF2 = filters.gaussian_filter(flat_data[1],sigma)
    else:
        nrows,ncols = np.shape(flat_data)
        GF = filters.gaussian_filter(flat_data,sigma)


    # Plot Gaussian filter evaluated at a column in centre of CCD
    plt.figure(figsize=(10,8))
    plt.subplot(211)

    if nwindows == 1:
        plt.plot(flat_data[:,ncols//2],lw=6)
        plt.plot(GF[:,ncols//2],'r',lw=2)
    else:
        plt.plot(flat_data[0][:,ncols1//2],lw=4,color='b',label='window 1')
        plt.plot(GF1[:,ncols1//2],'r',lw=2)
        plt.plot(flat_data[1][:,ncols2//2],lw=4,color='k',label='window 2')
        plt.plot(GF2[:,ncols2//2],'r',lw=2)
        plt.legend(loc='upper left')

    plt.xticks(visible=False)
    plt.ylabel('Counts')

    plt.subplot(212)
    if nwindows == 1:
        plt.plot(flat_data[:,ncols//2]/GF[:,ncols//2])
    else:
        plt.plot(flat_data[0][:,ncols1//2]/GF1[:,ncols1//2],color='b',label='window 1')
        plt.plot(flat_data[1][:,ncols2//2]/GF2[:,ncols2//2],color='k',label='window 2')
        plt.legend(loc='upper left')

    plt.xlabel('Y pixel')
    plt.ylabel('Residuals')
    plt.ylim(0.951,1.049)
    plt.subplots_adjust(hspace=0)
    plt.suptitle('Gaussian filter, example column')

    plt.show(block=False)
    plt.pause(5)
    plt.close()


    if nwindows == 1:
        divided_sky_flat = flat_data/GF
        normalised_sky_flat = divided_sky_flat/divided_sky_flat.mean()

    else:
        divided_sky_flat1 = flat_data[0]/GF1
        normalised_sky_flat1 = divided_sky_flat1/divided_sky_flat1.mean()

        divided_sky_flat2 = flat_data[1]/GF2
        normalised_sky_flat2 = divided_sky_flat2/divided_sky_flat2.mean()

        normalised_sky_flat = np.array([normalised_sky_flat1,normalised_sky_flat2])


    plt.figure(figsize=(10,10))
    if nwindows == 1:
        plt.imshow(normalised_sky_flat,vmin=0.95,vmax=1.05)
    else:
        plt.subplot(121)
        plt.imshow(normalised_sky_flat[0],vmin=0.95,vmax=1.05)
        plt.subplot(122)
        plt.imshow(normalised_sky_flat[1],vmin=0.95,vmax=1.05)

    plt.suptitle('Gaussian filter')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.colorbar()
    plt.show(block=False)

    plt.pause(5)
    plt.close()

    hdu = fits.PrimaryHDU(normalised_sky_flat)
    hdu.writeto('master_flat_'+name+'_Gaussian_filter_FITTED.fits',overwrite=clobber)


def pixel_mask_tight(frame,save):
    """A tighter definition of a bad pixel mask using 5 median absolute devitations from the median, computed column by column (along dispersion direction)"""
 
    medians = np.median(frame,axis=0)
    mads = mad(frame,axis=0,scale='normal')
    good_pixels = []

    for i,row in enumerate(frame):
        good_pixels.append(((row >= medians-5*mads) & (row <= medians+5*mads)))
    
    good_pixels = np.array(good_pixels)
    bad_pixels = ~good_pixels
    
    percentage_bad_pixels = 100*len(np.where(bad_pixels)[0])/(frame.shape[0]*frame.shape[1])
    print("Percentage of pixels deemed bad by medians and mads = %.2f%%"%percentage_bad_pixels)
    
    plt.figure()
    plt.imshow(bad_pixels)
    plt.title("Bad pixel mask using medians and MADS")
    plt.show()
    
    if save:
        pickle.dump(bad_pixels,open("bad_pixel_mask_tight.pickle","wb"))
        return
    
    return bad_pixels

    
def pixel_mask_loose(frame,save):
    """A looser definition of the bad pixel mask using 5 standard deviations from the global mean"""
    
    good_pixels = ((frame >= np.mean(frame)-5*np.std(frame)) & (frame <= np.mean(frame)+5*np.std(frame)))
    bad_pixels = ~good_pixels

    percentage_bad_pixels = 100*len(np.where(bad_pixels)[0])/(frame.shape[0]*frame.shape[1])
    print("Percentage of pixels deemed bad by means and stds = %.2f%%"%percentage_bad_pixels)
    
    plt.figure()
    plt.imshow(bad_pixels)
    plt.title("Bad pixel mask using means and stds")
    plt.show()
    
    if save:
        pickle.dump(bad_pixels,open("bad_pixel_mask_loose.pickle","wb"))
        return
    
    return bad_pixels
    

def pixel_mask_medfilt(frame,save):
    """A function that uses median filters along the rows/cross-dispersion direction to locate outliers. Can be more effective."""
    
    good_pixels = []

    cut_off = 5
    for i,row in enumerate(frame):
        MF = medfilt(row,5)
        residuals = row-MF
        good_pixels.append(((residuals >= -10*mad(residuals,scale='normal')) & (residuals <= 10*mad(residuals,scale='normal'))))

    good_pixels = np.array(good_pixels)
    bad_pixels = ~good_pixels
    
    percentage_bad_pixels = 100*len(np.where(bad_pixels)[0])/(frame.shape[0]*frame.shape[1])
    print("Percentage of pixels deemed bad by median filter = %.2f%%"%percentage_bad_pixels)
    
    plt.figure()
    plt.imshow(bad_pixels)
    plt.title("Bad pixel mask using median filter")
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.show()
    
    if save:
        pickle.dump(bad_pixels,open("bad_pixel_mask_medfilt.pickle","wb"))
        return
    
    return bad_pixels
    
    
if nwin == 1:
    f = combine_flats_1window(flats_files,master_bias_data,args.instrument,args.saturation_limit,args.verbose)
else:
    f = combine_flats_2windows(flats_files,master_bias_data,args.saturation_limit,args.verbose)

save_fits(f/np.median(f),'master_flat_'+args.flatslist+'.fits',args.clobber)

test_smoothing_widths(np.array(f),nwin)

chosen_box_width = int(input("Enter desired box width for running median (must be ODD INTEGER): "))

median_smooth(np.array(f),args.flatslist,nwin,chosen_box_width,args.clobber)

## Note: Gaussian smooth not currently used, despite good perfomance, as it removes features in x as well as y
# gaussian_smooth(np.array(f),args.flatslist,nwin,args.instrument,args.clobber)

## Plot master flat
plt.figure()
if nwin == 1:
    vmin,vmax = np.nanpercentile(f,[10,90])
    plt.imshow(f,vmin=vmin,vmax=vmax,cmap='hot')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
else:
    plt.subplot(121)
    
    vmin,vmax = np.nanpercentile(f[0],[10,90])
    plt.imshow(f[0],vmin=vmin,vmax=vmax,cmap='hot')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    
    plt.subplot(122)
    vmin,vmax = np.nanpercentile(f[1],[10,90])
    plt.imshow(f[1],vmin=vmin,vmax=vmax,cmap='hot')

plt.suptitle('Master flat before response fitting')
plt.colorbar()
plt.show()


# Make bad pixel masks
if nwin == 1:
    pixel_mask_tight(f,True)
    pixel_mask_loose(f,True)
    pixel_mask_medfilt(f,True)
else:
    pmt = np.array([pixel_mask_tight(f[0],False),pixel_mask_tight(f[1],False)])
    pickle.dump(pmt,open("bad_pixel_mask_tight.pickle","wb"))
    
    pml = np.array([pixel_mask_loose(f[0],False),pixel_mask_loose(f[1],False)])
    pickle.dump(pml,open("bad_pixel_mask_loose.pickle","wb"))
    
    pmmf = np.array([pixel_mask_medfilt(f[0],False),pixel_mask_medfilt(f[1],False)])
    pickle.dump(pmmf,open("bad_pixel_mask_medfilt.pickle","wb"))
    
