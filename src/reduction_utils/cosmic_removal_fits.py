import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import argparse
from scipy.signal import medfilt
from scipy.ndimage import filters

# Prevent matplotlib plotting frames upside down
plt.rcParams['image.origin'] = 'lower'


parser = argparse.ArgumentParser()
parser.add_argument('scienceframes', help="""Enter list of science file names""")
parser.add_argument('-v','--verbose',help="""Display plots""",action='store_true')
parser.add_argument('-b','--bias_frame',help="""Define the bias frame.""")
parser.add_argument('-inst','--instrument',help="""Define the instrument used, either EFOSC or ACAM""") #just for EFOSC for now
parser.add_argument('-c','--clobber',help="""Need this argument to save resulting fits file, default = False""",action='store_true')
parser.add_argument('-s','--saturation_limit',help="""Use this to exclude frames with counts above a satruation threshold. Default is 55000""",type=int,default=75000)
args = parser.parse_args()
#args.clobber = True;

if args.verbose:
    plt.ion()

files = np.loadtxt(args.scienceframes,str)

if args.instrument == 'EFOSC':
    nwin = 1

if args.instrument == 'ACAM':
    test = fits.open(files[0])
    nwin = len(test) - 1

master_bias_data = fits.open(args.bias_frame)[0].data

def bias_subtraction(data):
    return data - master_bias_data

def combine_science_frames(frames_list,master_bias,instrument,verbose=False):

    science_data = []

    if verbose:
        plt.figure()

    for f in frames_list:

        i = fits.open(f)
        if instrument == 'ACAM':
            data_frame = i[1].data
        if instrument == 'EFOSC':
            data_frame = i[0].data

        print('File = ',f,'; Mean = ',np.mean(data_frame),'; Shape = ',np.shape(data_frame), 'Max count = ',np.max(data_frame[:,20:-20])) # Need to clip extreme edges which can have very high counts
        science_data.append(data_frame-master_bias)

        if verbose:

            #plt.figure()
            plt.imshow(data_frame)#,norm=LogNorm(vmin=i[1].data.min(), vmax=i[1].data.max()))
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.5)
            #plt.close()
            plt.clf()
        i.close()

    science_data = np.array(science_data)

    mean_combine = np.mean(science_data,axis=0)
    return mean_combine

def combine_science_frames_2windows(frames_list,master_bias,verbose=False):

    science_data = [[],[]]

    if verbose:
        plt.figure()

    for f in frames_list:
        i = fits.open(f)

        for level in range(1,len(i)):
            data_frame = i[level].data

            science_data[level-1].append(data_frame-master_bias[level-1])


            if verbose:
                plt.subplot(1,2,level)
                plt.imshow(data_frame)
                plt.colorbar()

            print('File = %s ; Mean (window %d) = %f ; Shape (window %d) = %s ; Max count = %d '%(f,level,np.mean(data_frame),level,np.shape(data_frame),np.max(data_frame[:,20:-20])))
            science_data[level-1].append(data_frame-master_bias[level-1])

            if verbose:
                plt.subplot(1,2,level)
                plt.imshow(data_frame)

        if verbose:
            plt.show(block=False)
            plt.pause(0.5)
            plt.colorbar()
            plt.clf()

        i.close()

    science_data = np.array(science_data)

    mean_combine = [np.mean(science_data[0],axis=0),np.mean(science_data[1],axis=0)]

    return mean_combine

def save_fits(data,filename,clobber=True):
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename,overwrite=clobber)
    return


def test_smoothing_widths(science_data,nwindows):

    """Run a series of median filters with differing box widths to find optimal width."""

    if nwindows > 1:
        nrows,ncols1 = np.shape(science_data[0])
        _,ncols2 = np.shape(science_data[1])
        std1 = []
        std2 = []
    else:
        nrows,ncols = np.shape(science_data)
        std = []

    for i in range(1,nrows//10,2):

        box_width = i

        if nwindows == 1:
            MF = medfilt(science_data[:,ncols//2],box_width)
            residuals = science_data[:,ncols//2]/MF
            # rms.append(np.sqrt(np.mean(residuals**2)))
            std.append(np.std(residuals))

        else:
            MF1 = medfilt(science_data[0][:,ncols1//2],box_width) # just use single column for plotting
            MF2 = medfilt(science_data[1][:,ncols2//2],box_width)

            residuals1 = science_data[0][:,ncols1//2]/MF1
            residuals2 = science_data[1][:,ncols2//2]/MF2

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

def median_smooth(science_data,name,nwindows,box_width,clobber, science_frames):

    """Smooth the average science frame using a running median, and evaluated at each column individually"""

    if nwindows > 1:
        nrows1,ncols1 = np.shape(science_data[0])
        nrows2,ncols2 = np.shape(science_data[1])

        MF1 = medfilt(science_data[0][:,ncols1//2],box_width) # just use single column for plotting
        MF2 = medfilt(science_data[1][:,ncols2//2],box_width)
    else:
        nrows,ncols = np.shape(science_data)
        MF = medfilt(science_data[:,ncols//2],box_width)

    plt.figure(figsize=(10,8))
    plt.subplot(211)

    if nwindows == 1:
        plt.plot(science_data[:,ncols//2],lw=4)
        plt.plot(MF,'r',lw=2)
    else:
        plt.plot(science_data[0][:,ncols1//2],lw=4,color='b',label='window 1')
        plt.plot(MF1,'r',lw=2)
        plt.plot(science_data[1][:,ncols2//2],lw=4,color='k',label='window 2')
        plt.plot(MF2,'r',lw=2)
        plt.legend(loc='upper left')


    plt.xticks(visible=False)
    plt.ylabel('Counts')

    plt.subplot(212)
    if nwindows == 1:
        plt.plot(science_data[:,ncols//2]/MF)
    else:
        plt.plot(science_data[0][:,ncols1//2]/MF1,color='b',label='window 1')
        plt.plot(science_data[1][:,ncols2//2]/MF2,color='k',label='window 2')
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
        MF_reshaped =  np.array([medfilt(science_data[:,i],box_width) for i in range(ncols)]).transpose()

        MF_reshaped[MF_reshaped == 0] = 1 # have to replace 0s which occur near the edges, the bias over subtracts to give negative values which mess up this line

        divided_frame = science_data/MF_reshaped
        normalised_frame = divided_frame/divided_frame.mean()

    else:
        MF1_reshaped = np.array([medfilt(science_data[0][:,i],box_width) for i in range(ncols1)]).transpose()
        divided_frame1 = science_data[0]/MF1_reshaped
        normalised_frame1 = divided_frame1/divided_frame1.mean()

        MF2_reshaped = np.array([medfilt(science_data[1][:,i],box_width) for i in range(ncols2)]).transpose()
        divided_frame2 = science_data[1]/MF2_reshaped
        normalised_frame2 = divided_frame2/divided_frame2.mean()

        normalised_frame = np.array([normalised_frame1,normalised_frame2])


    plt.figure(figsize=(10,10))
    if nwindows == 1:
        plt.imshow(normalised_frame,vmin=0.95,vmax=1.05)
    else:
        plt.subplot(121)
        plt.imshow(normalised_frame[0],vmin=0.95,vmax=1.05)
        plt.subplot(122)
        plt.imshow(normalised_frame[1],vmin=0.95,vmax=1.05)

    plt.suptitle('Median filter, column by column')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.colorbar()
    plt.show(block=False)

    plt.pause(5)
    plt.close()

    hdu = fits.PrimaryHDU(normalised_frame)
    hdu.writeto('master_science_average_'+name+'_median_filter_column_by_column_box_width_%d_FITTED.fits'%box_width,overwrite=clobber)

    for i in files:
        f = fits.open(i)
        
        plt.figure()
        plt.imshow(f[0].data-normalised_frame,vmin=0.95,vmax=1.05)
        plt.show(block=False)

        plt.pause(5)
        plt.close()


#if nwin == 1:
#    f = combine_science_frames(files,master_bias_data,args.instrument,args.verbose)
#else:
#    f = combine_science_frames_2windows(files,master_bias_data,args.verbose)

#save_fits(f,'science_average_'+args.scienceframes+'.fits',True)

#test_smoothing_widths(np.array(f),nwin)

#chosen_box_width = int(input("Enter desired box width for running median (must be ODD INTEGER): "))

#median_smooth(np.array(f),args.scienceframes,nwin,chosen_box_width,args.clobber, files)
#f = fits.open(files[0])[0].data
#for i in files:
        #result = fits.open(i)
        
        
        #plt.figure()
        #plt.imshow(result[0].data/f,vmin=0.95,vmax=1.05)
        #plt.show(block=False)

        #plt.pause(5)
        #plt.close()
def surrounding_median(data, row, column, length):
    #print(column)
    #print(length-1)
    #print(column not in [0,length-1])
    if row==0 and column not in [0,length-1]:
        mean_value = np.median([data[row+1,column], data[row,column+1], data[row,column-1], data[row+1,column+1], data[row+1, column-1]])
        stand_dev = np.std([data[row+1,column], data[row,column+1], data[row,column-1], data[row+1,column+1], data[row+1, column-1]])
    if row==length-1 and column not in [0,length-1]:
        mean_value = np.median([data[row-1,column], data[row,column+1], data[row,column-1], data[row-1,column+1], data[row-1, column-1]])
        stand_dev = np.std([data[row-1,column], data[row,column+1], data[row,column-1], data[row-1,column+1], data[row-1, column-1]])
    if row==0 and column==0:
        #print(np.mean([1,2]))
        mean_value = np.median([data[row+1,column], data[row,column+1], data[row+1,column+1]])
        stand_dev = np.std([data[row+1,column], data[row,column+1], data[row+1,column+1]])
    if row==length-1 and column==length-1:
        mean_value = np.median([data[row-1,column], data[row,column-1], data[row-1,column-1]])
        stand_dev = np.std([data[row-1,column], data[row,column-1], data[row-1,column-1]])
    if row==length-1 and column==0:
        mean_value = np.median([data[row-1,column], data[row,column+1], data[row-1,column+1]])
        stand_dev = np.std([data[row-1,column], data[row,column+1], data[row-1,column+1]])
    if row==0 and column==length-1:
        mean_value = np.median([data[row+1,column], data[row,column-1], data[row+1,column-1]])
        stand_dev = np.std([data[row+1,column], data[row,column-1], data[row+1,column-1]])
    if row not in [0,length-1] and column==0:
        mean_value = np.median([data[row+1,column], data[row,column+1], data[row-1,column], data[row+1,column+1], data[row-1, column+1]])
        stand_dev = np.std([data[row+1,column], data[row,column+1], data[row-1,column], data[row+1,column+1], data[row-1, column+1]])
    if row not in [0,length-1] and column==length-1:
        mean_value = np.median([data[row+1,column], data[row,column-1], data[row-1,column], data[row+1,column-1], data[row-1, column-1]])
        stand_dev = np.std([data[row+1,column], data[row,column-1], data[row-1,column], data[row+1,column-1], data[row-1, column-1]])
    if row not in [0,length-1] and column not in [0,length-1]:
        mean_value = np.median([data[row+1,column], data[row,column+1], data[row-1,column], data[row+1,column+1], data[row-1, column+1], data[row+1, column-1], data[row,column-1], data[row-1, column-1]])
        stand_dev = np.std([data[row+1,column], data[row,column+1], data[row-1,column], data[row+1,column+1], data[row-1, column+1], data[row+1, column-1], data[row,column-1], data[row-1, column-1]])
    #print(mean_value)
    
    return mean_value, stand_dev

def column_mean(data,length):
    mean_column = np.zeros(length)
    data = np.array(data)
    data_reversed = np.swapaxes(data,0,1)
    for i in range(length):
        mean_column[i] = np.mean(data_reversed[i])

    return mean_column
        
n = len(fits.open(files[0])[0].data[0])
#print(fits.open(files[0])[0].data[0])


def column_mean_cosmic():
    no_count = 0
    for k in files:
        data_cosmic = np.zeros((n,n))
        result = fits.open(k)
        f = result[0].data
        no_cosmics=0
        #print(f[1029][1029])
        column_means = column_mean(f, n)
        print(column_means)
        
        for i in range(n):
            for j in range(n):
                #print(i,j)
                data_cosmic[i][j] = f[i][j]/column_means[j]
                
                if data_cosmic[i][j] > 3:
                    print("Cosmic")
                    no_cosmics+=1
        
        
        hdu = fits.PrimaryHDU(data_cosmic)
        hdu.writeto("Figures/Cosmic_analysis_column_means_" + str(no_count) + "_no_" + str(no_cosmics) + '.fits', overwrite=True)
        plt.figure()
        plt.imshow(data_cosmic)
        #plt.show(block=False)
        plt.savefig("Figures/Cosmic_analysis_column_means_" + str(no_count) + "_no_" + str(no_cosmics) + ".png", bbox_inches='tight')
        plt.pause(5)
        plt.close()
        print(no_count)
        no_count += 1
        #exit()

        
#exit()

def divide_by_spectrum(start_no):
    no_count = start_no
    offset = 1000
    for k in files[start_no:]:
        if no_count == 0:
            f_reference = bias_subtraction(fits.open(files[no_count+1])[0].data)+offset
            print(files[no_count+1])
        else:
            f_reference = bias_subtraction(fits.open(files[no_count-1])[0].data)+offset
            print(files[no_count-1])
        data_cosmic = np.zeros((n,n))
        result = fits.open(k)
        f = bias_subtraction(result[0].data)+offset
        hdr = result[0].header
        no_cosmics=0
        print(k)
        data_cosmic = np.true_divide(f,f_reference)
        #print(data_cosmic)
        standard_dev = np.std(data_cosmic)
        print("std = " + str(standard_dev))
        indices_cosmics = np.nonzero(data_cosmic > 1.+16*standard_dev)
        print(len(indices_cosmics[1]))
        print(np.where(f_reference == 0))
        #print(data_cosmic)
        no_cosmics = len(indices_cosmics[1])
        for i in range(len(indices_cosmics[0])):
            #print(indices_cosmics[0][i],indices_cosmics[1][i])
            f[indices_cosmics[0][i],indices_cosmics[1][i]],__ = surrounding_median(f, indices_cosmics[0][i], indices_cosmics[1][i], n)
    #print(column)
            
        
        
        #hdu = fits.PrimaryHDU(data_cosmic, header=hdr)
        #hdu.header = hdr
        #hdu.writeto("Cosmic_divide/Cosmic_analysis_divide_frame_after_" + str(no_count) + "_no_" + str(no_cosmics) + '.fits', overwrite=True)
        hdu = fits.PrimaryHDU(f-offset, header=hdr)
        hdu.writeto("Cosmic_divide/Spectrum_divide_frame_after_" + str(no_count) + "_no_" + str(no_cosmics) + '.fits', overwrite=True)
        #plt.figure()
        #plt.imshow(data_cosmic)
        #plt.show(block=False)
        #plt.savefig("Figures/Cosmic_divide/Cosmic_analysis_divide_frame_after_" + str(no_count) + "_no_" + str(no_cosmics) + ".png", bbox_inches='tight')
        #plt.pause(5)
        #plt.close()
        print(no_count)
        no_count += 1
        #exit()
        
    

def median_all_pixels(start_no):
    no_count = start_no
    for k in files[start_no:]:
        data_cosmic = np.zeros((n,n))
        result = fits.open(k)
        f = result[0].data
        no_cosmics=0
        #print(f[1029][1029])
        
        for i in range(n):
            for j in range(n):
                #print(i,j)
                median, std = surrounding_median(f, i, j, n)
                data_cosmic[i][j] = f[i][j]-median
                
                if data_cosmic[i][j] > 11*std:
                    print("Cosmic at row %i"%i + ", column %i"%j)
                    no_cosmics+=1
                    f[i][j] = median
                    
        
        
        hdu = fits.PrimaryHDU(data_cosmic)
        hdu.writeto("Figures/Cosmic_analysis_median_difference_" + str(no_count) + "_no_" + str(no_cosmics) + '.fits', overwrite=True)
        hdu = fits.PrimaryHDU(f)
        hdu.writeto("Figures/Spectrum_" + str(no_count) + "_no_" + str(no_cosmics) + '.fits', overwrite=True)
        plt.figure()
        plt.imshow(data_cosmic)
        #plt.show(block=False)
        plt.savefig("Figures/Cosmic_analysis_median_std_" + str(no_count) + "_no_" + str(no_cosmics) + ".png", bbox_inches='tight')
        plt.pause(5)
        plt.close()
        print(no_count)
        no_count += 1

#median_all_pixels(288)
divide_by_spectrum(0)
exit()        
## Plot master average frame
plt.figure()
if nwin == 1:
    plt.imshow(f)
else:
    plt.subplot(121)
    plt.imshow(f[0])
    plt.subplot(122)
    plt.imshow(f[1])
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.suptitle('Average science frame before response fitting')
plt.colorbar()
plt.savefig("average_science_frame.png")
plt.show()
