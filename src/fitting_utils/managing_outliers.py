#### Author of this code: James Kirk

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as US
from scipy.signal import medfilt


def clipping_outliers_with_median_clip(flux, flux_error, time, sigma_clip, show_plots, save_plots, output_foldername):

    print('Clipping outliers...')

    # Running median
    box_width = int(len(flux)/10)
    if box_width % 2 == 0:
        box_width += 1

    MF = medfilt(flux,box_width)

    # Use polynomial to remove edge effects of running median
    x = np.arange(len(flux))

    poly_left = np.poly1d(np.polyfit(x[:box_width*2],MF[:box_width*2],1))
    poly_right = np.poly1d(np.polyfit(x[-box_width*2:],MF[-box_width*2:],1))

    MF[:box_width] = poly_left(x[:box_width])
    MF[-box_width:] = poly_right(x[-box_width:])

    filtered_residuals = flux - MF
    standard_residuals = np.std(filtered_residuals)

    keep_idx = ((filtered_residuals <= sigma_clip*standard_residuals) & (filtered_residuals >= -sigma_clip*standard_residuals))

    clipped_flux = flux[keep_idx]
    clipped_flux_error = flux_error[keep_idx]
    clipped_time = time[keep_idx]

    print("\n %d data points (%.2f%%) clipped from fit"%(len(time)-len(clipped_time),100*(len(time)-len(clipped_time))/len(time)))


    if show_plots:
        plt.figure()
        plt.subplot(211)
        plt.errorbar(time[~keep_idx],flux[~keep_idx],yerr=flux_error[~keep_idx],fmt='o',ecolor='r',color='r',label='Clipped outliers')
        plt.errorbar(clipped_time,clipped_flux,yerr=clipped_flux_error,fmt='o',ecolor='k',color='k',alpha=0.5)
        plt.plot(time,MF,'r')
        plt.ylabel('Normalised flux')
        plt.title('Outlier clipping')
        plt.legend(loc='upper left')

        # residuals
        plt.subplot(212)
        plt.errorbar(time[~keep_idx],flux[~keep_idx]-MF[~keep_idx],yerr=flux_error[~keep_idx],fmt='o',ecolor='r',color='r',label='Clipped outliers')
        plt.errorbar(clipped_time,clipped_flux-MF[keep_idx],yerr=clipped_flux_error,fmt='o',ecolor='k',color='k',alpha=0.5)
        plt.axhline(sigma_clip*standard_residuals,ls='--',color='gray')
        plt.axhline(sigma_clip*-standard_residuals,ls='--',color='gray')
        plt.ylim(-10*standard_residuals,10*standard_residuals)

        plt.xlabel('Time (MJD)')
        plt.ylabel('Residuals')

        plt.show(block=False) # only show for 5 seconds. This is necessary when running fits to multiple bins so that the code doesn't have to wait for user to manually close windows before continuing.
        plt.pause(5)
        plt.close()

    if save_plots:
        plt.figure()
        plt.subplot(211)
        plt.errorbar(time[~keep_idx],flux[~keep_idx],yerr=flux_error[~keep_idx],fmt='o',ecolor='r',color='r',label='Clipped outliers')
        plt.errorbar(clipped_time,clipped_flux,yerr=clipped_flux_error,fmt='o',ecolor='k',color='k',alpha=0.5)
        plt.plot(time,MF,'r')
        plt.ylabel('Normalised flux')
        plt.title('Outlier clipping')
        plt.legend(loc='upper left')

        # residuals
        plt.subplot(212)
        plt.errorbar(time[~keep_idx],flux[~keep_idx]-MF[~keep_idx],yerr=flux_error[~keep_idx],fmt='o',ecolor='r',color='r',label='Clipped outliers')
        plt.errorbar(clipped_time,clipped_flux-MF[keep_idx],yerr=clipped_flux_error,fmt='o',ecolor='k',color='k',alpha=0.5)
        plt.axhline(sigma_clip*standard_residuals,ls='--',color='gray')
        plt.axhline(sigma_clip*-standard_residuals,ls='--',color='gray')
        plt.ylim(-10*standard_residuals,10*standard_residuals)

        plt.xlabel('Time (MJD)')
        plt.ylabel('Residuals')

        plt.savefig(output_foldername + '/Figures/Outlier_clipping.png', bbox_inches='tight') 
    
    return clipped_flux, clipped_flux_error, clipped_time, keep_idx

