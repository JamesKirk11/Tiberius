import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse
import os
import copy

parser = argparse.ArgumentParser(description='create new science .fits files where the A-B nod pair subtraction has been performed. ')
parser.add_argument("science_list",help="the list of science images",type=str)
parser.add_argument('-v','--verbose',help="""Display the image of each frame""",action='store_true')
args = parser.parse_args()

science_list = np.loadtxt(args.science_list,dtype=str)

nfiles = len(science_list)

try:
    os.mkdir("AB_subtracted")
except:
    pass

for i in range(0,nfiles,2):
    
    nod1 = science_list[i]
    nod2 = science_list[i+1]
    
    frame1 = fits.open(nod1,memmap=False)
    frame1_sub = copy.deepcopy(frame1)
    
    frame2 = fits.open(nod2,memmap=False)
    frame2_sub = copy.deepcopy(frame2)
    
    # note we don't need to subtract a bias since the bias is subtracted in the AB differencing
    frame1_sub[0].data = frame1[0].data - frame2[0].data
    frame2_sub[0].data = frame2[0].data - frame1[0].data
    
    frame1_sub.writeto("AB_subtracted/%s"%nod1[-21:],overwrite=True) # I know that the last 21 characters are what constitutes the nspec file name
    frame2_sub.writeto("AB_subtracted/%s"%nod2[-21:],overwrite=True)
    
    if args.verbose:
        plt.figure()

        plt.subplot(121)
        vmin,vmax = np.percentile(frame1_sub[0].data,[10,90])
        plt.imshow(frame1_sub[0].data,vmin=vmin,vmax=vmax,aspect="auto")
        plt.title("%s - %s"%(nod1[-21:],nod2[-21:]))
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.colorbar()
        
        plt.subplot(122)
        vmin,vmax = np.percentile(frame1_sub[0].data,[10,90])
        plt.imshow(frame2_sub[0].data,vmin=vmin,vmax=vmax,aspect="auto")
        plt.title("%s - %s"%(nod2[-21:],nod1[-21:]))
        plt.colorbar()
        plt.show()
            
    frame1.close()
    frame2.close()
    
