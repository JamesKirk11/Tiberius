#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk 

from astropy.io import fits
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Extract WHT rotator angle (and sky background from FITS headers) for a series of FITS files.')
parser.add_argument("science_list",help="List of science FITS files from which to extract the angles.")
args = parser.parse_args()

science_list = np.loadtxt(args.science_list,str)

diagnostic_tab = open('diagnostic_tab.txt','w')

parameters = ['ACAMFOFF','CAGAFOCU','CAGARADI','CAGATHET','CAGTVFOC','XAPOFF','YAPOFF','AZSTART','AZEND','ZDSTART','ZDEND','PLATESCA','TELFOCUS','ACTELFOC','ROTSKYPA','MNTPASTA',\
               'MNTPAEND','PARANSTA','PARANEND','DAZSTART','DAZEND','AIRMASS','AMSTART','AMEND','TEMPTUBE','FOCUSTMP','FOCUSALT','FOCUSFLT','SKYBRZEN','SKYBRTEL','CCDTEMP']

npars = len(parameters)

diagnostic_tab.write('# \t')

for p in parameters:
    diagnostic_tab.write('%s \t'%p)

diagnostic_tab.write('\n')


for f in science_list:

    hdu = fits.open(f)

    header = hdu[0].header

    for i,p in enumerate(parameters):
        diagnostic_tab.write("%f \t"%(header[p]))

    if i == npars-1:
        diagnostic_tab.write('\n')

diagnostic_tab.close()

values = np.loadtxt('diagnostic_tab.txt')

nfigures = 4
nsubplots = 8

count = 0

for i in range(nfigures):

    plt.figure(figsize=(10,8))

    for j in range(nsubplots):
        plt.subplot(nsubplots,1,j+1)
        plt.plot(values[:,count])
        plt.ylabel(parameters[count])
        plt.ticklabel_format(useOffset=False)
        plt.xticks(visible=False)

        count += 1

        if count == npars:
            break

    plt.xticks(visible=True)
    plt.xlabel('Frame')
    plt.savefig('diagnostic_plots_%d.pdf'%(i+1),bbox_inches='tight')

    plt.show()
