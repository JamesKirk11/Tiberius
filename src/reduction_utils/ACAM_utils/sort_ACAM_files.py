#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk 

from astropy.io import fits
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Load all fits files within a directory and produced lists of file types')
parser.add_argument("-c","--clobber",help="Overwrite previously saved lists if they exit. Particularly useful when doing reductions in real-time",action='store_true')
parser.add_argument('-w1','--window_1',help="""define the first window, as it appears in the header""")
parser.add_argument('-w2','--window_2',help="""define the second window, as it appears in the header. Leave blank if 1 window used.""")
parser.add_argument('-p1','--position_1',help="""define the position of the first window""",type=int)
parser.add_argument('-p2','--position_2',help="""define the position of the second window""",type=int)
args = parser.parse_args()

all_files = sorted(glob.glob('*.fit'))
pwd = os.getcwd()

if args.clobber:
     preexisting = [open(i,'w') for i in glob.glob("*_list")]
     [i.close() for i in preexisting]
else:
    pass
    
def split_list(file_names,pwd,window1=None,position1=None,window2=None,position2=None):
    for file_number,i in enumerate(file_names):
        f = fits.open(i)
        hdr = f[0].header
        
        try:
            
            # Object name
            obj = hdr['Object']
            obj = obj.replace(" ", "_") # replace white space
            obj = obj.replace("/", "_") # replace forward slashes from FOCRUN

            list_name = obj
            
            # Slit
            slit = hdr['ACAMSLI']
            if slit != 'CLR':
                if slit == 'SLIT':
                    slit = '40'
                else:
                    slit = slit.replace('.','p')
                
                slit = slit+'arcsec'
                list_name += '_'+slit

            
            # Grism
            grism = hdr['ACAMDISP']
            #if grism != 'NONE':
                #list_name += '_'+grism
            
            # Filter 1
            wheel1 = hdr['ACAMWH1']
            if wheel1 != 'CLEAR':
                list_name += '_'+wheel1
            
            wheel2 = hdr['ACAMWH2']
            if wheel2 != 'CLEAR':
                list_name += '_'+wheel2
            
            # Blocking filters etc.
            mask = hdr['ACAMMASK']
            #if mask != 'CLR':
                #list_name += '_'+mask
            
            # Readout speed
            readout_speed = hdr['CCDSPEED']
            list_name += '_'+readout_speed
        
        except:
            continue
            
        if window1 is not None:
            if hdr['WINSEC%d'%position1] != window1 +', enabled':
                list_name += '_wrong_window1'
        
        if window2 is not None:
            if hdr['WINSEC%d'%position2] != window2 +', enabled':
                list_name += '_wrong_window2'
        
        print(i, obj,slit,grism,mask,wheel1,wheel2)
        
        try:
            file_list = open(list_name+'_list','a')
        except:
            file_list = open(list_name+'_list','w')
        
        file_list.write(pwd+'/'+i+' \n')
        file_list.close()
        
        f.close()

split_list(all_files,pwd,args.window_1,args.position_1,args.window_2,args.position_2)
