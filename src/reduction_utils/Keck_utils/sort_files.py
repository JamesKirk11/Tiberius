import numpy as np
from astropy.io import fits
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='extract file lists based on image types from NIRSPEC fits files')
parser.add_argument("-pwd",help="Use to overwrite pwd with own, defined pwd which is used as the list prefix",type=str)
args = parser.parse_args()


if args.pwd is not None:
    file_path = ''
    all_files = sorted(glob.glob("%s*.fits"%args.pwd))
else:
    file_path = os.getcwd()
    all_files = sorted(glob.glob("*.fits"))
    

def split_list(file_names,file_path_name):
    
    for file_number,i in enumerate(file_names):
        f = fits.open(i)
        hdr = f[0].header

        try:

            obj = hdr["OBSTYPE"].strip()
            if obj == 'object':
                list_name = hdr["TARGNAME"].strip()
            else:
                list_name = obj

            # Slit
            slit = hdr['SLITNAME']
            slit.replace('.','p')
            list_name += '_%s'%slit

            # filter
            filter = hdr["FILTER"].strip()
            list_name += "_%s"%filter


        except:
            continue

        print(i, obj,slit)

        try:
            file_list = open(list_name+'_list','a')
        except:
            file_list = open(list_name+'_list','w')

        file_list.write(file_path_name+'/'+i+' \n')
        file_list.close()

        f.close()
    return

split_list(all_files,file_path)
