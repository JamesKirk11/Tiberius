#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

from astropy.io import fits
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Load all fits files within a directory and produced lists of file types')
parser.add_argument("-c","--clobber",help="Overwrite previously saved lists if they exit. Particularly useful when doing reductions in real-time",action='store_true')
args = parser.parse_args()

all_files = sorted(glob.glob('*.fits.gz'))
pwd = os.getcwd()

if args.clobber:
    print("Clobbering",glob.glob("*_list*"))
    preexisting = [open(i,'w') for i in glob.glob("*_list*")]
    [i.close() for i in preexisting]
else:
    pass

def split_list(file_names,pwd):

    for i,f in enumerate(file_names):

        # print(f)

        fits_file = fits.open(f)
        hdr = fits_file[0].header
        fits_file.close()

        INSTRUMENT = hdr["INSTRUME"] # MIKE-Blue or MIKE-Red
        OBJECT = hdr["OBJECT"]

        list_name = pwd + "/" + INSTRUMENT + "_" + OBJECT.strip() + "_list.txt"

        print(f,list_name)

        if glob.glob(list_name): # file exists
            table = open(list_name,"a")

        else:
            table = open(list_name,"w")

        table.write("%s/%s \n"%(pwd,f))

        table.close()

split_list(all_files,pwd)
