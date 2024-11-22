### Use this file to run all stage1 executables for a particular instrument

import os
import glob

### Assuming we are in the parent directory where all uncal.fits files are in sub-directories
all_uncal_files = sorted(glob.glob("**/*uncal*"))

### Double checking whether we've already run any stage1 extractions, so we don't need to run them again
completed_files = len(sorted(glob.glob("**/*gain*")))

cwd = os.getcwd()

### Now loop over the uncal.fits files that have not yet been processed through stage1
for i in all_uncal_files[completed_files:]:

    ### work out correct file path on the fly
    direc = cwd + "/" + i.split("/")[0] + "/"
    file = i.split("/")[1]
    root = file.split("_uncal.fits")[0]

    ### change into the correct subdirectory
    os.chdir(direc)

    ### run correct stage1 file -- note this stage1 file should have been copied into the parent directory and you need to change the below line to point to the correct stage1 executable
    os.system(". ../[stage1_file] %s"%root) # replace [stage1_file] with actual file name, e.g. stage1_g395h_nrs1 or stage1_MIRI
