.. _jwst:

Extracting JWST data
====================

Firstly, you'll need to download your data! Navigate to `MAST <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_ and use the search boxes to find your target / program. Add your desired observations to your download basket and then make sure to download the uncal.fits files, by marking the check box next to "UNCAL" in the "Group" filter within the download basket. These are what we will use for the stage 0 reduction. You don't need to download any more files than the uncal.fits files.

Alternatively, you can download the rateints.fits files if you don't want to perform stage 1 extraction yourself (not recommended) and jump straight to "Stage 2" below.

Stage 1
-------

After downloading and unpacking the JWST data, navigate to your downloaded directory. You will see that the JWST files are divided into segments. At this point, I like to take a quick look at the data and to make an initial bad pixel mask, which we will use later.

For the quick look and bad pixel mask creation, look at the example jupyter notebook under ``Tiberius/src/reduction_utils/JWST_utils/reduction_notebooks/0_quick_look_data.ipynb``.

Now you will want to run the relevant stage 1 executable found under ``Tiberius/src/reduction_utils/JWST_utils/stage1_*``. These are just executable text files that string together the relevant commands from the ``jwst`` pipeline, following the procedure outlined `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html#calwebb-detector1>`_.

***Note: the PRISM stage 1 includes a 1/f correction and saturation flagging override as default, following the procedures outlined in `Rustamkulov et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023Natur.614..659R/abstract>`_. If you don't wish to perform these corrections, you'll need to comment out these lines within the relevant "stage1_*" file.***

In ``stage1_NIRCam``, I demonstrate how to override the reference files that ``jwst`` will try to use by default, as I've found that it won't always use the most recent reference files. You can download the latest JWST reference files `here <https://jwst-crds.stsci.edu/>`_.

If you do download new reference files, I recommend putting them in:
``$HOME/crds_cache/jwst_pub/references/jwst/[instrument-name-in-lower-case]/``







Stage 1
-------

1. Download the data

Go to https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html

2. Install the STScI jwst pipeline within a new conda environment called "JWST"

Follow the jwst installation guidelines at https://jwst-pipeline.readthedocs.io/en/stable/

3. Run custom_strun on the uncal.fits JWST data files after first editing them so that they point to the correct path location on your file system

$ . /path/to/custom_strun_no_sat_override [file-name-before-the-_uncal.fits-suffix]

Notes:
1) if there is saturation within the images then you will need to look into the custom_strun script that includes prescriptions for overriding the saturation step.
2) if you are working with MIRI data then you will need to run custom_strun_MIRI


4. Run expand_JWST_fits.py to convert jwst .fits files into Tiberius-readable files
