.. _jwst:

Extracting JWST data
====================

Firstly, you'll need to download your data! Navigate to `MAST <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_ and use the search boxes to find your target / program. Add your desired observations to your download basket and then make sure to download the uncal.fits files, by marking the check box next to "UNCAL" in the "Group" filter within the download basket. These are what we will use for the stage 0 reduction. You don't need to download any more files than the uncal.fits files.

Alternatively, you can download the rateints.fits files if you don't want to perform stage 1 extraction yourself (not recommended) and jump straight to "Stage 2" below.

1. Stage 1
----------

1.1 Running the ``jwst pipeline`` on uncal.fits files
-----------------------------------------------------

After downloading and unpacking the JWST data, navigate to your downloaded directory. You will see that the JWST files are divided into separate segment subdirectories ``jw......-seg001``, ``jw.........-seg002``,... . I tend to leave these subdirectories as they are and run the below stage 1 steps separately within each segment sub-directory.

At this point, I like to take a quick look at the data and to make an initial bad pixel mask, which we will use later.

For the quick look and bad pixel mask creation, look at the example jupyter notebook under ``Tiberius/src/reduction_utils/JWST_utils/reduction_notebooks/0_quick_look_data.ipynb``.

Now you will want to run the relevant stage 1 executable found under ``Tiberius/src/reduction_utils/JWST_utils/stage1_*``. These are just executable text files that string together the relevant commands from the ``jwst`` pipeline, following the procedure outlined `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html#calwebb-detector1>`_.

**Note:** the PRISM stage 1 includes a 1/f correction and saturation flagging override as default, following the procedures outlined in `Rustamkulov et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023Natur.614..659R/abstract>`_. If you don't wish to perform these corrections, you'll need to comment out these lines within the relevant "stage1_*" file.

In ``stage1_NIRCam``, I demonstrate how to override the reference files that ``jwst`` will try to use by default, as I've found that it won't always use the most recent reference files. You can download the latest JWST reference files `here <https://jwst-crds.stsci.edu/>`_.

If you do download new reference files, I recommend putting them in:
``$HOME/crds_cache/jwst_pub/references/jwst/[instrument-name-in-lower-case]/``

Once you are happy with your ``stage1_*`` file, you can run the stage 1 extraction by doing the following (assuming you have installed ``jwst`` into a new conda environment called ``JWST``). Note if you copy/make a new ``stage1_*`` file, you'll need to make it exectuable by doing ``$ chmod +x stage1_*``.

.. code-block:: bash

    conda activate JWST
    export CRDS_PATH=$HOME/crds_cache/jwst_pub
    export CRDS_SERVER_URL=https://jwst-crds-pub.stsci.edu
    . /path/to/stage1_* [file-name-of-jwst-fits-file-truncated-before-_uncal.fits]

e.g.,

.. code-block:: bash

   . /path/to/stage1_PRISM jw01366004001_04101_00001-seg001

This will produce a series of fits files, with the main one of interest being the ``gainscalestep.fits`` files which is what we will work with in Stage 2. By default, Tiberius' stage1_ executables clean the subdirectories of other intermediate fits files, otherwise you can quickly run out of storage! You can prevent this behaviour by commenting out the relevant lines at the bottom of the ``stage1_`` text files.

1.2 Cleaning the cosmic rays / telegraph pixels
-----------------------------------------------
