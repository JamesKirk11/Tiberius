.. _jwst:

Extracting JWST data
====================

Firstly, you'll need to download your data! Navigate to `MAST <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_ and use the search boxes to find your target / program. Add your desired observations to your download basket and then make sure to download the ``uncal.fits`` files, by marking the check box next to "UNCAL" in the "Group" filter within the download basket. These are what we will use for the Stage 1 reduction. You don't need to download any more files than the ``uncal.fits`` files.

Alternatively, you can download the rateints.fits files if you don't want to perform stage 1 extraction yourself (not recommended) and jump straight to "Stage 2" below.

1. Stage 1
----------

.. _stage1:

1.1 Running the ``jwst`` pipeline on uncal.fits files
-----------------------------------------------------

After downloading and unpacking the JWST data, navigate to your downloaded directory. You will see that the JWST files are divided into separate segment subdirectories ``jw......-seg001``, ``jw.........-seg002``,... . I tend to leave these subdirectories as they are and run the below stage 1 steps separately within each segment sub-directory.

At this point, I like to take a quick look at the data and to make an initial bad pixel mask, which we will use later.

For the quick look and bad pixel mask creation, look at the example jupyter notebook under ``Tiberius/src/reduction_utils/JWST_utils/reduction_notebooks/0_quick_look_data.ipynb``.

Now you will want to run the relevant stage 1 executable found under ``Tiberius/src/reduction_utils/JWST_utils/stage1_*``. These are just executable text files that string together the relevant commands from the ``jwst`` pipeline, following the procedure outlined `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html#calwebb-detector1>`_.

**Note:** the PRISM stage 1 includes a 1/f correction and saturation flagging override as default, following the procedures outlined in `Rustamkulov et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023Natur.614..659R/abstract>`_. If you don't wish to perform these corrections, you'll need to comment out these lines within the relevant "stage1_*" file.

In ``stage1_NIRCam``, I demonstrate how to override the reference files that ``jwst`` will try to use by default, as I've found that it won't always use the most recent reference files. You can download the latest JWST reference files `here <https://jwst-crds.stsci.edu/>`_.

If you do download new reference files, I recommend putting them in:
``$HOME/crds_cache/jwst_pub/references/jwst/[instrument-name-in-lower-case]/``

Once you are happy with your ``stage1_*`` file, you can run the stage 1 extraction by doing the following (assuming you have installed ``jwst`` into a new conda environment called ``JWST``). Note if you copy/make a new ``stage1_*`` file, you'll need to make it executable by doing:

.. code-block:: bash

    chmod +x stage1_*``

Then you can proceed as follows:

.. code-block:: bash

    conda activate JWST
    export CRDS_PATH=$HOME/crds_cache/jwst_pub
    export CRDS_SERVER_URL=https://jwst-crds-pub.stsci.edu
    . /path/to/stage1_* [file-name-of-jwst-fits-file-truncated-before-_uncal.fits]

e.g.,

.. code-block:: bash

   . /path/to/stage1_PRISM jw01366004001_04101_00001-seg001

This will produce a series of fits files, with the main one of interest being the ``gainscalestep.fits`` files which is what we will work with in Stage 2. By default, Tiberius' ``stage1_*`` executables clean the subdirectories of other intermediate fits files, otherwise you can quickly run out of storage! You can prevent this behaviour by commenting out the relevant lines at the bottom of the ``stage1_`` text files.

1.2 Cleaning the cosmic rays / telegraph pixels
-----------------------------------------------

With the ``gainscalestep.fits`` in hand, you're ready to proceed with cleaning the fits files of cosmic rays.

Within the parent directory of your segment subdirectories, first make a list of the ``gainscalestep.fits`` files:

.. code-block:: bash

    ls **/*gainscalestep.fits > cosmic_file_list

Then run ``reduction_utils/locate_cosmics.py`` which will locate the cosmic rays and telegraph pixels by calculating medians for every pixel in the time-series and comparing each pixel to its respective median. Flagged outliers will then be replaced by the median for that pixel in the time-series.

I have set the default arguments to sensible values but you will want to experiment on a case-by-case basis to see whether these need altering. In most cases with Tiberius, adding ``-h`` as a command line argument will print help for that particular script along with argument definitions.

After generating ``cosmic_file_list`` do:

.. code-block:: bash

    python /path/to/Tiberius/src/reduction_utils/locate_cosmics.py cosmic_file_list -jwst -h

Once you have looked at the parameter definitions, run the above again without the ``-h`` parameter.

This will calculate all pixel medians and then plot all integrations that have a total number of flagged pixels greater than the threshold set by ``-frame_clip`` (default = 3, which might plot a lot of frames!).

For every frame that exceeds this threshold, it will ask you in the terminal:

.. code-block:: bash

  Reset mask for integration N? [y/n]

This gives you an opportunity to overwrite all pixel flags for a whole integration if you suspect the outlier detection was too aggressive. If you have the settings right, this should just plot integrations with massive cosmics, for which you can reply ``n`` to the command line question.

Once you have vetted all these flagged frames, it will ask you one last question (try not to be too hasty with your ``n`` key!!).

.. code-block:: bash

  Replace cosmic values with median and save to new fits? [y/n]:

Providing you are happy with everything up to this point, you can hit ``y`` which will replace all flagged pixels in the time-series with the medians and save the cleaned integrations to a new directory called ``cosmic_cleaned_fits/``. If you are not happy, hit ``n`` and play around with the command line arguments for ``locate_cosmics.py``.
