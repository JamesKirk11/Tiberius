.. _jwst:

Extracting JWST data
====================

Firstly, you'll need to download your data! Navigate to `MAST <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_ and use the search boxes to find your target / program. Add your desired observations to your download basket and then make sure to download the ``uncal.fits`` files, by marking the check box next to "UNCAL" in the "Group" filter within the download basket. These are what we will use for the Stage 1 reduction. You don't need to download any more files than the ``uncal.fits`` files.

Alternatively, you can download the rateints.fits files if you don't want to perform stage 1 extraction yourself (not recommended) and jump straight to "Stage 2" below.

1. Stage 1
----------

.. _stage1:

1.1 Running the ``jwst`` pipeline on ``uncal.fits`` files
-----------------------------------------------------

After downloading and unpacking the JWST data, navigate to your downloaded directory. You will see that the JWST files are divided into separate segment subdirectories ``jw......-seg001``, ``jw.........-seg002``,... . I tend to leave these subdirectories as they are and run the below stage 1 steps separately within each segment sub-directory.

At this point, I like to take a quick look at the data and to make an initial bad pixel mask, which we will use later.

For the quick look and bad pixel mask creation, look at the example jupyter notebook under ``Tiberius/src/reduction_utils/JWST_utils/reduction_notebooks/0_quick_look_data.ipynb``.

Now you will want to run the relevant stage 1 executable found under ``Tiberius/src/reduction_utils/JWST_utils/stage1_*``. These are just executable text files that string together the relevant commands from the ``jwst`` pipeline, following the procedure outlined `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html#calwebb-detector1>`_.

.. note::

  The stage 1 files include a 1/f correction by default and saturation flagging override for PRISM data as default, following the procedures outlined in `Rustamkulov et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023Natur.614..659R/abstract>`_. If you don't wish to perform these corrections, you'll need to comment out these lines within the relevant "stage1_*" file. If you do want to perform a 1/f correction, see the following sub-section, otherwise skip ahead.

1.1.1 Performing a 1/f correction
---------------------------------

In JWST observations, there is column-dependent (for NIRCam it's row-dependent) noise which can increase the noise in your extracted light curves. As shown in `Rustamkulov et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023Natur.614..659R/abstract>`_, this noise is best-removed at the group stage. In order to do this accurately, it is necessary to determine the locations of the trace so that it can be masked from the background estimation.

To do this, first make a "master" ``uncal.fits`` file (a median frame of all ``uncal.fits`` files -- see ``Tiberius/src/reduction_utils/JWST_utils/0_quick_look_data.ipynb`` for an example. You'd then want to run ``Tiberius/src/reduction_utils/spectral_extraction.py`` on this master ``uncal.fits`` file to determine the location of the trace (see section 1.3 below to see how to do this).

Then in the relevant line in your ``stage1_`` file, you would have to change the following:

.. code-block:: bash

    python $HOME/python/Tiberius/src/reduction_utils/JWST_utils/1overf_subtraction.py $1_darkcurrentstep.fits --trace_location /path-to-master_uncal.fits/pickled_objects/x_positions_1.pickle --extraction_input /path-to-master_uncal.fits/extraction_input.txt

Now you are ready to run stage 1, which is described in more detail below.

1.1.2 Running ``stage1`` files
------------------------------

In ``stage1_NIRCam``, I demonstrate how to override the reference files that ``jwst`` will try to use by default, as I've found that it won't always use the most recent reference files. You can download the latest JWST reference files `here <https://jwst-crds.stsci.edu/>`_.

If you do download new reference files, I recommend putting them in:
``$HOME/crds_cache/jwst_pub/references/jwst/[instrument-name-in-lower-case]/``

Once you are happy with your ``stage1_*`` file, you can run the stage 1 extraction by doing the following (assuming you have installed ``jwst`` into a new conda environment called ``JWST``). Note if you copy/make a new ``stage1_*`` file, you'll need to make it executable by doing:

.. code-block:: bash

    chmod +x stage1_*

Then you can proceed as follows:

.. code-block:: bash

    conda activate JWST
    export CRDS_PATH=$HOME/crds_cache/jwst_pub
    export CRDS_SERVER_URL=https://jwst-crds-pub.stsci.edu
    . /path/to/stage1_* [file-name-of-jwst-fits-file-truncated-before-_uncal.fits]

e.g.,

.. code-block:: bash

   . /path/to/stage1_PRISM jw01366004001_04101_00001-seg001

This will produce a series of fits files, with the main one of interest being the ``gainscalestep.fits`` files which is what we will work with in Stage 2. By default, Tiberius' ``stage1_*`` executables clean the subdirectories of other intermediate fits files, otherwise you can quickly run out of storage! You can prevent this behaviour by commenting out the relevant lines (``rm jw......fits``) in the ``stage1_`` text files.

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

1.3 Extracting stellar spectra
------------------------------

Now we have our cosmic-cleaned integration level fits files, we are ready to run aperture photometry on these to extract our stellar spectra.

I recommend you make a new directory (``reduction01, reduction02,...``) for each test reduction you perform (e.g., different aperture and background widths).

In each new reduction directory, you will need to make a new ``extraction_input.txt`` file (which can be copied from a previous reduction or from ``/path/to/Tiberius/src/reduction_utils/extraction_input.txt``). You will also need to make a text file with a list of filenames defining the fits files you will be running the extraction over. Assuming you're working with the cosmic-cleaned fits files, this can be made like so:

.. code-block:: bash

  ls /path/to/cosmic_cleaned_fits/*.fits > science_list

You then need to define the path to this ``science_list`` in your ``extraction_input.txt`` file. I don't explain the different parameters in ``extraction_input.txt`` at this point as they are each explained within the example ``extraction_input.txt`` bundled in the ``Tiberius`` download.

One thing I do recommend, however, is that every time you run a reduction for the first time, or with a new set of extraction parameters, that you set ``verbose = -2`` in ``extraction_input.txt``. This will plot a number of helpful plots for every integration and allow you to check whether the parameters you've selected are sensible. If they are, then you can quit the extraction and set ``verbose = -1`` (for no plots) or ``verbose = 0`` (which will only show plots for a particular integration if something has gone wrong with that integration).

.. note::

  ``Tiberius`` needs to have the dispersion/spectral direction along the vertical axis. That means for NIRSpec, NIRCam and NIRISS data you need to set ``rotate_frame = 1`` in ``extraction_input.txt``.

To actually run the extraction, you will need to run the following from within your reduction directory where you have put ``extraction_input.txt`` and ``science_list``:

.. code-block:: bash

  python /path/to/Tiberius/src/reduction_utils/spectral_extraction.py

This will loop through all integrations, performing aperture photometry, and print out its progress.

After running ``spectral_extraction.py``, you will see that two new sub-directories have been made:

* ``pickled_objects/`` which contains the extracted stellar flux (``star1_flux.pickle``), flux uncertainty (``star1_error.pickle``), time stamps (``time.pickle`` == ``int_mid_BJD_TDB`` from the FITS headers), measured FWHM (``fwhm_1.pickle``), x position (``x_positions_1.pickle``) and measured background (``background_avg_star1.pickle``) as pickled numpy arrays.
* ``initial_WL_fit/`` which contains the extracted white light light curve (``initial_WL_flux.pickle``), white light light curve error (``initial_WL_err.pickle``) and white light curve time arrays (``initial_WL_time.pickle``). These can be fitted with ``Tiberius``'s light curve fitting tools (read on to see how) to check the quality of your reduction.

1.3.1 A note on background subtraction
--------------------------------------

During the ``spectral_extraction.py`` step, you have the option to perform a background subtraction at the integration level, using the background parameters in ``extraction_input.txt``. ``Tiberius`` can fit any order of polynomial (or use a median) across two regions either side of the trace, as defined in ``extraction_input.txt``. I have found an additional background subtraction step to be advantageous even if you performed a 1/f correction at the group stage. This is because the background may have structure that is not well-described by the median that was used in the 1/f step.

1.3.2 A note on oversampling
----------------------------

``Tiberius`` allows you to oversample an integration's flux along the spatial dimension. This is done via a flux-conserving linear interpolation onto an axis with N times the original number of pixels. The motivation for this step is to be able to use sub-pixel apertures, which is particularly beneficial for curved and/or undersampled PSFs (e.g., PRISM). In tests on ERS PRISM data, setting ``oversampling_factor = 10`` in ``extraction_input.txt`` led to an improvement in white light scatter of 14%.

1.4 Post-processing the spectra
-------------------------------

After you've extracted the spectra using ``spectral_extraction.py``, you're ready to perform the wavelength calibration, correct for any shifts in the spectra and create your wavelength bins and light curves. These steps are done using a serious of Jupyter notebooks, with examples included in ``Tiberius/src/reduction_utils/JWST_utils/reduction_notebooks/``.

I tend to copy the example ``reduction_notebooks`` directory into each of my ``reductionNN/`` directories. I go through each of these notebooks below.

* ``0_quick_look_data.ipynb``:  I use this notebook to look at the uncal.fits files and make bad pixel masks
* ``1_cosmic_removal.ipynb``: this notebook describes how you can check for and remove residual cosmic rays and bad pixels from your extracted spectra. Typically, if you've run ``locate_cosmics.py`` this step is not necessary.
* ``2_spectra_resampling.ipynb``: this notebook cross-correlates each spectrum in the time-series with an reference spectrum from the time-series to determine how the spectra shift in the dispersion axis. You can then use these shifts to resample the spectra onto a common (sub)pixel grid. This is not strictly necessary given the shifts are typically << 1 pixel.
* ``3_wavelength_calibration.ipynb``: this notebook shows you how to get the wavelength solution from the ``extract2d.fits`` files.
* ``4_light_curve_creation.ipynb``: this notebook shows you how to make your spectroscopic light curves from your selected wavelength bins
* ``5_reformatting_results.ipynb``: an example notebook about how to reformat the outputs from ``Tiberius`` for easier comparison with other pipelines.

1.5 Outcome
-----------

At this stage, you should have extracted 2D stellar spectra and light curves (as pickled numpy arrays) and you're able to move onto light curve fitting!
