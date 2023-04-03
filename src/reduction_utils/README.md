# SPECTROSCOPIC REDUCTION TOOLS

**Note**: this was initially designed for LRG-BEASTS with WHT/ACAM and NTT/EFOSC2 data in mind. It has since been expanded to work on Keck/NIRSPEC data (although this needs improving) and JWST Stage 1 data products. ***Files that are relevant to JWST data reduction are marked with an asterisk below***, all other files can be ignored in the analysis of JWST data.

- **ACAM_utils/** \
Contains information useful for arc calibration with WHT/ACAM and example notebooks for data reduction.

- **EFOSC_utils/** \
Contains information useful for arc calibration with NTT/EFOSC2.

- **JWST_utils/** \
Contains scripts relevant to stage 1 reduction of JWST data using the STScI jwst pipeline, along with example reduction_notebooks.

- **cosmic_removal.py*** \
Contains the functions for removal of cosmic rays.

- **extraction_input.txt*** \
The input parameter file defining the flux extraction from the science images.

- **generate_ancillary_plots.py** \
Takes the output of spectral_extraction.py and plots all diagnostic data (FWHM, x, sky, airmass..) on one large figure along with the white light curve. Not used for JWST data.

- **locate_cosmics.py*** \
A script that can locate cosmic rays and bad pixels by using a running median for every pixel in a full series of science images. This then creates a cosmic pixel mask of dimensions nexposures x nrows x ncols that can be fed directly to spectral_extraction.py.

- **master_bias.py** \
Uses a list of bias frames to combine into single master bias

- **master_flat.py** \
Takes a list of flat frames and median combines them. Then uses a running median to remove the spectrum of the source used to illuminate the CCD (whether it be the sky or a lamp).
This returns a median combined flat, and a median combined flat divided by the running median with the suffix "_FITTED" added to the file name.

- **night_movie.py** \
Reads in science data from the night and plots a movie of the frames along with the white light curve (long_slit_science_extraction.py must be run first). Outputs a gif.

- **plot_extraction_frame.py** \
Plots a science image along with the traces and apertures used in the data extraction.

- **spectral_extraction.py*** \
The script that performs the spectral tracing and aperture photometry. It is run from the command line with no additional arguments.
All the necessary input parameters are defined within extraction_input.txt. Spits out a number of pickled numpy arrays which are named as "*.pickle"

- **utils.py** \
Useful functions used by other scripts within this directory.

- **wavelength_binning.py*** \
Functions used in the wavelength binning of the data.

- **wavelength_calibration.py*** \
Functions used in the wavelength calibration of the data.

## Contributors

James Kirk
Eva-Maria Ahrer

### Reference / citing Tiberius:

Please cite the following papers if you use this code: <br>

https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3907K/abstract <br>

https://ui.adsabs.harvard.edu/abs/2021AJ....162...34K/abstract <br>

A dedicated Tiberius paper is in the pipeline.
