# SPECTROSCOPIC REDUCTION TOOLS

**Note**: this was initially designed for LRG-BEASTS with WHT/ACAM and NTT/EFOSC2 data in mind. It has since been expanded to work on Keck/NIRSPEC data (although this needs improving) and JWST Stage 1 data products. ***Files that are relevant to JWST data reduction are marked with an asterisk below***, all other files can be ignored in the analysis of JWST data.

- **ACAM_utils/** \
Contains information useful for arc calibration with ACAM

- **conda_env_help.txt** \
Contains information about setting up conda environments. This allows you to self-manage Python and self-install packages on a computing system that is centrally managed by IT services who won't allow you to download and install your own Python version.

- **cosmic_removal.py*** \
Contains the functions for removal of cosmic rays. See example iPython notebook for how this implemented.

- **EFOSC_utils/** \
Contains information useful for arc calibration with EFOSC

- **example_notebooks***\
Example cosmic removal, wavelength calibration, spectral resampling, binning and white light curve generation.

- **extraction_input.txt*** \
The input parameter file defining the flux extraction from the science images.

- **generate_ancillary_plots.py** \
Takes the output of long_slit_science_extraction.py and plots all diagnostic data (FWHM, x, sky, airmass..) on one large figure along with the white light curve.

- **line_lists/**\
Useful line lists for wavelength calibration, thanks to Amanda Doyle. Also contains ATLAS9 stellar atmosphere models.

- **locate_cosmics.py*** \
A script that can locate cosmic rays and bad pixels by using a running median for every pixel in a full series of science images. This then creates a cosmic pixel mask of dimensions nexposures x nrows x ncols that can be fed directly to long_slit_science_extraction.py.

- **long_slit_science_extraction.py*** \
The script that performs the spectral tracing and aperture photometry. It is run from the command line with no additional arguments.
All the necessary input parameters are defined within extraction_input.txt. Spits out a number of pickled numpy arrays which are named as "*.pickle"

- **master_bias.py** \
Uses a list of bias frames to combine into single master bias

- **master_flat.py** \
Takes a list of flat frames and median combines them. Then uses a running median to remove the spectrum of the source used to illuminate the CCD (whether it be the sky or a lamp).
This returns a median combined flat, and a median combined flat divided by the running median with the suffix "_FITTED" added to the file name.

- **night_movie.py** \
Reads in science data from the night and plots a movie of the frames along with the white light curve (long_slit_science_extraction.py must be run first). Outputs a gif

- **night_movie_no_gif.py** \
Same as above but plots to screen without outputting a gif.

- **plot_extraction_frame.py** \
Plots an ACAM science image along with the traces and apertures used in the data extraction.

- **quick_light_curve_check.py** \
Plots pixel binned light curves as a quick qualitative check of the light curves before any post-extraction processing.

- **rotate_spectra.py*** \
This rotates all spectra from being dispersed along the x-direction to dispersed along the y-direction, as required by long_slit_science_extraction.py. This saves the rotated spectra to a new directory ("./spec_rot"). Note: this can also be performed on the fly by
long_slit_science_extraction.py so is not entirely necessary.

- **sort_ACAM_files.py** \
Takes all files within a current directory and sorts into bias, arcs, flats and science, given a specific
window setting for ACAM data.

- **sort_EFOSC_files.py** \
Takes all files within a current directory and sorts into bias, arcs, flats and science.

- **telescope_diagnostics_ACAM.py**
Reads in the fits headers of all science images and saves diagnostics (rotator angle etc.) to a table.

- **wavelength_binning.py*** \
Functions used in the wavelength binning of the data

- **wavelength_calibration.py*** \
Functions used in the wavelength calibration of the data.

- **white_light_fitting.py** \
Loads in system parameters from a file within the current directory (this needs to be created by the user) and fits a Mandel & Agol model to the white light curve with a quoted precision in the residuals of the fit. This is useful for a check of how well a reduction has performed. However, I now prefer to use fitting_utils to do the white light curve fitting.


## Dependencies (non-standard)

- peakutils (via pip install peakutils)
- pysynphot (via pip install pysynphot)
- fitting_utils (via git clone)
- astroscrappy (if wanting to perform autmoatic cosmic ray detection during extraction. This is not fully tested and hence not totally necessary)

## Contributors

James Kirk
Eva-Maria Ahrer

### Reference / citing Tiberius:

Please cite the following papers if you use this code: <br>

https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3907K/abstract <br>

https://ui.adsabs.harvard.edu/abs/2021AJ....162...34K/abstract <br>

A dedicated Tiberius paper is in the pipeline.
