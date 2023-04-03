## GP and parametric model fitting to exoplanet transit light curves

### Contains:

**compare_transmission_spectra.py** - the script that compares the transmission spectra resulting from n different transmission spectra. Can be useful to compare between nights or compare various fitting models for the same night. Also used to combine transmission spectra from multiple nights. <br>

**fit_all_parametric_models.py** - the script that fits all possible combinations of polynomial (up to cubic) in time, airmass, FWHM, x positions, y positions and sky background to the white light curve. NOTE: it has been a while since this was last used and it is possible that this will not work out of the box. <br>

**generate_LDCS.py** - the script that uses Limb Darkening Toolkit or ExoTiC-LD to generate the limb darkening coefficients for the user defined wavelength bins. <br>

**gppm_fit.py** - the script that fits a single light curve with a GP + (optional) polynomial in time. <br>

**__init__.py** <br>

**latex_table_generator.py** - script that converts results tables into LaTex tables - Note not fully tested. <br>

**lc_table_generator.py** - script that converts pickle files into text files with multiple columns for time, flux and ancillary data <br>

**fitting_input.txt** - example input file that controls much of what goes on here. Is used by gppm_fit.py, pm_fit.py, plot_output.py and generate_LDCS.py <br>

**mcmc_utils.py** - functions used by the MCMC <br>

**mjd2bjd.py** - a script that converts MJD to BJD. Note: not fully tested and observatory needs changing within script. Copyright James McCormac <br>

**model_table_generator.py** - script that converts pickle files of best fitting models into text files with multiple columns for time, flux, model, GP model etc. <br>

**parametric_fitting_functions.py** - functions used by parametric models <br>

**plot_output.py** - plot the fits to wavelength-binned light curves, transmission spectrum and expected vs. calculated LDCs. Can be run while MCMC is still running to remaining wavelength bins. <br>

**plotting_utils.py** - functions used for plotting.  <br>

**README.md** - This file. <br>

**run_gppm_fit** - A bash script that loops through all wavelength bins and runs gppm_fit.py to each. NOTE: Need to edit this file with path to fitting_utils. Run from the command line as $ ./run_gppm_fit [first_bin] [last_bin] where the bins are indexed from zero. <br>

**TransitModelGPPM.py** - the TransitModelGPPM (GP + parametric model) class. <br>

**workflow.txt** - an example workflow for using this library. <br>

### Contributors:

James Kirk

### Reference / citing Tiberius:

Please cite the following papers if you use this code: <br>

https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3907K/abstract <br>

https://ui.adsabs.harvard.edu/abs/2021AJ....162...34K/abstract <br>

A dedicated Tiberius paper is in the pipeline.
