## GP and parametric model fitting to exoplanet transit light curves

### Contains:

**compare_transmission_spectra.py** - the script that compares the transmission spectra resulting from n different transmission spectra. Can be useful to compare between nights or compare various fitting models for the same night. Also used to combine transmission spectra from multiple nights. <br>

**fitting_input.txt** - example input file that controls much of what goes on here. Is used by gppm_fit.py, pm_fit.py, plot_output.py and generate_LDCS.py <br>

**generate_LDCS.py** - the script that uses Limb Darkening Toolkit or ExoTiC-LD to generate the limb darkening coefficients for the user defined wavelength bins. <br>

**gppm_fit.py** - the script that fits a single light curve with a GP + (optional) polynomial in time. <br>

**mcmc_utils.py** - functions used by the MCMC and Levenberg-Marquadt fitters. <br>

**mjd2bjd.py** - a script that converts MJD to BJD. Written by James McCormac. <br>

**model_table_generator.py** - script that converts pickle files of best fitting models into text files with multiple columns for time, flux, model, GP model etc. <br>

**parametric_fitting_functions.py** - functions used by parametric/polynomial models. <br>

**plot_output.py** - plot the fits to wavelength-binned light curves, transmission spectrum and expected vs. calculated LDCs. Can be run while MCMC is still running to remaining wavelength bins. <br>

**plotting_utils.py** - functions used for plotting.  <br>

**README.md** - This file. <br>

**run_gppm_fit.py** - A python script that loops through all wavelength bins and runs gppm_fit.py to each. $ python run_gppm_fit.py [first_bin] [last_bin] where the bins are indexed from zero. <br>

**TransitModelGPPM.py** - the TransitModelGPPM (GP + parametric model) class. <br>

**workflow.txt** - an example workflow for using this library. <br>

### Contributors:

James Kirk

### Reference / citing Tiberius:

Please cite the following papers if you use this code: <br>

https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3907K/abstract <br>

https://ui.adsabs.harvard.edu/abs/2021AJ....162...34K/abstract <br>

A dedicated Tiberius paper is in the pipeline.
