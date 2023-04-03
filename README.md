# Tiberius
The Tiberius pipeline to extract time-series spectra and fit exoplanet transit light curves.

There is a readthedocs in development.

In the meantime, please checkout the READMEs under reduction_utils (to extract stellar spectra) and fitting_utils (to fit transit light curves).


### Reference / citing Tiberius:

Please cite the following papers if you use this code: <br>

https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3907K/abstract <br>

https://ui.adsabs.harvard.edu/abs/2021AJ....162...34K/abstract <br>

A dedicated Tiberius paper is in the pipeline.


### Dependencies:

To run stage 1 JWST data reduction, you need to install the STScI jwst pipeline following the instructions [here](https://jwst-pipeline.readthedocs.io/en/stable/index.html). I don't package this with Tiberius since the jwst installation requires some additional steps.  

 -- these should be automatically installed via pip install Tiberius but note that ExoTiC-LD also requires [additional files](https://exotic-ld.readthedocs.io/en/latest/views/installation.html). <br>

 argparse <br>
 astropy <br>
 astroscrappy <br>
 batman-package <br>
 corner <br>
 emcee <br>
 exotic-ld - Requires additional data download. See https://exotic-ld.readthedocs.io/en/latest/views/installation.html <br>
 george <br>
 h5netcdf <br>
 ipykernel <br>
 ldtk <br>
 jupyter <br>
 matplotlib <br>
 numpy <br>
 pandas <br>
 peakutils <br>
 photutils <br>
 scipy <br>
 xarray <br>
