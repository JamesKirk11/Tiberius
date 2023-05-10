# Tiberius
The Tiberius pipeline to extract time-series spectra and fit exoplanet transit light curves.

There is a [readthedocs](https://tiberius.readthedocs.io/en/latest/) in development.

In the meantime, please checkout the READMEs under reduction_utils (to extract stellar spectra) and fitting_utils (to fit transit light curves).


### Reference / citing Tiberius:

Please cite the following papers if you use this code: <br>

https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3907K/abstract <br>

https://ui.adsabs.harvard.edu/abs/2021AJ....162...34K/abstract <br>

A dedicated Tiberius paper is in the pipeline.


### Dependencies:

To run stage 1 JWST data reduction, you need to install the STScI jwst pipeline following the instructions [here](https://jwst-pipeline.readthedocs.io/en/stable/index.html). I don't package this with Tiberius since the jwst installation requires some additional steps.  

All necessary python modules should be installed via following the installation instructions on the readthedocs. However, ExoTiC-LD also requires [additional files](https://exotic-ld.readthedocs.io/en/latest/views/installation.html) that are not packaged with Tiberius.
