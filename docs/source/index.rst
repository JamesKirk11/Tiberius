Welcome to Tiberius' documentation!
===================================

**Tiberius** is a Python library for reducing time series spectra and fitting exoplanet transit light curves. This can be used to extract spectra from JWST (all 4 instruments), along with ground-based long-slit spectrographs and Keck/NIRSPEC echelle spectra (beta).

The light curve fitting routines can be used as as standalone to fit, for example, HST light curves extracted with other methods.

**NOTE:** This readthedocs is under heavy development and is not yet complete. In the simplest terms, spectral extraction is performed by running ``reduction_utils/spectral_extraction.py`` and is controlled by ``reduction_utils/extraction_input.txt``. Light curve fitting is performed by running ``fitting_utils/gppm_fit.py`` and is controlled by ``fitting_utils/fitting_input.txt``.

Reference / citing Tiberius
---------------------------

If you use ``Tiberius`` in published work, please cite the following papers.

.. code-block::

  @ARTICLE{Kirk2017,
         author = {{Kirk}, J. and {Wheatley}, P.~J. and {Louden}, T. and {Doyle}, A.~P. and {Skillen}, I. and {McCormac}, J. and {Irwin}, P.~G.~J. and {Karjalainen}, R.},
          title = "{Rayleigh scattering in the transmission spectrum of HAT-P-18b}",
        journal = {\mnras},
       keywords = {methods: observational, techniques: spectroscopic, planets and satellites: atmospheres, planets and satellites: individual: HAT-P-18b, Astrophysics - Earth and Planetary Astrophysics},
           year = 2017,
          month = jul,
         volume = {468},
         number = {4},
          pages = {3907-3916},
            doi = {10.1093/mnras/stx752},
  archivePrefix = {arXiv},
         eprint = {1611.06916},
   primaryClass = {astro-ph.EP},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3907K},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }

    @ARTICLE{Kirk2021,
           author = {{Kirk}, James and {Rackham}, Benjamin V. and {MacDonald}, Ryan J. and {L{\'o}pez-Morales}, Mercedes and {Espinoza}, N{\'e}stor and {Lendl}, Monika and {Wilson}, Jamie and {Osip}, David J. and {Wheatley}, Peter J. and {Skillen}, Ian and {Apai}, D{\'a}niel and {Bixel}, Alex and {Gibson}, Neale P. and {Jord{\'a}n}, Andr{\'e}s and {Lewis}, Nikole K. and {Louden}, Tom and {McGruder}, Chima D. and {Nikolov}, Nikolay and {Rodler}, Florian and {Weaver}, Ian C.},
            title = "{ACCESS and LRG-BEASTS: A Precise New Optical Transmission Spectrum of the Ultrahot Jupiter WASP-103b}",
          journal = {\aj},
         keywords = {Exoplanet astronomy, Exoplanet atmospheres, Exoplanet atmospheric composition, Extrasolar gaseous giant planets, Hot Jupiters, Planet hosting stars, 486, 487, 2021, 509, 753, 1242, Astrophysics - Earth and Planetary Astrophysics},
             year = 2021,
            month = jul,
           volume = {162},
           number = {1},
              eid = {34},
            pages = {34},
              doi = {10.3847/1538-3881/abfcd2},
    archivePrefix = {arXiv},
           eprint = {2105.00012},
     primaryClass = {astro-ph.EP},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2021AJ....162...34K},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

A dedicated Tiberius paper is in the pipeline.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   jwst
   api
