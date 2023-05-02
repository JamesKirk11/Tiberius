Usage
=====

.. _installation:

Installation
------------

I am working on making Tiberius pip installable from pypi.

In the meantime, if you use conda environments, I recommend first making a new conda environment:

.. code-block:: bash

   conda create -n Tiberius python=3.8
   conda activate Tiberius

Next you'll need to download the repository from GitHub, cd into the directory and pip install within the ``Tiberius`` directory. In a terminal, this would look like:

.. code-block:: bash

   git clone https://github.com/JamesKirk11/Tiberius.git
   cd Tiberius
   pip install -e .

You'll also need to download the stellar models and instrument throughputs for ``ExoTiC-LD`` (to calculate limb darkening coefficients) following the instructions `here <https://exotic-ld.readthedocs.io/en/latest/views/installation.html>`_.

Finally, if you want to run JWST stage 1 extraction, you'll also need to install STScI's ``jwst`` pipeline, following the instructions `here <https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html>`_. Note: it's probably best to create another separate conda environment for the `jwst` pipeline. The `jwst` and `Tiberius` functionality are never needed to be run simultaneously, hence they can be in different environments.


Extracting spectra
----------------

Check out the instrument-dependent examples (coming soon, in the meantime see the README, workflows and example notebooks in reduction_utils/).

To extract JWST spectra, follow the example `here <jwst-extraction-example>`.`

Fitting light curves
----------------

Check out the examples (coming soon, in the meantime see the README, workflows and example notebooks in fitting_utils/).
