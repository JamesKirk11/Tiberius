.. _keck:

Extracting Keck/NIRSPEC data
============================

Step 1: sort all .fits files from your night's observation

To do this, navigate to the directory containing your .fits files. Then from a terminal run the below command.

.. code-block:: bash

  $ python /path/to/Tiberius/src/reduction_utils/Keck_utils/sort_files.py -pwd [optional-path-to-Keck-fits-files]

This will write text files, with "list" in the filename, which list all combinations of object and image type in the directory.

Step 2: make flats and darks and bad pixel maps

Within the same directory, run the below command feeding the list of darks, flats and arcs to ``make_darks_and_flats.py''.

.. code-block:: bash

$ python reduction_utils/Keck_utils/make_darks_and_flats.py -dl dark_0.432x12_NIRSPEC-1_list -fl flatlamp_0.432x12_NIRSPEC-1_list -al arclamp_0.432x12_NIRSPEC-1_list -v

--> this outputs the master dark, master flat, master arc and 2 bad pixel maps, one using medians and mads (bad_pixel_mask_tight) and one using means and std devs (bad_pixel_mask_loose). The tight mask is preferred.

3. make AB difference images, as I've found that the standard sky subtraction with a polynomial performs much worse than using A-B + sky poly.

$ python ~/ACAMdata/reduction_utils/Keck_utils/AB_subtraction.py HAT-P-26_0.432x12_NIRSPEC-1_list -v

(optional) locate cosmics in frames
