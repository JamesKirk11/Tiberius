#### Author of this code: Natasha Batalha, adapted by James Kirk
#### Contact: jameskirk@live.co.uk

import pandas as pd
import numpy as np
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore")
	import pysynphot.binning as astrobin
import warnings as warn
import astropy.units as u
from Tiberius.src.fitting_utils import plotting_utils as pu

def binning(x, y,  dy=None, binwidth=None, r=None,newx= None, log = False, nan=False):
	"""
	This contains functionality for binning spectroscopy given an x, y and set of errors.
	This is similar to IDL's regroup but in Python (obviously). Note that y is binned as the
	mean(ordinates) instead of sum(ordinates), as you would want for cmputing flux in a set of
	pixels. User can input a constant resolution, constant binwidth or provide a user defined
	bin. The error is computed as sum(sqrt(sig1^2, sig2^2, sig3^2) )/3, for example
	if there were 3 points to bin.

	Parameters
	----------
	x : array, float
		vector containing abcissae.
	y : array,float
		vector containing ordinates.
	dy : array,float
		(Optional) errors on ordinates, can be float or array
	binwidth : float
		(Optional) constant bin width in same units as x
	r : float
		(Optional) constant resolution to bin to. R is defined as w[1]/(w[2] - w[0])
		to maintain consistency with `pandeia.engine`
	newx : array, float
		(Optional) new x axis to bin to
	log : bool
		(Optional) computes equal bin spacing logarithmically, Default = False
	sort : bool
		(Optional) sort into ascending order of x, default = True
	nan : bool
		(Optional) if true, this returns nan values where no points exist in a given bin
		Otherwise, all nans are dropped

	Returns
	-------
	dict
		bin_y : binned ordinates
		bin_x : binned abcissae
		bin_edge : edges of bins (always contains len(bin_x)+1 elements)
		bin_dy : error on ordinate bin
		bin_n : number of points contained in each bin

	Examples
	--------

	>>> from bintools import binning

	If you want constant resolution (using output dict from PandExo):

	>>> pandexo = result['FinalSpectrum']
	>>> x, y, err = pandexo['wave'], pandexo['spectrum_w_rand'], pandexo['error_w_floor']
	>>> final = binning(x, y, dy = err, r =100)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']

	If you have a x axis that you want PandExo output to be binned to

	>>> newx = np.linspace(1,5,10)
	>>> final = binning(x, y, dy = err, newx =newx)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']

	If you want a constant bin width and want everything to be spaced linearly

	>>> final = binning(x, y, dy = err, binwidth = 0.1)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']

	If you want constant bin width but want everything to spaced logarithmically

	>>> final = binning(x, y, dy = err, binwidth = 0.1, log=True)
	>>> newx, newy, newerr = final['bin_x'], final['bin_y'], final['bin_dy']
	"""
	#check x and y are same length
	if len(x) != len(y):
		raise Exception('X and Y are not the same length')

	#check that either newx or binwidth are specified

	if newx is None and binwidth is None and r is None:
		raise Exception('Need to either supply new x axis, resolution, or a binwidth')
	if (binwidth is None) and (log):
		raise Exception("Cannot do logarithmic binning with out a binwidth")

	if newx is not None:
		bin_x = newx
		bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
		bin_edge = astrobin.calculate_bin_edges(bin_x)

		return {'bin_y':bin_y, 'bin_x':bin_x, 'bin_edge':bin_edge, 'bin_dy':bin_dy, 'bin_n':bin_n}

	elif r is not None:
		bin_x = bin_wave_to_R(x, r)
		bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
		bin_edge = astrobin.calculate_bin_edges(bin_x)

		return {'bin_y':bin_y, 'bin_x':bin_x, 'bin_edge':bin_edge, 'bin_dy':bin_dy, 'bin_n':bin_n}


	elif binwidth is not None:
		if (binwidth < 0) and (log):
			warn.warn(UserWarning("Negative binwidth specified. Assuming this is log10(binwidth)"))
			binwidth = 10**binwidth
		if log:
			bin_x = np.arange(np.log10(min(x)),np.log10(max(x)),np.log10(binwidth))
			bin_x = 10**bin_x
		elif not log:
			bin_x = np.arange(min(x),max(x),binwidth)
		bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
		bin_edge = astrobin.calculate_bin_edges(bin_x)

		return {'bin_y':bin_y, 'bin_x':bin_x, 'bin_edge':bin_edge, 'bin_dy':bin_dy, 'bin_n':bin_n}




def uniform_tophat_mean(newx,x, y, dy=None,nan=False):
	"""Adapted from Mike R. Line to rebin spectra

	Takes mean of groups of points in certain wave bin

	Parameters
	----------
	newx : list of float or numpy array of float
	    New wavelength grid to rebin to
	x : list of float or numpy array of float
	    Old wavelength grid to get rid of
	y : list of float or numpy array of float
	    New rebinned y axis

	Returns
	-------
	array of floats
	    new wavelength grid

	Examples
	--------

	>>> from pandexo.engine.jwst import uniform_tophat_sum
	>>> oldgrid = np.linspace(1,3,100)
	>>> y = np.zeros(100)+10.0
	>>> newy = uniform_tophat_sum(np.linspace(2,3,3), oldgrid, y)
	>>> newy
	array([ 240.,  250.,  130.])
	"""
	newx = np.array(newx)
	szmod=newx.shape[0]
	delta=np.zeros(szmod)
	ynew=np.zeros(szmod)
	bin_dy =np.zeros(szmod)
	bin_n =np.zeros(szmod)

	delta[0:-1]=newx[1:]-newx[:-1]
	delta[szmod-1]=delta[szmod-2]

	for i in range(szmod-1):
		i=i+1
		loc=np.where((x >= newx[i]-0.5*delta[i-1]) & (x < newx[i]+0.5*delta[i]))
		#make sure there are values within the slice
		if len(loc[0]) > 0:
			ynew[i]=np.mean(y[loc])
			if dy is not None:
				bin_dy[i] = np.sqrt(np.sum(dy[loc]**2.0))/len(y[loc])
			bin_n[i] = len(y[loc])
		#if not give empty slice a nan
		elif len(loc[0]) is 0 :
			warn.warn(UserWarning("Empty slice exists within specified new x, replacing value with nan"))
			ynew[i]=np.nan
			bin_n[i] = np.nan

	#fill in zeroth entry
	loc=np.where((x > newx[0]-0.5*delta[0]) & (x < newx[0]+0.5*delta[0]))
	if len(loc[0]) > 0:
		ynew[0]=np.mean(y[loc])
		bin_n[0] = len(y[loc])
		if dy is not None:
			bin_dy[0] = np.sqrt(np.sum(dy[loc]**2.0))/len(y[loc])
	elif len(loc[0]) is 0 :
		ynew[0]=np.nan
		bin_n[0] = np.nan
		if dy is not None:
			bin_dy[0] = np.nan

	#remove nans if requested
	out = pd.DataFrame({'bin_y':ynew, 'bin_x':newx, 'bin_dy':bin_dy, 'bin_n':bin_n})
	if not nan:
		out = out.dropna()

	return out['bin_x'].values,out['bin_y'].values, out['bin_dy'].values, out['bin_n'].values


# def bin_wave_to_R(w, R):
# 	"""Creates new wavelength axis at specified resolution
#
# 	Parameters
# 	----------
# 	w : list of float or numpy array of float
# 	    Wavelength axis to be rebinned
# 	R : float or int
# 	    Resolution to bin axis to
#
# 	Returns
# 	-------
# 	list of float
# 	    New wavelength axis at specified resolution
#
# 	Examples
# 	--------
#
# 	>>> newwave = bin_wave_to_R(np.linspace(1,2,1000), 10)
# 	>>> print((len(newwave)))
# 	11
# 	"""
# 	wave = []
# 	tracker = min(w)
# 	i = 1
# 	ind= 0
# 	firsttime = True
# 	while(tracker<max(w)):
# 	    if i <len(w)-1:
# 	        dlambda = w[i]-w[ind]
# 	        newR = w[i]/dlambda
# 	        if (newR < R) & (firsttime):
# 	            tracker = w[ind]
# 	            wave += [tracker]
# 	            ind += 1
# 	            i += 1
# 	            firsttime = True
# 	        elif newR < R:
# 	            tracker = w[ind]+dlambda/2.0
# 	            wave +=[tracker]
# 	            ind = (np.abs(w-tracker)).argmin()
# 	            i = ind+1
# 	            firsttime = True
# 	        else:
# 	            firsttime = False
# 	            i+=1
# 	    else:
# 	        tracker = max(w)
# 	        wave += [tracker]
# 	return wave


def bin_wave_to_R(w,R):
	"""JK's function to bin a wavelength grid to a specified resolution

	Parameters
	----------
	w : list of float or numpy array of float
	    Wavelength axis to be rebinned
	R : float or int
	    Resolution to bin axis to

	Returns
	-------
	list of float
	    New wavelength axis at specified resolution
	"""

	wvls = []
	starting_wvl = w[0]
	stopping_wvl = w[-1]
	i = starting_wvl
	while i < stopping_wvl:
	    delta_lam = i/R
	    wvls.append(i+delta_lam)
	    i += delta_lam
	return np.array(wvls)


def bin_trans_spec(bin_edges,x,y,e1,e2=None):
    """JK's function to bin a transmission spectrum to a lower resolution using weighted means of depths within a bin.

    Inputs:
    bin_edges -- an array of the *edges* of the bins, not the centres
    x -- the input wavelength array
    y -- the input spectrum (depths or Rp/Rs)
    e1 -- the 1 sigma uncertainties on y. Can be defined as the upper errors if errors are asymmetric
	e2 -- the 1 sigma uncertainties on y. Can be defined as the lower errors if errors are asymmetric

    Returns:
    A dictionary of 'bin_x' the bin *centres*, 'bin_dx' the bin widths, 'bin_y' the binned spectrum, 'bin_dy' the binned uncertainties on the spectrum"""

    digitized = np.digitize(x,bin_edges)

    binned_x = []
    binned_xe = []
    binned_y = []
    binned_e = []

    for i in range(1,len(bin_edges)):
        if e2 is not None:
            mean_y,mean_e = pu.weighted_mean_uneven_errors(y[digitized==i],e1[digitized==i],e2[digitized==i])
        else:
	        mean_y,weights = np.average(y[digitized==i],weights=1/e[digitized==i]**2,returned=True)
	        mean_e = np.sqrt(1/weights)

        binned_x.append(x[digitized==i].mean())
        binned_xe.append(x[digitized==i].max()-x[digitized==i].min())
        binned_y.append(mean_y)
        binned_e.append(mean_e)

    binned = {"bin_x":np.array(binned_x),"bin_dx":np.array(binned_xe),"bin_y":np.array(binned_y),"bin_dy":np.array(binned_e)}

    return binned
