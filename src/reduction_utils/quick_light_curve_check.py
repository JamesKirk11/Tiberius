#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import matplotlib.pyplot as plt
import pickle
import argparse
import wavelength_binning as wb
import numpy as np

# plt.ion()

parser = argparse.ArgumentParser(description='Plot pixel-binned light curves prior to any post extraction processing.')
parser.add_argument("-s","--save_fig",help="Use if wanting to save the resulting figure to file.")
parser.add_argument("-c1","--contact1",help="Frame number of first contact point for normalisation",type=int)
parser.add_argument("-c4","--contact4",help="Frame number of fourth contact point for normalisation",type=int)
args = parser.parse_args()


# Load in arrays

mjd = pickle.load(open('mjd_time.pickle','rb'))

s1 = pickle.load(open('star1_flux.pickle','rb'))
s1e = pickle.load(open('star1_error.pickle','rb'))

s2 = pickle.load(open('star2_flux.pickle','rb'))
s2e = pickle.load(open('star2_error.pickle','rb'))

sky1 = pickle.load(open('sky_avg_star1.pickle','rb'))
sky2 = pickle.load(open('sky_avg_star2.pickle','rb'))

# ~ approx_wvl_solution = np.arange(3800,3800+len(s1[0])*3.3,3.3) # For ACAM reductions, min wavelength is ~3500 and pixel scale is 3.5A per pixel
# ~ bins = np.arange(4000,9200,200)#3500+len(s1[0])*3.3+200,200) # 200A bins
approx_wvl_solution = np.linspace(4000,9000,len(s1[0])) # For ACAM reductions, min wavelength is ~3500 and pixel scale is 3.5A per pixel
bins = np.arange(4500,8500,200)#3500+len(s1[0])*3.3+200,200) # 200A bins
digitized_wvls = np.digitize(approx_wvl_solution,bins)

nrows = len(s1[0])

nbins = len(bins)

nframes = len(s1)


def normalise(data):
	return data/data.mean()


try:
	# Plot of target, comparison and bins
	wb.plot_spectra(s1[nframes//2],s2[nframes//2],approx_wvl_solution,bin_edges=bins,alkali=False)

	# Plot of sky and bins
	wb.plot_spectra(sky1[nframes//2],sky2[nframes//2],approx_wvl_solution,bin_edges=bins,alkali=False)
except:
	pass

# Wavelength bin data
binned_flux = []

for i in range(1,nbins+1):
	binned_flux.append(s1[:,digitized_wvls==i].sum(axis=1)/s2[:,digitized_wvls==i].sum(axis=1))

wf = np.array(binned_flux)

wf_norm = wf/wf.mean(axis=1).reshape(len(wf),1)
we_norm = np.zeros_like(wf_norm)

# Plot bins
wb.plot_all_bins(mjd,wf_norm,we_norm)

if args.contact1 is not None and args.contact4 is not None:
	rms = []
	for i in range(1,nbins):
		rms.append(np.hstack((wf_norm[i][:args.contact1],wf_norm[i][args.contact4:])).std())

	print("\n Ignoring first and last bin...")
	print("Light curve RMS med, max, min (ppm) = %d, %d, %d"%(np.median(rms)*1e6,max(rms)*1e6,min(rms)*1e6))
