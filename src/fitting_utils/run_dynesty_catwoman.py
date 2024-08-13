import catwoman
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
import argparse
from scipy.interpolate import UnivariateSpline as US


from global_utils import parseInput
from fitting_utils import mcmc_utils as mc
from fitting_utils import CatwomanModel as cwm
from fitting_utils import priors

from dynesty import utils as dyfunc

import pandas as pd
from scipy.stats import chisquare

import corner

parser = argparse.ArgumentParser(description='Run fit to a single light curve that is either a wavelength-binned or white light curve.')
parser.add_argument('wavelength_bin', help="which wavelength bin are we running the fit to? This is indexed from 0. If running fit to the white light curve, this must be given as '0'",type=int)
args = parser.parse_args()

wb = args.wavelength_bin


input_dict = parseInput('fitting_input_catwoman.txt')


### Load in time input array
time = pickle.load(open(input_dict['time_file'],'rb'))

# clipping integrations
try:
    first_integration = int(input_dict["first_integration"])
    print("\n...Clipping first %d integrations (%d minutes)"%(first_integration,24*60*(time[first_integration]-time[0])))
except:
    first_integration = 0
try:
    last_integration = int(input_dict["last_integration"])
    print("\n...Clipping beyond integration %d (%d minutes)"%(last_integration,24*60*(time[-1]-time[last_integration])))
except:
    last_integration = len(time)

time = time[first_integration:last_integration]
flux = pickle.load(open(input_dict['flux_file'],'rb'))[wb][first_integration:last_integration]
flux_error = pickle.load(open(input_dict['error_file'],'rb'))[wb][first_integration:last_integration]

### Nested sampling parameters

nested_parameters = [int(input_dict['nLive_pdim']), float(input_dict['precision_criterion'])]


### Load limb-darkening
if input_dict['use_ld_file'] is not None:
    ld_file_name = input_dict['use_ld_file']
else:
    ld_file_name = 'LD_coefficients.txt'
try:
    wc,we,u1,u1_err,u2,u2_err,u3,u3_err,u4,u4_err = np.loadtxt(ld_file_name,unpack=True)[:,wb]
except:
    raise SystemError('Need limb-darkening input!')


### define fitted parameters and setup prior dictionary
nDims = 0 
prior_dict = {}

prior_dict['k_m_prior'] = input_dict['k_m_prior']
prior_dict['k_m_1'] = float(input_dict['k_m_1'])
prior_dict['k_m_2'] = float(input_dict['k_m_2'])

prior_dict['k_e_prior'] = input_dict['k_e_prior']
prior_dict['k_e_1'] = float(input_dict['k_e_1'])
prior_dict['k_e_2'] = float(input_dict['k_e_2'])

prior_dict['u1'] = np.atleast_1d(u1)
prior_dict['u1_err'] = np.atleast_1d(u1_err)
prior_dict['u2'] = np.atleast_1d(u2)
prior_dict['u2_err'] = np.atleast_1d(u2_err)
prior_dict['u3'] = np.atleast_1d(u3)
prior_dict['u3_err'] = np.atleast_1d(u3_err)
prior_dict['u4'] = np.atleast_1d(u4)
prior_dict['u4_err'] = np.atleast_1d(u4_err)

prior_dict['ld_unc_multiplier'] = float(input_dict['ld_unc_multiplier'])

prior_dict['error_infl_prior'] = input_dict['error_infl_prior']
prior_dict['err_1'] = float(input_dict['err_1'])
prior_dict['err_2'] = float(input_dict['err_2'])

## Other fitting parameters
ld_law = input_dict["ld_law"]
FIX_U1 = bool(int(input_dict['fix_u1']))
FIX_U2 = bool(int(input_dict['fix_u2']))
FIX_U3 = bool(int(input_dict['fix_u3']))
FIX_U4 = bool(int(input_dict['fix_u4']))

k_m_e_equal = bool(int(input_dict['k_m_e_equal']))


### Initiate dictionaries with parameters to set up model
d = OrderedDict()

d['t0'] = float(input_dict['t0'])
d['inc'] = float(input_dict['inclination'])
d['aRs'] = float(input_dict['aRs'])
d['period'] = float(input_dict['period'])
d['ecc'] = float(input_dict['ecc'])
d['w'] = float(input_dict['omega'])

if k_m_e_equal:
    d['k'] = cwm.Param(float(input_dict['k']))
else:
    d['k_e'] = cwm.Param(float(input_dict['k']))
    d['k_m'] = cwm.Param(float(input_dict['k']))

if FIX_U1:
    d['u1'] = u1
else:
    d['u1'] = cwm.Param(u1)
if ld_law != "linear":
    if FIX_U2:
        d['u2'] = u2
    else:
        d['u2'] = cwm.Param(u2)
        
if ld_law == "nonlinear": 
    if FIX_U3:
        d['u3'] = u3
    else:
        d['u3'] = cwm.Param(u3)
    if FIX_U4:
        d['u4'] = u4
    else:
        d['u4'] = cwm.Param(u4)

d['infl_err'] = cwm.Param(1.)

print(d)
model = cwm.CatwomanModel(d,flux,flux_error,time,prior_dict,nested_parameters,k_m_e_equal,ld_law) #,cw_fac=0.0001
print(model.calc(time))
print(model.return_curr_parameters())
print(model.loglikelihood([10560,10560,1.0]))
result = model.run_dynesty()

pickle.dump(result, open('result_wb%s.pickle'%(str(wb+1).zfill(2)),'wb'))

result.summary()

from dynesty import utils as dyfunc
samples, weights = result.samples, result.importance_weights()
mean, cov = dyfunc.mean_and_cov(samples, weights)
equal_weights_samples = result.samples_equal()

pickle.dump(samples, open('samples_wb%s.pickle'%(str(wb+1).zfill(2)),'wb'))
pickle.dump(weights, open('weights_wb%s.pickle'%(str(wb+1).zfill(2)),'wb'))
pickle.dump(equal_weights_samples, open('equal_weights_samples_wb%s.pickle'%(str(wb+1).zfill(2)),'wb'))



## plotting
from dynesty import plotting as dyplot

# plot run
fig, ax = dyplot.cornerplot(result, color='dodgerblue', truths=np.zeros(model.nDims),
                           truth_color='black', show_titles=True,
                           quantiles=None, max_n_ticks=3)

plt.savefig('result_corner_plot_wb%s.pdf'%(str(wb+1).zfill(2)))


fig, axes = dyplot.traceplot(result, truths=np.zeros(model.nDims),
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))

plt.savefig('result_trace_plot_wb%s.pdf'%(str(wb+1).zfill(2)))

