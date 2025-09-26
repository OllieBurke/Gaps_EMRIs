import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import os
import numpy as np
import sys
sys.path.append("../..")
from few.utils.constants import YRSID_SI
from stableemrifisher.noise import write_psd_file, load_psd_from_file
from utility_funcs.hdf5_file_management import load_fisher_results_from_hdf5, add_monte_carlo_to_existing_file, cp
from CV_bias_func import generate_colored_noise, inner_product_frequency_domain
import h5py

noise_direc = "/work/scratch/data/burkeol/Gaps_EMRIs/noise/"
if cp is None:
    xp = np
    return_as_cupy = False
else:
    xp = cp
    return_as_cupy = True
# ================== BASIC LOADING ==================

# Load in Fisher matrix results
filename = "/work/scratch/data/burkeol/Gaps_EMRIs/Fisher_Matrices/Fisher_Matrix_Case_1.h5"
fisher_results = load_fisher_results_from_hdf5(filename, return_as_cupy=return_as_cupy)
param_cov = fisher_results['inverse_fisher_matrix']
derivs = fisher_results['derivatives']
dt = fisher_results['observation_parameters']['sampling_interval_seconds']
T = fisher_results['observation_parameters']['observation_time_years']
gap_window_func = fisher_results['gap_analysis']['window_function']
EMRI_parameters = fisher_results['emri_parameters']

# Define the EMRI_parameters
Fisher_EMRI_params = EMRI_parameters.copy()
Fisher_EMRI_params.pop("xI0", None)  # None as default if key doesn't exist
Fisher_EMRI_params.pop("Phi_theta0", None)

Fisher_EMRI_param_values = xp.asarray(list(Fisher_EMRI_params.values()))

N = len(gap_window_func)
freq_bin = np.fft.rfftfreq(N, dt)
freq_bin[0] = freq_bin[1]

PSD_filename = "tdi2_AE_w_background.npy"
kwargs_PSD = {"stochastic_params": [T*YRSID_SI]} # We include the background

write_PSD = write_psd_file(model='scirdv1', channels='AE', 
                           tdi2=True, include_foreground=True, 
                           filename = noise_direc + PSD_filename, **kwargs_PSD)

PSD_AE_interp = load_psd_from_file(noise_direc + PSD_filename, xp=cp)

PSD = PSD_AE_interp(freq_bin)

variance_noise_f = N * PSD[0] / (4*dt)
derivs_f = [xp.fft.rfft(derivs[j]) for j in range(0,len(derivs))]

noise_MLE_vec = []

N_total = 10
seeds_used = np.arange(0,N_total,1)
for i in tqdm(seeds_used):

    noise_f = generate_colored_noise(variance_noise_f, seed = i, window_function = None, return_time_domain=False)

    bias_vec = xp.asarray([inner_product_frequency_domain(derivs_f[j], noise_f, PSD, N, dt) for j in range(0,len(derivs))])

    noise_MLE = Fisher_EMRI_param_values + param_cov @ bias_vec 

    noise_MLE_vec.append(noise_MLE)

noise_MLE_vec_array = xp.asarray(noise_MLE_vec)
True_Cov_Matrix = xp.cov(noise_MLE_vec_array, rowvar = False)

# Package for HDF5
monte_carlo_results = {
    'noise_MLE_vec': noise_MLE_vec_array,
    'seeds': seeds_used,
    'True_Cov_Matrix': True_Cov_Matrix,
    'n_realizations': N_total
}

add_monte_carlo_to_existing_file(filename, monte_carlo_results)




