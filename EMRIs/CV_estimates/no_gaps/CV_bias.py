from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.noise import sensitivity_LWA, noise_PSD_AE
from stableemrifisher.utils import inner_product

from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t, get_separatrix
from fastlisaresponse import ResponseWrapper  # Response function 

import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from tqdm import tqdm as tqdm
import os
from pathlib import Path
import cupy as xp
import numpy as np
use_gpu=True

def inner_product_f(a_fft, b_fft, PSD, N, dt):
    return xp.real(xp.sum(xp.asarray([(4 * dt / N) * xp.sum((a_fft[j][1:] * b_fft[j][1:].conj())/PSD[j][1:]) for j in range(2)])))

M = 1e6
mu = 10.0
a = 0.9
p0 = 9.05
e0 = 0.20
iota0 = 0.3 ; Y0 = np.cos(iota0)
Phi_phi0 = 2.0
Phi_theta0 = 3.0
Phi_r0 = 4.0

qS = 1.5
phiS = 0.7
qK = 1.2
phiK = 0.6
dist = 4.0

dt = 10.0
T = 2.0


use_gpu = True

traj = EMRIInspiral(func="pn5")  # Set up trajectory module, pn5 AAK

# Compute trajectory 
if a < 0:
    a_val = -1.0 * a
    Y0_val = -1.0 * Y0
else:
    a_val = a; Y0_val = Y0

t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(M, mu, a_val, p0, e0, Y0_val,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)
traj_args = [M, mu, a, e_traj[0], Y_traj[0]]
index_of_p = 3

# Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 

p_new = get_p_at_t(
    traj,
    T,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=[8,13],
)

print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], Y_traj[-1]))

# ================ Set up inspiral kwargs ==================
inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e4),
        "err": 1e-14,  # To be set within the class
        "use_rk4": True,
        }

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": True,  # GPU is available for this type of summation
    "pad_output": True,
}

amplitude_kwargs = {
    }

outdir = 'basic_usage_stability_outdir'

Path(outdir).mkdir(exist_ok=True)
waveform_model = GenerateEMRIWaveform('Pn5AAKWaveform', return_list=False, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

t0 = 20000.0   # How many samples to remove from start and end of simulations.
order = 25

orbit_file = "../../../../../Github_repositories/lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
orbit_kwargs = dict(orbit_file=orbit_file)

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "2nd generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs, order=order, tdi=tdi_gen, tdi_chan="AE",
    )

print("Building response")
EMRI_TDI = ResponseWrapper(waveform_model,T,dt,
                index_lambda,index_beta,t0=t0,
                flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                remove_garbage = "zero", **tdi_kwargs_esa)


params = [M,mu,a,p0,e0,Y0,dist,qS,phiS,qK,phiK,Phi_phi0, Phi_theta0, Phi_r0]
h_t = EMRI_TDI(*params)

N = len(h_t[0])

h_f = [xp.fft.rfft(wave) for wave in h_t]
#varied parameters
param_names = ['M','mu','a','p0','e0','Y0','dist','qS','phiS','qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']
#initialization
fish = StableEMRIFisher(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
              Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T, EMRI_waveform_gen=EMRI_TDI,
              param_names=param_names, Ndelta = 16, interpolation_factor=10, window = None, stats_for_nerds=True, stability_plot=False, use_gpu=True, return_derivs=True,
              filename="FM_file", CovEllipse=False, live_dangerously=False)

#execution
derivs, fisher_matrix = fish()

Cov_Matrix = xp.linalg.inv(xp.asarray(fisher_matrix))

freq_bin = np.fft.rfftfreq(len(h_t[0]), dt)
freq_bin[0] = freq_bin[1]
if tdi_gen == "1st generation":
    PSD = xp.asarray(2*[noise_PSD_AE(freq_bin, TDI = "TDI1")])
else:
    PSD = xp.asarray(2*[noise_PSD_AE(freq_bin, TDI = "TDI2")]) 

SNR2 = xp.asarray([4 * dt * xp.sum(abs(wave)**2 / (N*PSD[0])) for wave in h_f])
SNR = xp.sum(SNR2)**(1/2) 

variance_noise_f = N * PSD[0] / (4*dt)
derivs_f = [xp.fft.rfft(derivs[j]) for j in range(0,len(derivs))]

noise_MLE_vec = []
for i in tqdm(range(0,10000)):

    np.random.seed(i)

    noise_f_I = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j * xp.random.normal(0,xp.sqrt(variance_noise_f))
    noise_f_I[0] = np.sqrt(2)*noise_f_I[0].real
    noise_f_I[-1] = np.sqrt(2)*noise_f_I[-1].real

    noise_f_II = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j * xp.random.normal(0,xp.sqrt(variance_noise_f))
    noise_f_II[0] = np.sqrt(2)*noise_f_II[0].real
    noise_f_II[-1] = np.sqrt(2)*noise_f_II[-1].real
    
    noise_f = xp.asarray([noise_f_I,noise_f_II])

    # Compute inner products


    bias_vec = xp.asarray([inner_product_f(derivs_f[j], noise_f, PSD, N, dt) for j in range(0,len(derivs))])

    noise_MLE = xp.asarray(params[0:len(param_names)]) + Cov_Matrix @ bias_vec 

    noise_MLE_vec.append(noise_MLE)

True_Cov_Matrix = xp.cov(xp.asarray(noise_MLE_vec), rowvar = False)

for i in range(len(params)):
    print("Approx precision, using whittle, in parameter {} is {}".format(param_names[i], np.sqrt(np.diag(Cov_Matrix))[i]))
for i in range(len(params)):
    print("True precision in parameter {} is {}".format(param_names[i], np.sqrt(np.diag(True_Cov_Matrix))[i]))

breakpoint()



