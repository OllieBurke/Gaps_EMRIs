# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastKerrEccentricEquatorialFlux,
)

from fastlisaresponse import ResponseWrapper  # Response
from lisatools.detector import EqualArmlengthOrbits

import numpy as np

from stableemrifisher.fisher import StableEMRIFisher
from few.utils.constants import YRSID_SI
from lisagap import GapMaskGenerator, GapWindowGenerator
from stableemrifisher.noise import load_psd_from_file, write_psd_file
from window_func_info import (gap_definitions, taper_defs, 
                              include_planned, include_unplanned, 
                              planned_seed, unplanned_seed)


import sys
sys.path.append("../..")
from utility_funcs.hdf5_file_management import save_fisher_results_to_hdf5, cp

noise_direc = "/work/scratch/data/burkeol/Gaps_EMRIs/noise/"
# ================== My own settings ========================
if cp is not None:
    xp = cp
else:
    xp = np

# User settings
NO_MASK = False
MASK = False
WINDOW = True
if NO_MASK:
    filename = "Fisher_Matrix_Case_1_no_window.h5"
elif MASK:
    # filename = "Fisher_Matrix_Case_1_w_mask_antenna.h5"
    # filename = "Fisher_Matrix_Case_1_w_mask_PAAM_and_antenna.h5"
    # filename = "Fisher_Matrix_Case_1_w_mask_PAAM.h5"
    # filename = "Fisher_Matrix_Case_1_w_mask_big_gaps.h5"
    # filename = "Fisher_Matrix_Case_1_w_mask_full_shamalama.h5"
    filename = "Fisher_Matrix_Case_1_w_mask_full_shamalama_with_spice.h5"
elif WINDOW:
    # filename = "Fisher_Matrix_Case_1_w_window.h5"
    # filename = "Fisher_Matrix_Case_1_w_window_full_shamalama_with_spice_planned_unplanned_10min_PAAM_10_sec.h5"
    filename = "Fisher_Matrix_Case_1_w_window_full_shamalama_with_spice_planned_unplanned_10min_PAAM_1_min.h5"
gap_info = {
    'planned_seed': planned_seed,
    'unplanned_seed': unplanned_seed,
    'include_planned': include_planned,
    'include_unplanned': include_unplanned,
    'apply_tapering': True,
    'WINDOW': WINDOW,
    'gap_definitions': gap_definitions,
    'taper_definitions': taper_defs
}
ONE_HOUR = 60 * 60
xI0 = 1.0
# ================== CASE 1 PARAMETERS ======================
T = 2.0
dt = 5.0
# EMRI Case 1 parameters as dictionary
emri_params = {
    # Masses and spin
    "m1": 1e6,  # Primary mass (solar masses)
    "m2": 10,  # Secondary mass (solar masses)
    "a": 0.998,  # Dimensionless spin parameter (near-extremal)
    # Orbital parameters
    # "p0": 7.7275,  # Initial semi-latus rectum
    "p0": 7.7375,  # Initial semi-latus rectum
    "e0": 0.73,  # Initial eccentricity
    "xI0": xI0,  # cos(inclination) - equatorial orbit
    # Source properties
    "dist": 2.20360838037185,  # Distance (Gpc) - calibrated for target SNR
    # Sky location (source frame)
    "qS": 0.8,  # Polar angle
    "phiS": 2.2,  # Azimuthal angle
    # Kerr spin orientation
    "qK": 1.6,  # Spin polar angle
    "phiK": 1.2,  # Spin azimuthal angle
    # Initial phases
    "Phi_phi0": 2.0,  # Azimuthal phase
    "Phi_theta0": 0.0,  # Polar phase
    "Phi_r0": 3.0,  # Radial phase
}

###========================SET UP GAP SITUATION ===========================
# Initialise the class with simulation properties and whether or not to treat gaps with
# nans or not. 
sim_t = np.arange(0,T*YRSID_SI - dt,dt) #TODO: UNDERSTAND WHY NOT COMPATIBLE WITH FEW!
gap_mask_gen = GapMaskGenerator(sim_t, 
                                gap_definitions, 
                                treat_as_nan = False, 
                                planseed = planned_seed,
                                unplanseed = unplanned_seed)

gap_window_func = GapWindowGenerator(gap_mask_gen)


if NO_MASK:
    gap_window_array = xp.ones(len(sim_t)) # No window at all
elif MASK:
    gap_window_array = xp.asarray(gap_window_func.generate_window(include_planned=include_planned, 
                                                   include_unplanned=include_unplanned, 
                                                   apply_tapering=False, 
                                                   taper_definitions=taper_defs))
elif WINDOW:
    gap_window_array = xp.asarray(gap_window_func.generate_window(include_planned=include_planned, 
                                                   include_unplanned=include_unplanned, 
                                                   apply_tapering=False, 
                                                   taper_definitions=taper_defs))
    print(f"For lobes = {taper_defs['planned']['PAAM']} we find sum of window is {np.sum(gap_window_array)}")
    # breakpoint()

PSD_filename = "tdi2_AE_w_background.npy"
kwargs_PSD = {"stochastic_params": [T*YRSID_SI]} # We include the background

write_PSD = write_psd_file(model='scirdv1', channels='AE', 
                           tdi2=True, include_foreground=True, 
                           filename = noise_direc + PSD_filename, **kwargs_PSD)

PSD_A_interp = load_psd_from_file(noise_direc + PSD_filename, xp=cp)

####=======================True Responsed waveform==========================
# waveform class setup
waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = {
    "inspiral_kwargs": {
        "err": 1e-11,
    },
    "sum_kwargs": {"pad_output": True},  # Required for plunging waveforms
    # "mode_selector_kwargs": {"mode_selection_threshold": 1e-5},
}

# waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = {"return_list": False, "frame": "detector"}


# ========================= SET UP RESPONSE FUNCTION ===============================#
USE_GPU = True

tdi_kwargs = dict(
    orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
    order=25,  # Order of Lagrange interpolant, used for fractional delays.
    tdi="2nd generation",  # Use second generation TDI variables
    tdi_chan="AE",
)

INDEX_LAMBDA = 8
INDEX_BETA = 7

t0 = 20000.0  # throw away on both ends when our orbital information is weird

# Set up Response key word arguments
ResponseWrapper_kwargs = dict(
    Tobs=T,
    dt=dt,
    index_lambda=INDEX_LAMBDA,
    index_beta=INDEX_BETA,
    t0=t0,
    flip_hx=True,
    use_gpu=USE_GPU,
    is_ecliptic_latitude=False,
    remove_garbage="zero",
    **tdi_kwargs,
)

der_order = 4  # Fourth order derivatives

Ndelta = 8  # Check 8 possible delta values to check convergence of derivatives

# No noise model provided so will default to TDI2 A and E channels with galactic confusion noise
# extracts relevant noise model from information provided to tdi_kwargs.

# Initialise fisher matrix
sef = StableEMRIFisher(
    # Set up waveform class
    waveform_class=waveform_class,
    waveform_class_kwargs=waveform_class_kwargs,
    # Set up waveform generator
    waveform_generator=waveform_generator,
    waveform_generator_kwargs=waveform_generator_kwargs,
    # Set up response
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=ResponseWrapper_kwargs,
    # Set up noise model
    noise_model = PSD_A_interp,
    stats_for_nerds=True,  # Output useful information governing stability
    use_gpu=USE_GPU,  # select whether or not to use gpu
    der_order=der_order,  # derivative order
    Ndelta=Ndelta,  # delta spacing
    return_derivatives=True,  # Do not return derivatives
    filename="saved_FM/",
    deriv_type="stable",  # Type of derivative
)

# Specify full parameter set to compute Fisher matrix over
param_names = [
    "m1",
    "m2",
    "a",
    "p0",
    "e0",
    "dist",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_r0",
]

# Compute specific delta ranges
delta_range = dict(
    m1=np.geomspace(1e2, 1e-3, Ndelta),
    m2=np.geomspace(1e-1, 1e-7, Ndelta),
    a=np.geomspace(1e-5, 1e-9, Ndelta),
    p0=np.geomspace(1e-5, 1e-9, Ndelta),
    e0=np.geomspace(1e-5, 1e-9, Ndelta),
    Y0=np.geomspace(1e-4, 1e-9, Ndelta),
    qS=np.geomspace(1e-3, 1e-7, Ndelta),
    phiS=np.geomspace(1e-3, 1e-7, Ndelta),
    qK=np.geomspace(1e-3, 1e-7, Ndelta),
    phiK=np.geomspace(1e-3, 1e-7, Ndelta),
)

print("Computing FM")
# Compute the fisher matrix

# rho = sef.VSNRcalc_SEF(*list(emri_params.values()),
#                     window=gap_window_array)

derivs, fisher_matrix = sef(emri_params, 
                            param_names=param_names, 
                            live_dangerously = True, 
                            delta_range=delta_range,
                            window = gap_window_array)
# Compute paramter covariance matrix
param_cov = np.linalg.inv(fisher_matrix)

# Print precision measurements on parameters
for k, item in enumerate(param_names):
    print(
        "Precision measurement in param {} is {}".format(
            item, param_cov[k, k] ** (1 / 2)
        )
    )

# After you generate your gap window


save_fisher_results_to_hdf5(
    filename, 
    param_cov, 
    derivs, 
    param_names, 
    additional_metadata=None, 
    base_dir="/work/scratch/data/burkeol/Gaps_EMRIs/Fisher_Matrices",
    emri_params=emri_params,
    T=T,
    dt=dt,
    gap_window_array=gap_window_array,  # Your window function
    gap_info=gap_info                   # Gap generation settings
)