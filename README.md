# Gaps_EMRIs

A comprehensive toolkit for studying the impact of data gaps on long-duration Extreme Mass Ratio Inspiral (EMRI) signals in LISA gravitational wave analysis.

## Overview

This repository provides tools and analysis pipelines to investigate how data gaps affect parameter estimation and signal detection for EMRIs. The codebase includes methods for gap simulation, noise imputation, PSD estimation with missing data, and Fisher matrix calculations with masked time series.

## Key Features

- **Gap Simulation**: Generate realistic LISA data gaps (planned and unplanned)
- **Noise Imputation**: Fill data gaps using segment-based PSD estimation
- **Bias Analysis**: Study parameter estimation bias due to data gaps
- **Smooth Windowing**: Apply tapering functions to mitigate spectral leakage
- **Fisher Matrix Analysis**: Calculate uncertainties with masked data
- **PSD Estimation**: Welch method with gap-segmented time series

## Installation

### Dependencies

Install the following packages in order:

```bash
# Core LISA analysis tools
pip install lisaanalysistools
pip install fastemriwaveforms

# Gap and glitch simulation
pip install lisaglitch
pip install lisagap

# Standard scientific packages
pip install numpy matplotlib scipy tqdm
```

### Environment Setup

For parameter estimation simulations, create a conda environment:

```bash
conda env create -f EMRIs/PE_simulations/config_files/PE_env.yml
conda activate pe_env
```

## Repository Structure

```
├── EMRIs/
│   ├── CV_estimates/          # Cramér-Rao bound calculations
│   └── PE_simulations/        # MCMC parameter estimation
│       ├── config_files/      # Configuration and PSD data
│       └── mcmc_code/         # MCMC implementation
├── toy_problem/
│   ├── long_duration_signal_imputation.ipynb  # Main analysis notebook
│   ├── long_duration_signal.ipynb             # Basic signal analysis
│   ├── notebooks/             # Additional analysis notebooks
│   └── utility_files/         # Core processing functions
└── LICENSE
```

## Usage

### Quick Start

1. **Basic Gap Analysis**: Start with `toy_problem/long_duration_signal_imputation.ipynb`
   - Demonstrates gap generation, noise imputation, and bias analysis
   - Includes PSD estimation from segmented data
   - Shows Fisher matrix calculations with masks

2. **Signal Processing**: Explore `toy_problem/long_duration_signal.ipynb`
   - Basic EMRI waveform generation
   - SNR calculations
   - Noise generation and PSD validation

3. **Parameter Estimation**: Use notebooks in `EMRIs/PE_simulations/mcmc_code/`
   - Full MCMC analysis with realistic gaps
   - Comparison of gapped vs ungapped results

### Core Workflow

```python
import numpy as np
from lisa_gap import GapMaskGenerator
from utility_files.gap_utils import *
from utility_files.psd_utils import *

# 1. Generate EMRI signal
h = waveform(params, time_array)

# 2. Create realistic gaps
gap_mask_gen = GapMaskGenerator(time_array, dt, gap_definitions)
mask = gap_mask_gen.generate_mask(include_planned=True, include_unplanned=True)

# 3. Segment data and estimate PSD
segments = segment_data_by_gaps(signal, mask)
psd_estimates = [welch_psd_estimate(seg) for seg in segments]

# 4. Generate noise for missing segments
imputed_noise = generate_imputed_noise(psd_estimates, gap_lengths)

# 5. Analyze parameter estimation bias
bias = calculate_cv_bias(noise, signal, fisher_matrix)
```

### Gap Types Supported

- **Planned Gaps**: Antenna repointing, PAAM operations
- **Unplanned Gaps**: Platform safe mode, instrument failures
- **Custom Gaps**: User-defined gap patterns

## Key Analysis Results

The repository demonstrates several important findings:

1. **PSD Units Consistency**: Welch estimates require `(1/dt)` scaling factor for proper variance calculations
2. **Smooth Windowing**: Tapering reduces spectral artifacts but increases uncertainty
3. **Noise Imputation**: Segment-based PSD estimation enables realistic gap filling
4. **Bias Distribution**: Multiple noise realizations reveal parameter estimation bias patterns

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gaps_emris,
  author = {Burke, Oliver},
  title = {Gaps_EMRIs: Data Gap Analysis for LISA EMRI Signals},
  url = {https://github.com/OllieBurke/Gaps_EMRIs},
  year = {2025}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Support

For questions or issues, please open a GitHub issue or contact the maintainer.
