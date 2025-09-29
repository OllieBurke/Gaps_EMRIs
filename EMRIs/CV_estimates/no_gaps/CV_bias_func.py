import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    xp = cp
except ImportError:
    cp = None
    cp = np
    CUPY_AVAILABLE = False

# Handle array backend selection
def get_array_module(array):
    """Get the appropriate array module (numpy or cupy) for the given array."""
    if CUPY_AVAILABLE and hasattr(array, '__cuda_array_interface__'):
        return cp
    return np

def inner_product_frequency_domain(a_fft, b_fft, variance_noise_f, n_samples, dt):
    """
    Compute the frequency-domain inner product between two signals.
    
    This function calculates the noise-weighted inner product:
    <a|b> = 4 * dt/N * Re[sum_{f>0} (a*(f) * b(f)) / S_n(f)]
    
    Parameters:
    -----------
    a_fft : array-like
        FFT of first signal, shape (n_channels, n_freq)
    b_fft : array-like  
        FFT of second signal, shape (n_channels, n_freq)
    variance_noise_f : array-like
        Noise variance per frequency bin for weighting, shape (n_channels, n_freq)
        This should be the same variance array used for noise generation
    n_samples : int
        Total number of time-domain samples
    dt : float
        Sampling interval (seconds)
        
    Returns:
    --------
    float : Real-valued inner product
        
    Notes:
    -----
    - Assumes first frequency bin (DC) is excluded from sum
    - Uses the convention for two-sided FFT normalization
    - Compatible with both NumPy and CuPy arrays
    """
    # Get appropriate array module
    xp = get_array_module(a_fft)
    
    # Ensure inputs are arrays
    a_fft = xp.asarray(a_fft)
    b_fft = xp.asarray(b_fft)
    variance_noise_f = xp.asarray(variance_noise_f)
    
    # Input validation
    if a_fft.shape != b_fft.shape:
        raise ValueError(f"Signal FFTs must have same shape: {a_fft.shape} vs {b_fft.shape}")
    if a_fft.shape != variance_noise_f.shape:
        raise ValueError(f"FFT and variance must have same shape: {a_fft.shape} vs {variance_noise_f.shape}")
    
    n_channels = a_fft.shape[0]
    
    # Compute inner product for each channel
    # Skip DC component (index 0) and sum over positive frequencies
    inner_products = []
    for channel in range(n_channels):
        # Element-wise multiplication and noise weighting
        weighted_product = (a_fft[channel][1:] * b_fft[channel][1:].conj()) / variance_noise_f[channel][1:]
        channel_sum = xp.sum(weighted_product)
        inner_products.append(channel_sum)
    
    # Combine channels and apply normalization
    total_sum = xp.sum(xp.asarray(inner_products))
    normalization = 4 * dt / n_samples
    
    return float(xp.real(normalization * total_sum))


def generate_colored_noise(variance_noise_AET, seed=0, window_function=None, return_time_domain=False):
    """
    Generate colored noise with specified noise variance per frequency bin.
    
    Parameters:
    -----------
    variance_noise_f : array-like
        Noise variance per frequency bin. Can be:
        - 1D array (n_freq,) - same variance applied to both channels
        - 2D array (n_channels, n_freq) - different variance per channel
        This is the variance of the noise at each frequency
    seed : int, optional
        Random seed for reproducible noise generation. Default: 0
    window_function : array-like, optional
        Time-domain window to apply (e.g., for gaps). If None, no windowing applied.
        Shape should match time-domain length.
    return_time_domain : bool, optional
        If True, return time-domain noise. If False, return frequency-domain. Default: False
        
    Returns:
    --------
    array : Noise realization
        Shape (2, n_freq) if frequency domain, or (2, n_time) if time domain
        Always returns 2 channels (for TDI A and E channels)
        
    Notes:
    -----
    - Generates complex Gaussian noise with proper normalization for real FFTs
    - DC and Nyquist components are real-valued as required
    - If window is applied, noise is transformed to time domain, windowed, then back to frequency
    - Always generates 2 channels matching your original function behavior
    """
    # Get appropriate array module  
    xp = get_array_module(variance_noise_AET[0])
     
    # Ensure variance is an array and determine shape
    
    N_channels = 2  # Always generate 2 channels like your original
      


    xp.random.seed(seed)
    noise_f_AET_real = [xp.random.normal(0,xp.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
    noise_f_AET_imag = [xp.random.normal(0,xp.sqrt(variance_noise_AET[k])) for k in range(N_channels)]

    # Compute noise in frequency domain
    noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])

    for i in range(N_channels):
        noise_f_AET[i][0] = noise_f_AET[i][0].real + 0j
        noise_f_AET[i][-1] = noise_f_AET[i][-1].real + 0j 

    
    # Apply window function if provided
    if window_function is not None:
        return _apply_window_to_noise(noise_f_AET, window_function, xp)
    
    # Return based on requested domain
    if return_time_domain:
        # Convert to time domain
        noise_time = xp.array([xp.fft.irfft(noise_f_AET[ch]) for ch in range(n_channels)])
        return noise_time
    else:
        return noise_f_AET


def _apply_window_to_noise(noise_freq, window_function, xp):
    """
    Apply time-domain window to frequency-domain noise.
    
    Parameters:
    -----------
    noise_freq : array
        Frequency-domain noise, shape (n_channels, n_freq)
    window_function : array-like
        Time-domain window function
    xp : module
        Array module (numpy or cupy)
        
    Returns:
    --------
    array : Windowed noise in frequency domain, shape (n_channels, n_freq)
    """
    window_function = xp.asarray(window_function)
    n_channels = noise_freq.shape[0]
    
    windowed_noise = []
    
    for channel in range(n_channels):
        # Transform to time domain
        noise_time = xp.fft.irfft(noise_freq[channel], n = len(window_function))
         
        # Apply window
        windowed_time = window_function * noise_time
        
        # Transform back to frequency domain
        windowed_freq = xp.fft.rfft(windowed_time)
        windowed_noise.append(windowed_freq)
    
    return xp.asarray(windowed_noise)

# def zero_pad(data, xp = np):
#     """
#     Inputs: data stream of length N
#     Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
#     """
#     N = len(data)
#     pow_2 = xp.ceil(np.log2(N))
#     return xp.pad(data,(0,int((2**pow_2)-N)),'constant')
def zero_pad(data, xp=np):
    """
    Inputs: data stream of length N (1D) or shape (channels, N) (2D)
    Returns: zero_padded data stream of new length 2^{J} for J ∈ ℕ
    """
    if data.ndim == 1:
        N = len(data)
        pow_2 = xp.ceil(xp.log2(N))  # Use xp.log2 instead of np.log2
        return xp.pad(data, (0, int((2**pow_2) - N)), 'constant')
    elif data.ndim == 2:
        N = data.shape[1]  # Get length from second dimension
        pow_2 = xp.ceil(xp.log2(N))
        pad_width = ((0, 0), (0, int((2**pow_2) - N)))  # No padding on first dim, pad second dim
        return xp.pad(data, pad_width, 'constant')
    else:
        raise ValueError("Data must be 1D or 2D")

def pad_to_length(array, N_t, pad_value=0.0, pad_mode='end', use_gpu=False):
    """
    Pad a gap_window_generator array (or any array) to a desired length N_t.
    
    This function is GPU and CPU agnostic, handling both numpy and cupy arrays
    depending on the use_gpu flag and availability of cupy.
    
    Parameters
    ----------
    array : numpy.ndarray or cupy.ndarray
        Input array to be padded. Can be 1D or 2D.
        For 2D arrays, padding is applied along the last axis (time axis).
    N_t : int
        Target length for the array after padding.
    pad_value : float, optional
        Value to use for padding. Default is 0.0.
        For gap masks, you might want to use 1.0 (valid data) or 0.0 (gap).
    pad_mode : str, optional
        Where to add padding. Options:
        - 'end': Add padding at the end (default)
        - 'start': Add padding at the beginning  
        - 'both': Add padding symmetrically on both sides
    use_gpu : bool, optional
        Whether to use GPU acceleration with cupy. Default is False.
        If True but cupy is not available, falls back to numpy with a warning.
        
    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Padded array of length N_t (or shape [..., N_t] for 2D arrays).
        Returns same type as input (numpy/cupy) unless use_gpu is specified.
        
    Examples
    --------
    >>> import numpy as np
    >>> from gap_window_generator import GapWindowGenerator
    >>> 
    >>> # Create a simple gap mask
    >>> mask = np.ones(100)
    >>> mask[40:60] = 0  # Add a gap
    >>> 
    >>> # Pad to desired length
    >>> padded_mask = pad_to_length(mask, N_t=128, pad_value=1.0)
    >>> print(f"Original length: {len(mask)}, Padded length: {len(padded_mask)}")
    >>> 
    >>> # For 2D arrays (multiple channels)
    >>> mask_2d = np.ones((2, 100))
    >>> mask_2d[:, 40:60] = 0
    >>> padded_mask_2d = pad_to_length(mask_2d, N_t=128)
    >>> print(f"Original shape: {mask_2d.shape}, Padded shape: {padded_mask_2d.shape}")
    >>> 
    >>> # GPU usage (if available)
    >>> padded_mask_gpu = pad_to_length(mask, N_t=128, use_gpu=True)
    
    Raises
    ------
    ValueError
        If N_t is smaller than the current array length and truncation would be needed.
    ValueError
        If pad_mode is not one of the supported options.
    """
    
    # Determine array library to use
    if use_gpu and CUPY_AVAILABLE:
        xp = cp
        # Convert input to cupy if it's numpy
        if isinstance(array, np.ndarray):
            array = cp.asarray(array)
    else:
        xp = np
        if use_gpu and not CUPY_AVAILABLE:
            print("Warning: cupy not available, falling back to numpy")
        # Convert input to numpy if it's cupy
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            array = array.get()
    
    # Validate inputs
    if not isinstance(N_t, int) or N_t <= 0:
        raise ValueError("N_t must be a positive integer")
    
    if pad_mode not in ['end', 'start', 'both']:
        raise ValueError("pad_mode must be one of: 'end', 'start', 'both'")
    
    # Handle 1D and 2D arrays
    array = xp.atleast_1d(array)
    
    if array.ndim == 1:
        current_length = len(array)
    elif array.ndim == 2:
        current_length = array.shape[1]  # Assume time is last axis
    else:
        raise ValueError("Only 1D and 2D arrays are supported")
    
    # Check if padding is needed
    if current_length == N_t:
        return array
    elif current_length > N_t:
        raise ValueError(f"Array length {current_length} is greater than target N_t={N_t}. "
                        f"Truncation not supported. Consider using array[:N_t] if truncation is desired.")
    
    # Calculate padding needed
    pad_needed = N_t - current_length
    
    # Determine padding distribution
    if pad_mode == 'end':
        pad_left = 0
        pad_right = pad_needed
    elif pad_mode == 'start':
        pad_left = pad_needed
        pad_right = 0
    elif pad_mode == 'both':
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
    
    # Create padding arrays
    if array.ndim == 1:
        left_pad = xp.full(pad_left, pad_value, dtype=array.dtype)
        right_pad = xp.full(pad_right, pad_value, dtype=array.dtype)
        padded_array = xp.concatenate([left_pad, array, right_pad])
    else:  # 2D array
        n_channels = array.shape[0]
        left_pad = xp.full((n_channels, pad_left), pad_value, dtype=array.dtype)
        right_pad = xp.full((n_channels, pad_right), pad_value, dtype=array.dtype)
        padded_array = xp.concatenate([left_pad, array, right_pad], axis=1)
    
    return padded_array
# def inner_product_f(a_fft, b_fft, PSD, N, dt):
#     return xp.real(xp.sum(xp.asarray([(4 * dt / N) * xp.sum((a_fft[j][1:] * b_fft[j][1:].conj())/PSD[j][1:]) for j in range(2)])))

# def noise_gen(variance_noise_f, seed_number = 0, window = None):
#     np.random.seed(seed_number)
#     noise_f_I = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j * xp.random.normal(0,xp.sqrt(variance_noise_f))
#     noise_f_I[0] = xp.sqrt(2)*noise_f_I[0].real
#     noise_f_I[-1] = xp.sqrt(2)*noise_f_I[-1].real

#     noise_f_II = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j * xp.random.normal(0,xp.sqrt(variance_noise_f))
#     noise_f_II[0] = xp.sqrt(2)*noise_f_II[0].real
#     noise_f_II[-1] = xp.sqrt(2)*noise_f_II[-1].real
#     if window is None:
#         return xp.asarray([noise_f_I, noise_f_II])
#     elif window is not None:
#         noise_t_I = xp.fft.irfft(noise_f_I)
#         noise_t_I_window = gap_window_func * noise_t_I 
#         noise_f_I_window = xp.fft.rfft(noise_t_I_window)

#         noise_t_II = xp.fft.irfft(noise_f_II)
#         noise_t_II_window = gap_window_func * noise_t_II
#         noise_f_II_window = xp.fft.rfft(noise_t_II_window)
#         return xp.asarray([noise_f_I_window, noise_f_II_window])