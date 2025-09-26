import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
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


def generate_colored_noise(variance_noise_f, seed=0, window_function=None, return_time_domain=False):
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
    xp = get_array_module(variance_noise_f)
    
    # Set random seed (must use numpy for consistency across backends)
    np.random.seed(seed)
    
    # Ensure variance is an array and determine shape
    variance_noise_f = xp.asarray(variance_noise_f)
    
    if variance_noise_f.ndim == 1:
        # 1D variance - use same variance for both channels (like your original)
        n_freq = len(variance_noise_f)
        variance_both_channels = xp.array([variance_noise_f, variance_noise_f])  # Shape: (2, n_freq)
    elif variance_noise_f.ndim == 2:
        # 2D variance - already has channel dimension
        variance_both_channels = variance_noise_f
        n_freq = variance_noise_f.shape[1]
    else:
        raise ValueError(f"variance_noise_f must be 1D or 2D array, got {variance_noise_f.ndim}D")
    
    n_channels = 2  # Always generate 2 channels like your original
    
    # Calculate standard deviation from variance
    std_dev = xp.sqrt(variance_both_channels)
    
    # Generate complex Gaussian noise for each channel
    noise_channels = []
    
    for channel in range(n_channels):
        # Generate complex noise - use numpy random then convert to maintain consistency
        real_part = np.random.normal(0, 1, n_freq)
        imag_part = np.random.normal(0, 1, n_freq)
        
        # Convert to appropriate backend and scale by std_dev
        if xp is cp:
            real_part = cp.asarray(real_part)
            imag_part = cp.asarray(imag_part)
        
        noise_freq = (real_part + 1j * imag_part) * std_dev[channel]
        
        # Handle special cases for real FFT:
        # DC component (f=0) must be real
        noise_freq[0] = xp.sqrt(2) * noise_freq[0].real + 0j
        
        # Nyquist component (f=f_max) must be real  
        noise_freq[-1] = xp.sqrt(2) * noise_freq[-1].real + 0j
        
        noise_channels.append(noise_freq)
    
    noise_array = xp.asarray(noise_channels)
    
    # Apply window function if provided
    if window_function is not None:
        return _apply_window_to_noise(noise_array, window_function, xp)
    
    # Return based on requested domain
    if return_time_domain:
        # Convert to time domain
        noise_time = xp.array([xp.fft.irfft(noise_array[ch]) for ch in range(n_channels)])
        return noise_time
    else:
        return noise_array


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
        
        # Ensure window and noise have compatible shapes
        if len(window_function) != len(noise_time):
            min_length = min(len(window_function), len(noise_time))
            window_trimmed = window_function[:min_length]
            noise_trimmed = noise_time[:min_length]
            print(f"Warning: Window length ({len(window_function)}) doesn't match "
                  f"noise length ({len(noise_time)}). Trimming to {min_length}.")
        else:
            window_trimmed = window_function
            noise_trimmed = noise_time
        
        # Apply window
        windowed_time = window_trimmed * noise_trimmed
        
        # Transform back to frequency domain
        windowed_freq = xp.fft.rfft(windowed_time)
        windowed_noise.append(windowed_freq)
    
    return xp.asarray(windowed_noise)

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