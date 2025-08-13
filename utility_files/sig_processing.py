import numpy as np

def waveform(params, t, mask = None):
    """
    Generate a sinusoidal waveform with linear frequency evolution.
    
    Args:
        params: List/array containing [amplitude, frequency, frequency_derivative]
        t: Time vector
        
    Returns:
        numpy.ndarray: Waveform signal a * sin(2π(f*t + 0.5*fdot*t²))
    """
    amplitude, frequency, frequency_derivative = params
    phase = 2 * np.pi * (frequency * t + 0.5 * frequency_derivative * t**2)
    if mask is not None:
        amplitude *= mask
    return amplitude * np.sin(phase)

def zero_pad(data):
    """
    Zero-pad data to the next power of 2 length.
    
    Args:
        data: Input data array of length N
        
    Returns:
        numpy.ndarray: Zero-padded data array of length 2^J where J is the 
                      smallest integer such that 2^J >= N
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    """
    Compute stationary noise-weighted inner product
    Inputs: sig1_f and sig2_f are signals in frequency domain 
            N_t length of padded signal in time domain
            delta_t sampling interval
            PSD Power spectral density

    Returns: Noise weighted inner product 
    """
    prefac = 4*delta_t / N_t
    sig2_f_conj = np.conjugate(sig2_f)
    return prefac * np.real(np.sum((sig1_f * sig2_f_conj)/PSD))


def deriv_waveform(params,t, phi, mask = None):
    """
    It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR. 
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important 
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    a = params[0]
    f = params[1]
    fdot = params[2]
    if mask is not None:
        a *= mask

    return (a *(np.sin((2*np.pi)*(f*t + 0.5*fdot * t**2) + phi) ))

def likelihood_function_chunk(params, data_stream_seg, delta_t, N_t, psd):
    """
    Compute the likelihood function for a given set of parameters and data segment.
    
    Args:
        params: List/array containing [amplitude, frequency, frequency_derivative]
        data_stream_seg: Data segment to compare against the model
        delta_t: Sampling interval
        N_t: Length of the padded signal in time domain
        psd: Power spectral density
    
    Returns:
        float: Likelihood value
    """
    # Compute the waveform in frequency domain
    waveform_f = np.fft.rfft(zero_pad(waveform(params, np.arange(0, N_t * delta_t, delta_t))))[1:]
    
    # Compute the inner product with the data segment
    inner_prod_value = inner_prod(waveform_f, data_stream_seg, N_t, delta_t, psd)
    
    return inner_prod_value
def fisher_matrix(true_params, sim_t, delta_t, N_t, psd, mask = None):
    # Compute FM
    N_params = len(true_params)

    exact_deriv_a = (true_params[0])** -1  * deriv_waveform(true_params, sim_t, 0, mask)
    exact_deriv_a_pad = zero_pad(exact_deriv_a)
    deriv_a_fft = np.fft.rfft(exact_deriv_a_pad)

    exact_deriv_f = (2*np.pi*sim_t) * deriv_waveform(true_params, sim_t, np.pi/2, mask)
    exact_deriv_f_pad = zero_pad(exact_deriv_f)
    deriv_f_fft = np.fft.rfft(exact_deriv_f_pad)

    exact_deriv_fdot = (0.5) * (2*np.pi * sim_t**2) * deriv_waveform(true_params, sim_t, np.pi/2, mask)
    exact_deriv_fdot_pad = zero_pad(exact_deriv_fdot)
    deriv_fdot_fft = np.fft.rfft(exact_deriv_fdot_pad)

    deriv_vec = [deriv_a_fft, deriv_f_fft, deriv_fdot_fft]
    Fisher_Matrix = np.eye(N_params)
    for i in range(N_params):
        for j in range(N_params):
            Fisher_Matrix[i,j] = inner_prod(deriv_vec[i],deriv_vec[j],N_t,delta_t,psd)

    Cov_Matrix = np.linalg.inv(Fisher_Matrix)
    return deriv_vec, Cov_Matrix

def CV_bias(Cov_Matrix, deriv_vec, noise_f, true_params, delta_t, N_t, psd):

    # deriv_vec, Cov_Matrix = fisher_matrix(true_params, sim_t, delta_t, N_t, psd, mask)

    b_vec = [inner_prod(noise_f, deriv_vec[j], N_t, delta_t, psd) for j in range(len(deriv_vec))]

    bias_vec = Cov_Matrix @ b_vec + np.array(true_params)


    return bias_vec