import h5py
import numpy as np
import os
from pathlib import Path

# Handle CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

def to_numpy(array):
    """
    Convert array to numpy, handling both numpy and cupy arrays.
    
    Parameters:
    -----------
    array : np.ndarray or cp.ndarray
        Input array (could be on CPU or GPU)
        
    Returns:
    --------
    np.ndarray : Array converted to numpy (on CPU)
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)

def get_array_info(array):
    """
    Get array type and device information for logging.
    
    Parameters:
    -----------
    array : np.ndarray or cp.ndarray
        Input array
        
    Returns:
    --------
    str : Description of array type and location
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        device_id = array.device.id if hasattr(array.device, 'id') else 'unknown'
        return f"CuPy array on GPU {device_id}"
    return "NumPy array on CPU"

def _save_nested_dict_safely(group, data_dict):
    """
    Safely save nested dictionary to HDF5 group, handling complex data types.
    
    Parameters:
    -----------
    group : h5py.Group
        HDF5 group to save data to
    data_dict : dict
        Dictionary to save
    """
    for key, value in data_dict.items():
        try:
            if isinstance(value, str):
                group.attrs[key] = value.encode('utf-8')
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                nested_group = group.create_group(key)
                _save_nested_dict_safely(nested_group, value)
            elif isinstance(value, (list, tuple)):
                # Try to save as array if all elements are the same type
                try:
                    arr = np.array(value)
                    if arr.dtype.kind in ['i', 'f', 'b']:  # int, float, bool
                        group.create_dataset(key, data=arr)
                    else:
                        # Convert to string if mixed types
                        group.attrs[key] = str(value).encode('utf-8')
                except:
                    group.attrs[key] = str(value).encode('utf-8')
            elif isinstance(value, (int, float, bool, np.integer, np.floating)):
                group.attrs[key] = value
            elif hasattr(value, '__array__'):
                # NumPy arrays or array-like objects
                group.create_dataset(key, data=np.asarray(value))
            else:
                # For any other complex types, convert to string
                group.attrs[key] = str(value).encode('utf-8')
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not save {key} to HDF5: {e}")
            print(f"  Type: {type(value)}, attempting string conversion...")
            try:
                group.attrs[f"{key}_str"] = str(value).encode('utf-8')
            except Exception as e2:
                print(f"  String conversion also failed: {e2}")

def save_fisher_results_to_hdf5(filename, param_cov, derivs, param_names, 
                                additional_metadata=None, 
                                base_dir="/work/scratch/data/burkeol/Gaps_EMRIs/Fisher_Matrices",
                                emri_params=None, T=None, dt=None, gap_window_array=None,
                                gap_info=None, monte_carlo_results=None):
    """
    Save Fisher matrix results to HDF5 file with organized structure.
    
    Parameters:
    -----------
    filename : str
        Output HDF5 filename (should end with .h5 or .hdf5)
        Can be just the filename or a full path. If just filename,
        it will be saved to the default base_dir.
    param_cov : np.ndarray
        Inverse Fisher matrix (covariance matrix) of shape (12, 12)
    derivs : list
        List of derivative arrays, each of shape (2, N_samples)
    param_names : list
        List of parameter names corresponding to derivatives
    additional_metadata : dict, optional
        Additional metadata to store in the file
    base_dir : str, optional
        Default directory to save files. Defaults to 
        "/work/scratch/data/burkeol/Gaps_EMRIs/Fisher_Matrices"
    emri_params : dict, optional
        Dictionary of EMRI parameters used for the computation
    T : float, optional
        Total observation time (years)
    dt : float, optional
        Sampling interval (seconds)
    gap_window_array : array-like, optional
        Window function array (same length as time series)
    gap_info : dict, optional
        Information about gap generation (seeds, settings, etc.)
    monte_carlo_results : dict, optional
        Monte Carlo noise realization results with keys:
        - 'noise_MLE_vec': List/array of parameter estimates per noise realization
        - 'seeds': List/array of seed numbers used for each realization
        - 'True_Cov_Matrix': Empirical covariance matrix from noise realizations
        - 'n_realizations': Number of Monte Carlo realizations
        
    Returns:
    --------
    str : Full path to the saved file
    """
    
    # Handle path construction
    if os.path.isabs(filename):
        # If filename is already an absolute path, use it as-is
        full_path = filename
    else:
        # Create the full path using base_dir
        base_path = Path(base_dir)
        full_path = base_path / filename
    
    # Create directory if it doesn't exist
    full_path = Path(full_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Fisher matrix results to: {full_path}")
    print(f"Input data types:")
    print(f"  - param_cov: {get_array_info(param_cov)}")
    if derivs:
        print(f"  - derivs[0]: {get_array_info(derivs[0])}")
    
    print(f"=== WINDOW FUNCTION DEBUG ===")
    print(f"gap_window_array is None: {gap_window_array is None}")
    if gap_window_array is not None:
        print(f"  - gap_window_array: {get_array_info(gap_window_array)}")
        print(f"  - shape: {gap_window_array.shape}")
        print(f"  - min/max: {gap_window_array.min()}/{gap_window_array.max()}")
    else:
        print("  - gap_window_array: None (no window will be saved)")
    
    # Convert arrays to numpy for HDF5 compatibility
    param_cov_cpu = to_numpy(param_cov)
    derivs_cpu = [to_numpy(deriv) for deriv in derivs]
    gap_window_cpu = to_numpy(gap_window_array) if gap_window_array is not None else None
    
    print(f"After conversion to CPU:")
    print(f"gap_window_cpu is None: {gap_window_cpu is None}")
    if gap_window_cpu is not None:
        print(f"gap_window_cpu shape: {gap_window_cpu.shape}")
    
    with h5py.File(full_path, 'w') as f:
        # Create main groups for organization
        fisher_group = f.create_group('fisher_analysis')
        derivatives_group = f.create_group('derivatives')
        metadata_group = f.create_group('metadata')
        simulation_group = f.create_group('simulation_parameters')
        gaps_group = f.create_group('gap_analysis')
        
        # Add Monte Carlo group if results provided
        if monte_carlo_results is not None:
            monte_carlo_group = f.create_group('monte_carlo_analysis')
        
        # Save inverse Fisher matrix (covariance matrix)
        fisher_group.create_dataset('inverse_fisher_matrix', data=param_cov_cpu)
        
        # Save parameter names as metadata
        metadata_group.create_dataset('parameter_names', 
                                     data=[name.encode('utf-8') for name in param_names])
        
        # Save EMRI parameters if provided
        if emri_params is not None:
            emri_group = simulation_group.create_group('emri_parameters')
            
            # Save parameter order to preserve dictionary order
            param_order = list(emri_params.keys())
            emri_group.create_dataset('parameter_order', 
                                    data=[name.encode('utf-8') for name in param_order])
            
            # Save parameter values
            for param_name, param_value in emri_params.items():
                if isinstance(param_value, str):
                    emri_group.attrs[param_name] = param_value.encode('utf-8')
                else:
                    emri_group.attrs[param_name] = param_value
        
        # Save observation parameters if provided
        if T is not None or dt is not None:
            obs_group = simulation_group.create_group('observation_parameters')
            if T is not None:
                obs_group.attrs['observation_time_years'] = T
            if dt is not None:
                obs_group.attrs['sampling_interval_seconds'] = dt
                if T is not None:
                    # Calculate total samples for convenience
                    total_samples = int(T * 365.25 * 24 * 3600 / dt)
                    obs_group.attrs['total_samples'] = total_samples
                    obs_group.attrs['nyquist_frequency_hz'] = 0.5 / dt
        
        # Save gap window information if provided
        if gap_window_cpu is not None:
            # Save the window function array
            gaps_group.create_dataset('window_function', data=gap_window_cpu)
            
            # Calculate gap statistics
            gap_fraction = 1.0 - np.mean(gap_window_cpu)
            gaps_group.attrs['gap_fraction'] = gap_fraction
            gaps_group.attrs['window_length'] = len(gap_window_cpu)
            gaps_group.attrs['window_min'] = np.min(gap_window_cpu)
            gaps_group.attrs['window_max'] = np.max(gap_window_cpu)
            gaps_group.attrs['effective_observation_time'] = gap_fraction * T if T is not None else gap_fraction
            
            # Save gap generation info if provided
            if gap_info is not None:
                gap_settings = gap_settings_group = gaps_group.create_group('gap_settings')
                for key, value in gap_info.items():
                    try:
                        if isinstance(value, str):
                            gap_settings.attrs[key] = value.encode('utf-8')
                        elif isinstance(value, dict):
                            # Handle nested dictionaries (like gap_definitions)
                            nested_group = gap_settings.create_group(key)
                            _save_nested_dict_safely(nested_group, value)
                        elif isinstance(value, (list, tuple)):
                            # Convert lists/tuples to string representation for HDF5
                            gap_settings.attrs[key] = str(value).encode('utf-8')
                        elif isinstance(value, (int, float, bool, np.integer, np.floating)):
                            gap_settings.attrs[key] = value
                        else:
                            # For any other complex types, convert to string
                            gap_settings.attrs[key] = str(value).encode('utf-8')
                    except (TypeError, ValueError) as e:
                        print(f"Warning: Could not save gap_info['{key}'] to HDF5: {e}")
                        print(f"  Type: {type(value)}, Value: {value}")
                        # Save as string representation as fallback
                        gap_settings.attrs[f"{key}_str"] = str(value).encode('utf-8')
        else:
            # No gaps applied
            gaps_group.attrs['gap_fraction'] = 0.0
            gaps_group.attrs['gaps_applied'] = False
        
        # ========== SAVE MONTE CARLO RESULTS ==========
        if monte_carlo_results is not None:
            print("Saving Monte Carlo analysis results...")
            
            # Convert to CPU arrays for HDF5 saving
            noise_MLE_vec = to_numpy(monte_carlo_results['noise_MLE_vec'])
            True_Cov_Matrix = to_numpy(monte_carlo_results['True_Cov_Matrix'])
            seeds = monte_carlo_results.get('seeds', list(range(len(noise_MLE_vec))))
            
            # Save the main Monte Carlo data
            monte_carlo_group.create_dataset('noise_MLE_realizations', data=noise_MLE_vec)
            monte_carlo_group.create_dataset('empirical_covariance_matrix', data=True_Cov_Matrix)
            monte_carlo_group.create_dataset('random_seeds', data=seeds)
            
            # Save Monte Carlo metadata
            n_realizations = len(noise_MLE_vec)
            n_params = len(param_names)
            
            monte_carlo_group.attrs['n_realizations'] = n_realizations
            monte_carlo_group.attrs['n_parameters'] = n_params
            monte_carlo_group.attrs['data_shape_description'] = f'noise_MLE_realizations: ({n_realizations}, {n_params})'
            
            # Calculate and save Monte Carlo statistics
            if n_realizations > 1:
                param_means = np.mean(noise_MLE_vec, axis=0)
                param_stds = np.std(noise_MLE_vec, axis=0)
                
                monte_carlo_group.create_dataset('parameter_means', data=param_means)
                monte_carlo_group.create_dataset('parameter_std_devs', data=param_stds)
                
                # Compare Fisher vs Monte Carlo uncertainties
                fisher_uncertainties = np.sqrt(np.diag(param_cov_cpu))
                monte_carlo_uncertainties = param_stds
                
                monte_carlo_group.create_dataset('fisher_vs_monte_carlo_ratio', 
                                                data=fisher_uncertainties / monte_carlo_uncertainties)
                
                monte_carlo_group.attrs['monte_carlo_completed'] = True
            else:
                monte_carlo_group.attrs['monte_carlo_completed'] = False
                monte_carlo_group.attrs['note'] = 'Insufficient realizations for statistics'
        
        # Save derivatives with descriptive names
        for i, (param_name, deriv) in enumerate(zip(param_names, derivs_cpu)):
            # Create dataset name like "derivative_m1", "derivative_m2", etc.
            dataset_name = f"derivative_{param_name}"
            derivatives_group.create_dataset(dataset_name, data=deriv)
            
            # Add attributes to each derivative dataset
            derivatives_group[dataset_name].attrs['parameter_index'] = i
            derivatives_group[dataset_name].attrs['parameter_name'] = param_name
            derivatives_group[dataset_name].attrs['shape_description'] = '(2, N_samples) - [real, imag] components'
        
        # Add file path to metadata
        metadata_group.attrs['file_path'] = str(full_path)
        metadata_group.attrs['base_directory'] = base_dir
        # Add general metadata
        metadata_group.attrs['fisher_matrix_shape'] = param_cov.shape
        metadata_group.attrs['n_samples'] = derivs[0].shape[1] if derivs else 0
        metadata_group.attrs['creation_time'] = str(np.datetime64('now')).encode('utf-8')
        metadata_group.attrs['cupy_available'] = CUPY_AVAILABLE
        metadata_group.attrs['original_param_cov_type'] = get_array_info(param_cov).encode('utf-8')
        if derivs:
            metadata_group.attrs['original_derivs_type'] = get_array_info(derivs[0]).encode('utf-8')
        
        # Add any additional metadata
        if additional_metadata:
            for key, value in additional_metadata.items():
                if isinstance(value, str):
                    metadata_group.attrs[key] = value.encode('utf-8')
                else:
                    metadata_group.attrs[key] = value
    
    print(f"Successfully saved Fisher matrix with {len(param_names)} parameters")
    print(f"File size: {full_path.stat().st_size / (1024**2):.2f} MB")
    return str(full_path)

def load_fisher_results_from_hdf5(filename, return_as_cupy=False):
    """
    Load Fisher matrix results from HDF5 file.
    
    Parameters:
    -----------
    filename : str
        Input HDF5 filename
    return_as_cupy : bool, optional
        If True and CuPy is available, return arrays as CuPy arrays.
        If False, always return NumPy arrays.
        
    Returns:
    --------
    dict : Dictionary containing loaded data with keys:
        - 'inverse_fisher_matrix': Covariance matrix
        - 'derivatives': List of derivative arrays  
        - 'parameter_names': List of parameter names
        - 'metadata': File metadata
        - 'emri_parameters': EMRI source parameters (if saved)
        - 'observation_parameters': T, dt, etc. (if saved)
        - 'gap_analysis': Gap window and statistics (if saved)
    """
    
    results = {}
    
    with h5py.File(filename, 'r') as f:
        # Load inverse Fisher matrix
        fisher_matrix = f['fisher_analysis/inverse_fisher_matrix'][:]
        if return_as_cupy and CUPY_AVAILABLE:
            results['inverse_fisher_matrix'] = cp.asarray(fisher_matrix)
        else:
            results['inverse_fisher_matrix'] = fisher_matrix
        
        # Load parameter names
        param_names_bytes = f['metadata/parameter_names'][:]
        results['parameter_names'] = [name.decode('utf-8') for name in param_names_bytes]
        
        # Load derivatives
        results['derivatives'] = []
        derivatives_group = f['derivatives']
        
        # Load derivatives in the correct order based on parameter names
        for param_name in results['parameter_names']:
            dataset_name = f"derivative_{param_name}"
            if dataset_name in derivatives_group:
                deriv_data = derivatives_group[dataset_name][:]
                if return_as_cupy and CUPY_AVAILABLE:
                    results['derivatives'].append(cp.asarray(deriv_data))
                else:
                    results['derivatives'].append(deriv_data)
        
        # Load metadata
        results['metadata'] = dict(f['metadata'].attrs)
        
        # Decode string metadata
        for key, value in results['metadata'].items():
            if isinstance(value, bytes):
                results['metadata'][key] = value.decode('utf-8')
        
        # ========== LOAD EMRI PARAMETERS ==========
        if 'simulation_parameters/emri_parameters' in f:
            emri_group = f['simulation_parameters/emri_parameters']
            
            # Check if parameter order was saved (for newer files)
            if 'parameter_order' in emri_group:
                # Use saved order to reconstruct ordered dictionary
                param_order = [name.decode('utf-8') for name in emri_group['parameter_order'][:]]
                results['emri_parameters'] = {}
                
                for param_name in param_order:
                    if param_name in emri_group.attrs:
                        value = emri_group.attrs[param_name]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        results['emri_parameters'][param_name] = value
            else:
                # Fallback for older files without saved order
                results['emri_parameters'] = {}
                for key in emri_group.attrs.keys():
                    value = emri_group.attrs[key]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    results['emri_parameters'][key] = value
        
        # ========== LOAD OBSERVATION PARAMETERS ==========
        if 'simulation_parameters/observation_parameters' in f:
            obs_group = f['simulation_parameters/observation_parameters']
            results['observation_parameters'] = {}
            for key in obs_group.attrs.keys():
                results['observation_parameters'][key] = obs_group.attrs[key]
        
        # ========== LOAD GAP ANALYSIS ==========
        if 'gap_analysis' in f:
            gaps_group = f['gap_analysis']
            results['gap_analysis'] = {}
            
            # Load gap statistics (attributes)
            for key in gaps_group.attrs.keys():
                results['gap_analysis'][key] = gaps_group.attrs[key]
            
            # Load window function if present
            if 'window_function' in gaps_group:
                window_data = gaps_group['window_function'][:]
                if return_as_cupy and CUPY_AVAILABLE:
                    results['gap_analysis']['window_function'] = cp.asarray(window_data)
                else:
                    results['gap_analysis']['window_function'] = window_data
            
        # ========== LOAD MONTE CARLO ANALYSIS ==========
        if 'monte_carlo_analysis' in f:
            monte_carlo_group = f['monte_carlo_analysis']
            results['monte_carlo_analysis'] = {}
            
            # Load main datasets
            if 'noise_MLE_realizations' in monte_carlo_group:
                mle_data = monte_carlo_group['noise_MLE_realizations'][:]
                if return_as_cupy and CUPY_AVAILABLE:
                    results['monte_carlo_analysis']['noise_MLE_realizations'] = cp.asarray(mle_data)
                else:
                    results['monte_carlo_analysis']['noise_MLE_realizations'] = mle_data
            
            if 'empirical_covariance_matrix' in monte_carlo_group:
                cov_data = monte_carlo_group['empirical_covariance_matrix'][:]
                if return_as_cupy and CUPY_AVAILABLE:
                    results['monte_carlo_analysis']['empirical_covariance_matrix'] = cp.asarray(cov_data)
                else:
                    results['monte_carlo_analysis']['empirical_covariance_matrix'] = cov_data
            
            if 'random_seeds' in monte_carlo_group:
                results['monte_carlo_analysis']['random_seeds'] = monte_carlo_group['random_seeds'][:]
            
            # Load statistics if available
            if 'parameter_means' in monte_carlo_group:
                results['monte_carlo_analysis']['parameter_means'] = monte_carlo_group['parameter_means'][:]
            
            if 'parameter_std_devs' in monte_carlo_group:
                results['monte_carlo_analysis']['parameter_std_devs'] = monte_carlo_group['parameter_std_devs'][:]
                
            if 'fisher_vs_monte_carlo_ratio' in monte_carlo_group:
                results['monte_carlo_analysis']['fisher_vs_monte_carlo_ratio'] = monte_carlo_group['fisher_vs_monte_carlo_ratio'][:]
            
            # Load gap settings if present
            if 'gap_settings' in gaps_group:
                results['gap_analysis']['gap_settings'] = _load_nested_group(gaps_group['gap_settings'])
        
        # ========== LOAD MONTE CARLO ANALYSIS ==========
        if 'monte_carlo_analysis' in f:
            monte_carlo_group = f['monte_carlo_analysis']
            results['monte_carlo_analysis'] = {}
            
            # Load main datasets
            if 'noise_MLE_realizations' in monte_carlo_group:
                mle_data = monte_carlo_group['noise_MLE_realizations'][:]
                if return_as_cupy and CUPY_AVAILABLE:
                    results['monte_carlo_analysis']['noise_MLE_realizations'] = cp.asarray(mle_data)
                else:
                    results['monte_carlo_analysis']['noise_MLE_realizations'] = mle_data
            
            if 'empirical_covariance_matrix' in monte_carlo_group:
                cov_data = monte_carlo_group['empirical_covariance_matrix'][:]
                if return_as_cupy and CUPY_AVAILABLE:
                    results['monte_carlo_analysis']['empirical_covariance_matrix'] = cp.asarray(cov_data)
                else:
                    results['monte_carlo_analysis']['empirical_covariance_matrix'] = cov_data
            
            if 'random_seeds' in monte_carlo_group:
                results['monte_carlo_analysis']['random_seeds'] = monte_carlo_group['random_seeds'][:]
            
            # Load statistics if available
            if 'parameter_means' in monte_carlo_group:
                results['monte_carlo_analysis']['parameter_means'] = monte_carlo_group['parameter_means'][:]
            
            if 'parameter_std_devs' in monte_carlo_group:
                results['monte_carlo_analysis']['parameter_std_devs'] = monte_carlo_group['parameter_std_devs'][:]
                
            if 'fisher_vs_monte_carlo_ratio' in monte_carlo_group:
                results['monte_carlo_analysis']['fisher_vs_monte_carlo_ratio'] = monte_carlo_group['fisher_vs_monte_carlo_ratio'][:]
            
            # Load attributes
            for key in monte_carlo_group.attrs.keys():
                results['monte_carlo_analysis'][key] = monte_carlo_group.attrs[key]
    
    # Print loading information
    array_type = "CuPy" if return_as_cupy and CUPY_AVAILABLE else "NumPy"
    print(f"Loaded Fisher matrix data as {array_type} arrays")
    print(f"Parameters: {len(results['parameter_names'])}")
    print(f"Fisher matrix shape: {results['inverse_fisher_matrix'].shape}")
    
    # Print what additional data was found
    additional_data = []
    if 'emri_parameters' in results:
        additional_data.append(f"EMRI parameters ({len(results['emri_parameters'])} params)")
    if 'observation_parameters' in results:
        additional_data.append("observation parameters")
    if 'gap_analysis' in results:
        gap_data = results['gap_analysis']
        window_info = f"window function ({gap_data['window_function'].shape})" if 'window_function' in gap_data else "gap statistics only"
        additional_data.append(f"gap analysis ({window_info})")
    
    if 'monte_carlo_analysis' in results:
        mc_data = results['monte_carlo_analysis']
        n_realizations = mc_data.get('n_realizations', 0)
        additional_data.append(f"Monte Carlo analysis ({n_realizations} realizations)")
    
    if additional_data:
        print(f"Additional data loaded: {', '.join(additional_data)}")
    
    return results

def _load_nested_group(group):
    """
    Recursively load nested HDF5 group structure into dictionaries.
    
    Parameters:
    -----------
    group : h5py.Group
        HDF5 group to load
        
    Returns:
    --------
    dict : Nested dictionary with group contents
    """
    result = {}
    
    # Load attributes
    for key in group.attrs.keys():
        value = group.attrs[key]
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        result[key] = value
    
    # Load subgroups and datasets
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            # Recursively load subgroups
            result[key] = _load_nested_group(item)
        elif isinstance(item, h5py.Dataset):
            # Load datasets
            result[key] = item[:]
    
    return result

# Utility function to add Monte Carlo results to existing file
def add_monte_carlo_to_existing_file(filename, monte_carlo_results):
    """Add Monte Carlo results to an existing Fisher matrix file."""
    
    # Load existing data
    data = load_fisher_results_from_hdf5(filename)
    
    # Add Monte Carlo results
    data['monte_carlo_results'] = monte_carlo_results
    
    # Re-save with Monte Carlo data
    save_fisher_results_to_hdf5(
        filename.replace('.h5', '.h5'),  # New filename
        data['inverse_fisher_matrix'],
        data['derivatives'], 
        data['parameter_names'],
        emri_params=data.get('emri_parameters'),
        T=data.get('observation_parameters', {}).get('observation_time_years'),
        dt=data.get('observation_parameters', {}).get('sampling_interval_seconds'),
        gap_window_array=data.get('gap_analysis', {}).get('window_function'),
        gap_info=data.get('gap_analysis', {}).get('gap_settings'),
        monte_carlo_results=monte_carlo_results
    )


# Usage example for your specific case:
if __name__ == "__main__":
    # Your parameter names
    param_names = [
        "m1", "m2", "a", "p0", "e0", "dist", 
        "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0"
    ]
    
    # Assuming you have param_cov and derivs from StableEMRIFisher
    # param_cov: (12, 12) inverse Fisher matrix
    # derivs: list of 12 arrays, each (2, 12623259)
    
    # Example usage:
    # Simple filename with gap window
    # save_fisher_results_to_hdf5(
    #     filename='my_emri_fisher.h5',
    #     param_cov=param_cov,
    #     derivs=derivs,
    #     param_names=param_names,
    #     emri_params=emri_params,
    #     T=T,
    #     dt=dt,
    #     gap_window_array=gap_window_array,
    #     gap_info={
    #         'planned_seed': 1234,
    #         'unplanned_seed': 4321,
    #         'include_planned': True,
    #         'include_unplanned': False,
    #         'apply_tapering': False,
    #         'gap_definitions': gap_definitions,
    #         'taper_definitions': taper_defs
    #     }
    # )
    # Result: /work/scratch/data/burkeol/Gaps_EMRIs/Fisher_Matrices/my_emri_fisher.h5
    
    # Custom subdirectory within default
    # save_fisher_results_to_hdf5(
    #     filename='run_001/fisher_results.h5',
    #     param_cov=param_cov,
    #     derivs=derivs,
    #     param_names=param_names,
    #     emri_params=emri_params,
    #     T=T,
    #     dt=dt
    # )
    # Result: /work/scratch/data/burkeol/Gaps_EMRIs/Fisher_Matrices/run_001/fisher_results.h5
    
    # Full custom path (overrides default directory)
    # save_fisher_results_to_hdf5(
    #     filename='/custom/path/fisher_results.h5',
    #     param_cov=param_cov,
    #     derivs=derivs,
    #     param_names=param_names,
    #     emri_params=emri_params,
    #     T=T,
    #     dt=dt
    # )
    # Result: /custom/path/fisher_results.h5
    
    # Additional utility functions for CuPy/NumPy compatibility
    
def check_array_compatibility(arrays, operation_name="operation"):
    """
    Check if arrays are compatible for operations and warn about mixed types.
    
    Parameters:
    -----------
    arrays : list
        List of arrays to check
    operation_name : str
        Name of the operation for error messages
    """
    has_cupy = any(CUPY_AVAILABLE and isinstance(arr, cp.ndarray) for arr in arrays)
    has_numpy = any(isinstance(arr, np.ndarray) for arr in arrays)
    
    if has_cupy and has_numpy:
        print(f"Warning: Mixed array types detected for {operation_name}. "
              f"Consider converting all arrays to the same type for optimal performance.")
    
    return has_cupy, has_numpy

def convert_arrays_to_same_type(arrays, prefer_cupy=True):
    """
    Convert all arrays to the same type (either all CuPy or all NumPy).
    
    Parameters:
    -----------
    arrays : list
        List of arrays to convert
    prefer_cupy : bool
        If True and CuPy is available, convert to CuPy. Otherwise convert to NumPy.
        
    Returns:
    --------
    list : List of converted arrays
    """
    if prefer_cupy and CUPY_AVAILABLE:
        return [cp.asarray(arr) for arr in arrays]
    else:
        return [to_numpy(arr) for arr in arrays]