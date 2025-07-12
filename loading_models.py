import pickle
import pandas as pd
import torch
import numpy as np
import h5py

# Pickle

def save_training_data_pickle(training_data, filepath):
    """Save training data as a Pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(training_data, f)

def load_training_data_pickle(filepath):
    """Load training data from a Pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# HDF5

def save_training_data_hdf5(training_data, filepath):
    """Save training data as an HDF5 file."""
    python_codes = [py for py, c in training_data]
    c_codes = [c for py, c in training_data]
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('python', data=np.array(python_codes, dtype='S'))
        f.create_dataset('c', data=np.array(c_codes, dtype='S'))

def load_training_data_hdf5(filepath):
    """Load training data from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        python_codes = [x.decode('utf-8') for x in f['python'][:]]
        c_codes = [x.decode('utf-8') for x in f['c'][:]]
    return list(zip(python_codes, c_codes))

# Parquet

def save_training_data_parquet(training_data, filepath):
    """Save training data as a Parquet file."""
    df = pd.DataFrame(training_data, columns=['python', 'c'])
    df.to_parquet(filepath, index=False)

def load_training_data_parquet(filepath):
    """Load training data from a Parquet file."""
    df = pd.read_parquet(filepath)
    return list(df[['python', 'c']].itertuples(index=False, name=None))

# Torch dataset (as .pt file)

def save_training_data_torch(training_data, filepath):
    """Save training data as a Torch .pt file."""
    torch.save(training_data, filepath)

def load_training_data_torch(filepath):
    """Load training data from a Torch .pt file."""
    return torch.load(filepath) 