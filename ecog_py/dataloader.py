"""
##############################################
DataLoader
##############################################
Loads necessary ECoG data files.
"""

import pandas as pd 
import numpy as np
import re
from os import path
import os
from process_nwb import resample
import gc
import h5py
import scipy

def load_ecog(data_dir, fsds = 3000, delimiter='\t'):
    """
    Load ECoG data and downsample from fs to fsds

    Parameters
    ----------
    data_dir : string
        Data directory to TDT ECoG data. Should be a folder than contains all the data from each channel.

    fsds : float
        Downsampled frequency for more efficient processing and reasonable system requirements.
        Must be atleast twice the highest frequency you want to analyze. 
        Note that twice the nyquist frequency is the theoretical minimum, so it is recommended to go higher to have some wiggle room.
    
    delimiter : string, default \t
        Delimiter used in csv file.

    Returns
    -------
    (num_samples, num_channels) : ndarray
        Downsampled ecog data.

    """
    channels = [int(re.match('.*CH(\d*)\..*', f)[1]) for f in os.listdir(data_dir)]
    missing_channels = list(set(np.arange(32) + 1) - set(channels))
    if(len(missing_channels) > 0):
        print("WARNING: Missing channel(s): {}".format(missing_channels))
        print("Missing channels will be replaced by the channel average.")

    num_channels = np.max(channels)
    if(num_channels <= 16):
        num_channels = 16
    elif(num_channels <= 32):
        num_channels = 32
    elif(num_channels <= 64):
        num_channels = 64
    else:
        num_channels = 128
    print("Detected {} Channels".format(num_channels))

    
    print("Loading ECoG data...\n")
    # Load data file
    ecog_raw = []
    for f in os.listdir(data_dir):
        if(re.match('.*CH.*', f)):
            df = pd.read_csv(os.path.join(data_dir, f))
            channel_num = int(re.match('.*CH(\d*)\..*', f)[1])

            fs = float(list(df['Sampling_Freq'])[0])

            print("Loading and downsampling channel {} from {}".format(channel_num, f))
            channel = np.hstack(np.array(df.iloc[:,6:-1]))
            channel_ds = resample.resample(channel, fsds, fs, real=True, axis=0)

            if(len(ecog_raw) == 0): # Initliaze ecog_raw with correct shape after loading first channel
                ecog_raw = np.zeros((num_channels, np.max(channel_ds.shape)))
            ecog_raw[channel_num-1] = channel_ds

    print("\nCleaning up...\n")
    # Free up some memory
    data = None
    channel = None
    channel_ds = None
    df = None
    gc.collect()

    print("ECoG data loading complete!\n")
    print('Data summary: \nOriginal sampling frequency: {} hz\nNew sampling frequency: {} hz\nNum channels: {}\nNum samples: {}\nRecording length: ~{} minutes'.format(fs, fsds, num_channels, np.max(ecog_raw.shape), int(np.max(ecog_raw.shape)/(fsds*60))))
    
    assert ecog_raw.T.shape[0] > ecog_raw.T.shape[1], 'ecog_raw shape is incorrect'

    return ecog_raw.T

def load_trial_timings(file_path, fs, fsds = 3000, delimiter='\t'):
    """
    Load trial timing information from Sweep_Start waveform. Sweep_Start is a square wave that encodes when trials occur.

    Parameters
    ----------
    file_path : string
        File path to sweep start data.
        Data must be stored in either a .csv or .npy with shape (num_samples, 1).

    fs : float
        Original data sampling frequency (Probably either 24kHz or 48kHz).

    fsds : float, default 3000
        Downsampled frequency for more efficient processing and reasonable system requirements.
        Must be atleast twice the highest frequency you want to analyze. 
        Note that twice the nyquist frequency is the theoretical minimum, so it is recommended to go higher to have some wiggle room.
    
    delimiter : string, default \t
        Delimiter used in csv file.

    Returns
    -------
    (num_samples,) : ndarray
        Square wave sweep start (high during trial, low in between trial).
        Downsampled to fsds (Must match fsds used for loading ECoG data!!!).

    (num_trials, 2) : ndarray
        Array of trial start / end indices.
        Example) [[0, 100], [150, 250], ..., [trial_start, trial_end]].
    """

    ext_matches = re.findall('\.(\w\w\w)', file_path)
    assert len(ext_matches) == 1, "{} extension is not valid, check that it is either a .csv or .npy".format(file_path)
    ext = ext_matches[0]
    assert ext == 'npy' or ext == 'csv', "{} extension is not valid, check that it is either a .csv or .npy".format(file_path)

    # Check file exists
    assert path.exists(file_path), "{} does not exist".format(file_path)

    # Load data file
    print("File found, loading Sweep Start data...")
    data = None
    if (ext == 'npy'):
        data = np.load(file_path)
    elif (ext == 'csv'):
        data = np.array(pd.read_csv(file_path))

    # Check ECoG data shape is correct
    assert len(data.shape) == 2 and data.shape[0] > data.shape[1] and data.shape[1] == 1, "Mismatch shape error, received shape {}, expected shape (num_samples, 1)".format(data.shape)

    # Flatten shape to 1D
    data = data.reshape(-1)

    # Downsample
    print("Downsampling data...")
    sweep_start_ds = scipy.signal.resample(data, int(len(data)/(fs/fsds)))
    sweep_start_ds = np.round(sweep_start_ds/5)*5

    # Compute trial indices
    print("Computing trial start / end indices")
    trial_indices = get_trial_indices(sweep_start_ds)

    print('Trial timing loading complete!')

    return sweep_start_ds, trial_indices

def load_stimuli_info(*file_paths, delimiter='\t'):
    """
    Loads and combines stimuli information from multiple stimuli.csv files

    Parameters
    ----------
    *file_paths : string
        File names for stimuli.csv files for each segment (S1, S2, S3 ...).
        Make sure the file names are in chronological order!.
    
    delimiter : string, default \t
        Delimiter used in csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing combined stimuli csv information.


    Examples
    --------
    >>> load_stimuli_info('S1_stimuli.csv', 'S2_stimuli.csv', 'S3_stimuli.csv')

    """

    # Load csv files into pandas DataFrames
    dfs = []
    for file_path in file_paths:
        ext_matches = re.findall('\.(\w\w\w)', file_path)
        assert len(ext_matches) == 1, "{} extension is not valid, check that it is a .csv".format(file_path)
        ext = ext_matches[0]
        assert ext == 'csv', "{} extension is not valid, check that it is a .csv".format(file_path)

        # Check file exists
        assert path.exists(file_path), "{} does not exist".format(file_path)

        dfs.append(pd.read_csv(file_path, delimiter=delimiter))

    # Combine DataFrames
    df = pd.concat(dfs)

    # Assert files are concatenated in order continuously (no missing trials)
    #assert np.max(np.diff(list(df['Trial'][df['Trial'] != 0]))) == 1 and np.min(np.diff(list(df['Trial'][df['Trial'] != 0]))) == 0, "Files were not in order"

    # Print a few rows
    #print(df[:20])

    print("Loaded stimuli.csv")
    return df

def load_trial_info(*file_paths, delimiter='\t'):
    """
    Loads and combines trial information from multiple trials.csv files.

    Parameters
    ----------
    *file_paths : string
        File names for trials.csv files for each segment (S1, S2, S3 ...).
        Make sure the file names are in chronological order!
    
    delimiter : string, default \t
        Delimiter used in csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing combined trials information.


    Examples
    --------
    >>> load_trial_info('S1_trials.csv', 'S2_trials.csv', 'S3_trials.csv')

    """

    # Load csv files into pandas DataFrames
    dfs = []
    for file_path in file_paths:
        ext_matches = re.findall('\.(\w\w\w)', file_path)
        assert len(ext_matches) == 1, "{} extension is not valid, check that it is a .csv".format(file_path)
        ext = ext_matches[0]
        assert ext == 'csv', "{} extension is not valid, check that it is a .csv".format(file_path)

        # Check file exists
        assert path.exists(file_path), "{} does not exist".format(file_path)

        dfs.append(pd.read_csv(file_path, delimiter=delimiter))

    # Combine DataFrames
    df = pd.concat(dfs)

    # Assert files are concatenated in order continuously (no missing trials)
    #assert np.max(np.diff(np.array(list(df['TrNum']))) - 1) == 0, "Files were not in order"

    # Print a few rows
    # print(df.head())

    print("Loaded trials.csv")
    return df

def load_header_info(*file_paths, delimiter='\t'):
    """
    Loads and combines header information from multiple header.csv files.

    Parameters
    ----------
    *file_paths : string
        File names for header.csv files for each segment (S1, S2, S3 ...).
        Make sure the file names are in chronological order!
    
    delimiter : string, default \t
        Delimiter used in csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing combined trials information.


    Examples
    --------
    >>> load_header_info('S1_header.csv', 'S2_header.csv', 'S3_header.csv')

    """

    # Load csv files into pandas DataFrames
    dfs = []
    for file_path in file_paths:
        ext_matches = re.findall('\.(\w\w\w)', file_path)
        assert len(ext_matches) == 1, "{} extension is not valid, check that it is a .csv".format(file_path)
        ext = ext_matches[0]
        assert ext == 'csv', "{} extension is not valid, check that it is a .csv".format(file_path)

        # Check file exists
        assert path.exists(file_path), "{} does not exist".format(file_path)

        dfs.append(pd.read_csv(file_path, delimiter=delimiter, header=None).T.drop([0]))

    # Get headers
    df_header = list(pd.read_csv(file_paths[0], delimiter=delimiter, header=None).T.iloc[0])

    # Combine DataFrames
    df = pd.concat(dfs)
    df.columns = df_header

    # Assert num rows match num of segments
    segments = np.array(list(df['SegmentNum'])).astype(int)
    assert len(df) == np.max(segments) - np.min(segments) + 1, "Number of segments do not match number of header files"
    #assert np.max(np.diff(segments) - 1) == 0, "Header files are not in order"

    # Print a few rows
    #pd.options.display.max_columns = 300
    #pd.options.display.max_rows = 300
    #print(df.T)

    print("Loaded headers.csv")
    return df

def load_sweep_start(raw_mat_events_path, fsds):
    """
    Extracts sweep_start waveform (used to align trials with ECoG data collected by TDT). This function only works with Tomer's Matlab code (contained in matlab/ folder).
    
    Parameters
    ----------
    raw_mat_events_path : string
        Relative path to RawMatEvents.mat file extracted by Tomer's Matlab code
    
    Returns
    -------
    (num_samples,) : ndarray
        Sweep_start waveform downsampled to 1000hz (1 sample per ms). Ensure sampling frequencies match when aligning trials!
    """
    with h5py.File(raw_mat_events_path, 'r') as f:
        print(f['Events'].keys())
        sampling_rate = list(f['Events']['SampleRate'])[0]
        sweep_start = list(f['Events']['Sweep_Start'])[0]
    
    print("Raw sampling rate: {}, new sampling rate: {}". format(sampling_rate[0], fsds))
    sweep_start_ds = scipy.signal.resample(sweep_start, int(len(sweep_start)/(sampling_rate[0]/fsds)))
    sweep_start_ds = np.round(sweep_start_ds/5)*5

    return sweep_start_ds

# ----------------
# Helper Functions
# ----------------

def get_trial_indices(sweep_start):
    """
    Compute trial (start,end) indices from sweep_start. If sweep_start is at 1000hz, then trial indices will be in milliseconds.

    Parameters
    ----------
    sweep_start : ndarray (num_samples,)
        sweep_start waveform (as extracted by get_sweep_start)

    Returns
    -------
    (num_trials,) : ndarray<(int, int)> 
        List of (start, end) tuples corresponding to when each trials occur
    """
    delta = np.diff(sweep_start)
    start = (delta == 5).astype(int)
    end = (delta == -5).astype(int)
    trials = list(zip(np.arange(len(start))[start == 1], np.arange(len(start))[end == 1]))
    return np.array(trials)
