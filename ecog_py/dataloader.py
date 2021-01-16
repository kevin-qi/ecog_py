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
from process_nwb import resample
import gc
import scipy

def load_ecog(file_path, fs, fsds = 3000, delimiter='\t'):
	"""
	Load ECoG data and downsample from fs to fsds

	Parameters
	----------
	file_path : string
		File path to ecog data.
		ECoG data must be stored in either a .csv or .npy with shape (num_samples, num_channels)

	fs : float
		Original data sampling frequency (Probably either 24kHz or 48kHz).

	fsds : float
		Downsampled frequency for more efficient processing and reasonable system requirements.
		Must be atleast twice the highest frequency you want to analyze. 
		Note that twice the nyquist frequency is the theoretical minimum, so it is recommended to go higher to have some wiggle room.
	
	delimiter : string, default \t
		Delimiter used in csv file.

	Returns
	-------
	ndarray (num_samples, num_channels)
		Downsampled ecog data with shape (num_samples, num_channels).

	"""

	# Check extension is valid
	ext_matches = re.findall('\.(\w\w\w)', file_path)
	assert len(ext_matches) == 1, "{} extension is not valid, check that it is either a .csv or .npy".format(file_path)
	ext = ext_matches[0]
	assert ext == 'npy' or ext == 'csv', "{} extension is not valid, check that it is either a .csv or .npy".format(file_path)

	# Check file exists
	assert path.exists(file_path), "{} does not exist".format(file_path)

	print("File found, loading ECoG data...")
	# Load data file
	data = None
	if (ext == 'npy'):
		data = np.load(file_path)
	elif (ext == 'csv'):
		data = np.array(pd.read_csv(file_path))

	# Check ECoG data shape is correct
	assert data.shape[0] > data.shape[1], "Mismatch shape error, received shape {}, expected shape (num_samples, num_channels)".format(data.shape)

	print("Downsampling data... (this may take a while)")
	# Resample data
	ecog_ds = []
	for channel in data.T:
	    ecog_ds.append(resample.resample(channel, fsds, fs, real=True, axis=0))

	ecog_ds = np.array(ecog_ds).T

	print("Cleaning up...")
	# Free up some memory
	data = None
	channel = None
	gc.collect()

	print("ECoG data loading complete!\n")
	print('Data summary: \nOriginal sampling frequency: {} hz\nNew sampling frequency: {} hz\nNum channels: {}\nNum samples: {}\nRecording length: ~{} minutes'.format(fs, fsds, ecog_ds.shape[1], ecog_ds.shape[0], int(ecog_ds.shape[0]/(fsds*60))))
	

	return ecog_ds

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
	ndarray (num_samples,)
		Square wave sweep start (high during trial, low in between trial).
		Downsampled to fsds (Must match fsds used for loading ECoG data!!!).

	ndarray (num_trials, 2)
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
	assert np.max(np.diff(list(df['Trial'][df['Trial'] != 0]))) == 1 and np.min(np.diff(list(df['Trial'][df['Trial'] != 0]))) == 0, "Files were not in order"

	# Print a few rows
	print(df[:20])

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
	assert np.max(np.diff(np.array(list(df['TrNum']))) - 1) == 0, "Files were not in order"

	# Print a few rows
	print(df.head())

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
	assert len(df) == np.max(segments), "Number of segments do not match number of header files"
	assert np.max(np.diff(segments) - 1) == 0, "Header files are not in order"

	# Print a few rows
	pd.options.display.max_columns = 300
	pd.options.display.max_rows = 300
	print(df.T)

	return df

# ----------------
# Helper Functions
# ----------------

def get_trial_indices(sweep_waveform):
    delta = np.diff(sweep_waveform)
    start = (delta == 5).astype(int)
    end = (delta == -5).astype(int)
    trials = list(zip(np.arange(len(start))[start == 1], np.arange(len(start))[end == 1]))
    return np.array(trials)
