"""
##############################################
Preprocessing
##############################################
Preprocess ECoG data using CAR and Notch filtering line noise
"""

import process_nwb
import numpy as np
import scipy
import matplotlib.pyplot as plt

def preprocess(ecog, fs, car=True, mean_frac=0.95, round_func=np.ceil):
    """
    Performs common average referencing and removes 60hz line noise via Notch filter.

    Parameters
    ----------
    ecog : ndarray (num_samples, num_channels)
        ECoG data
    fs : float
        ECoG data sampling frequency
    mean_frac : float, default 0.95
        Fraction of channels to include in mean. Interpolates between mean and median.
    round_func : callable, default np.ceil
        Function that determines how to round the channel number

    Returns
    -------
    (num_samples, num_channels) : ndarray
        Processed ECoG data (Re-referenced and de-noised)
    """

    # Assert data shape is correct
    assert ecog.shape[0] > ecog.shape[1], "Shape mismatch error, expected {}, got {}".format('(num_samples, num_channels)', ecog.shape)

    if(car):
        ecog_car = process_nwb.common_referencing.subtract_CAR(ecog, mean_frac=mean_frac, round_func=round_func)
    else:
        ecog_car = ecog
        
    # Notch filter 60hz power line noise
    ecog_notch = np.zeros(ecog_car.shape).T
    for channel_num in range(np.min(ecog_car.shape)):
        print("Notch Filtered Channel", channel_num)
        channel_notch = process_nwb.linenoise_notch.apply_linenoise_notch(ecog_car[:, channel_num].reshape(-1, 1), int(fs), fft=True)
        ecog_notch[channel_num] = channel_notch.reshape(-1)
    ecog_notch = ecog_notch.T

    return ecog_car, ecog_notch

def find_bad_channels(ecog_raw, threshold=0.2):
    """
    Finds any bad channels in the ecog recording.

    Parameters
    ----------
    ecog_raw : ndarray (num_samples, num_channels)
        Raw ecog signal (ideally downsampled to a manageable frequency, ~3000hz)
    threshold : float
        Pearson correlation coefficient threshold to be considered a bad channel.

    Returns
    -------
    list<int>
        List of bad channel(s)
    """

    # Check ecog_raw has the expected shape
    assert ecog_raw.shape[0] > ecog_raw.shape[1], "Shape Mismatch Error: ecog_raw should have shape (num_samples, num_channels)"

    num_channels = ecog_raw.shape[1]
    dropped_channel_map = []

    cutoff = 50000
    pearsonr = np.zeros((num_channels,num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            pearsonr[i][j] = scipy.stats.pearsonr(ecog_raw[:cutoff, i], ecog_raw[:cutoff, j])[0]

    for ch in range(num_channels):
        plt.plot(ecog_raw[:cutoff, ch])
    plt.title('Raw ECoG Channels')
    plt.show()

    plt.title('Pearson correlation plot')
    plt.imshow(pearsonr)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    blacklist = []
    for ch in range(num_channels):
        if (pearsonr[0] < threshold)[ch]:
            print("Bad channel {}".format(ch))
            blacklist.append(ch)
        else:
            dropped_channel_map.append(ch)
    
    ecog_clean = np.delete(ecog_raw, blacklist, axis=1)
    for ch in range(np.min(ecog_clean.shape)):
        assert np.allclose(ecog_raw[:,dropped_channel_map[ch]], ecog_clean[:,ch])

    if(blacklist == []):
        print("No bad channels found!")

    return ecog_clean, blacklist, dropped_channel_map