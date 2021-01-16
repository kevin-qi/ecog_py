"""
##############################################
Preprocessing
##############################################
Preprocess ECoG data using CAR and Notch filtering line noise
"""

import process_nwb
import numpy as np

def preprocess(ecog, fs, mean_frac=0.95, round_func=np.ceil):
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

    # Common Average Reference
    ecog_car = process_nwb.common_referencing.subtract_CAR(ecog, mean_frac=mean_frac, round_func=round_func)

    # Notch filter 60hz power line noise
    ecog_notch = np.zeros(ecog_car.shape).T
    for channel_num in range(np.min(ecog_car.shape)):
        print("Notch Filtered Channel", channel_num)
        channel_notch = process_nwb.linenoise_notch.apply_linenoise_notch(ecog_car[:, channel_num].reshape(-1, 1), fs, fft=True)
        ecog_notch[channel_num] = channel_notch.reshape(-1)
    ecog_notch = ecog_notch.T

    return ecog_notch