"""
##############################################
DWT
##############################################
Performs discrete wavelet transforms.
"""

from process_nwb import wavelet_transform
import numpy as np

def dwt(data, fs, fsds=1000, zscore=True):
    """
    Performs Discrete Wavelet Transform on multi-channel ECoG data. The processing is done per channel to reduce RAM usage.
    
    Parameters
    ----------
    data : ndarray (num_samples, num_channels)
        ECoG data array.
    fs : float
        ECoG data sampling frequency
    fsds: 
        Post-wavelet transform downsampled frequency (to reduce memory usage).
        
    Returns
    -------
    ndarray (num_channels, num_freq_bands, num_samples)
        Wavelet transformed ECoG data.
    ndarray (num_freq_bands)
        Center frequencies of DWT filterbank 
        
    """
    # Assert data shape is correct
    assert data.shape[0] > data.shape[1], "Shape mismatch error, expected {}, got {}".format('(num_samples, num_channels)', data.shape)
    # Assert fs and fsds make sense
    assert fsds <= fs, "fsds should be less than or equal to fs. Got fs={}, fsds={}".format(fs, fsds)
    
    tf_data = []
    i = 0
    print("Wavelet transforming per channel... (This may take a while)")
    for channel in data.T:
        print('Processing channel {} ...'.format(i))
        tf, _, ctr_freq, _ = wavelet_transform.wavelet_transform(channel.reshape(-1, 1), fs, filters='rat', hg_only=False, X_fft_h=None, npad=None)
        
        if(zscore): # Z-score channel
            ztf = np.zeros(tf.shape)
            for freq_band in range(len(ctr_freq)):
                ztf[:, 0, freq_band] = zscore_signal(np.abs(tf[:, 0, freq_band]))


        tf_data.append(np.abs(ztf))
        i += 1
        
    # Stacked wavelet transformed channels back together
    tf_data = np.concatenate(tf_data, axis=1)
    # Permute the axis to get shape (num_channels, num_bands, num_samples)
    tf_data = np.moveaxis(tf_data, 0, -1)
    # Flip the bands ordering so that low frequencies are at the bottom
    tf_data = np.flip(tf_data, axis=1)
    
    return tf_data, ctr_freq

def zscore_signal(signal, axis=0, mean=None, std=None):
    if mean is None:
        mean = np.mean(signal, axis=axis)
    if std is None:
        std = np.std(signal, axis=axis)
    return (signal - mean)/std