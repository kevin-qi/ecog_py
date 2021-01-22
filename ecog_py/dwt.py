"""
##############################################
DWT
##############################################
Performs discrete wavelet transforms.
"""

from process_nwb import wavelet_transform
import numpy as np
import scipy

def dwt(data, fs, fsds=1000, zscore=True, hg_only=False):
    """
    Performs Discrete Wavelet Transform on multi-channel ECoG data. The processing is done channel by channel to reduce RAM usage.
    
    Parameters
    ----------
    data : ndarray (num_samples, num_channels)
        ECoG data array.
    fs : float
        ECoG data sampling frequency
    fsds: 
        Post-wavelet transform downsampled frequency (to reduce memory usage).
    zscore : bool, default True
        If True, zscore each frequency band. (Enables comparison across frequency bands despite 1/f power falloff)
    hg_only : bool, default False
        If True, only compute high gamma bands (70-150hz)
        
    Returns
    -------
    (num_samples, num_channels) : ndarray
        Wavelet transformed ECoG data.
    (num_freq_bands,) : ndarray
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
        tf, _, ctr_freq, _ = wavelet_transform.wavelet_transform(channel.reshape(-1, 1), fs, filters='rat', hg_only=hg_only, X_fft_h=None, npad=None)

        if(zscore): # Z-score channel
            ztf = np.zeros((int(np.max(tf.shape)/int(fs/int(fsds))), 1, len(ctr_freq)))
            for freq_band in range(len(ctr_freq)):
                z = zscore_signal(np.abs(tf[:,0, freq_band]))
                z_ds = scipy.signal.resample(z, int(np.max(z.shape)/int(fs/int(fsds))), axis=0) # Downsample
                ztf[:, 0, freq_band] = z_ds
            tf_data.append(ztf)
        else: # No Z-score
            tf = scipy.signal.resample(tf, int(np.max(tf.shape)/int(fs/int(fsds))), axis=0) # Downsample
            tf_data.append(np.abs(tf))
        i += 1


    
    # Stacked wavelet transformed channels back together
    tf_data = np.concatenate(tf_data, axis=1)
    # Permute the axis to get shape (num_channels, num_bands, num_samples)
    tf_data = np.moveaxis(tf_data, 0, -1)
    # Flip the bands ordering so that low frequencies are at the bottom
    tf_data = np.flip(tf_data, axis=1)
    
    return tf_data, np.flip(ctr_freq)

def zscore_signal(signal, axis=0, mean=None, std=None):
    if mean is None:
        mean = np.mean(signal, axis=axis)
    if std is None:
        std = np.std(signal, axis=axis)
    return (signal - mean)/std