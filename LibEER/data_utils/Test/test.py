from scipy import signal
from  scipy.signal.windows import hann
import numpy as np
import math
from scipy.fftpack import fft,ifft


def segment_data(data, time_window, sample_rate, overlap):

    window_len = int(time_window * sample_rate)

    # Step 1: get the total number of samples
    total_samples = data.shape[1]

    # Step 2: calculate how many extra points don't fit into 200-sample blocks
    extra = total_samples % window_len

    # Step 3: trim the first `extra` samples
    data_trimmed = data[:, extra:]  # discard from the beginning

    # Step 4: reshape into blocks of shape (62, 200, num_blocks)
    num_blocks = data_trimmed.shape[1] // window_len
    segmented_data = data_trimmed.reshape(data.shape[0], num_blocks, window_len)

    # (Optional) Reorder to (num_blocks, 62, 200) for convenience
    segmented_data = np.transpose(segmented_data, (1, 0, 2))

    return segmented_data


def de_extraction_fourier(data_trial, sample_rate, extract_bands,  time_window, overlap):
    '''
    compute DE
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency
    output: DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    fStart, fEnd = map(list, zip(*extract_bands))
    # STFTN is the same 
    STFTN = sample_rate


    #Convert frecuency from Hz to positions of a vector
    fStartNum =(np.array(fStart) / sample_rate * STFTN).astype(int)
    fEndNum = (np.array(fEnd)/sample_rate*STFTN).astype(int)

    #Hanning window
    Hlength=time_window*sample_rate
    Hwindow=hann(Hlength)
    #Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])
    de_trial = []
    for data in data_trial:
        #Compute differential entropy
        n=data.shape[0]
        #Compute fft
        Hdata = data*Hwindow
        FFTdata=fft(Hdata,STFTN)
        magFFTdata=abs(FFTdata[:, 0:int(STFTN/2)])
        #Compute energy and de per frecuancy band
        de = np.zeros([n,len(fStart)])
        for i, (start, end) in enumerate(zip(fStartNum, fEndNum)):
            band = magFFTdata[:, start-1:end]  # (n_channels, ancho de banda)
            E = np.sum(band**2, axis=1) / (end - start + 1)  # (n_channels,)
            de[:, i] = np.log2(100 * E)
        de_trial.append(de)

    return de_trial



def de_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    DE feature extraction
    :param data: original eeg data, input shape: (channel, filter_data)
    :param sample_rate: sample rate of eeg signal
    :param extract_bands: the frequency bands that needs to be extracted
    :param time_window: time window of one extract part
    :param overlap: overlap
    :return: de feature need to be computed
    """
    if extract_bands is None:
        extract_bands = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]
    nyq = 0.5 * sample_rate
    noverlap = int(overlap * sample_rate)
    window_size = int(time_window * sample_rate)
    if noverlap != 0:
        sample_num = (data.shape[1] - window_size) // (window_size - noverlap)
    else:
        sample_num = (data.shape[1]) // window_size
    de_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))
    for b_idx, band in enumerate(extract_bands):
        b, a = signal.butter(3, [band[0]/nyq, band[1]/nyq], 'bandpass')
        band_data = signal.filtfilt(b, a, data)
        t = 0
        for i in range(sample_num):
            de_data[i,:,b_idx] = 1 / 2 * np.log2(2 * np.pi * np.e * np.var(band_data[:,t:t+window_size], axis=1, ddof=1))
            t += window_size-noverlap
    return de_data

def get_DE(data, stft_para):
    '''
    compute DE
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency
    output: DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    STFTN= stft_para['stftn']
    fStart = stft_para['fStart']
    fEnd = stft_para['fEnd']
    window = stft_para['window']
    fs = stft_para['fs']


    #Convert frecuency from Hz to positions of a vector
    fStartNum =(np.array(fStart) / fs * STFTN).astype(int)
    fEndNum = (np.array(fEnd)/fs*STFTN).astype(int)

    #Hanning window
    Hlength=window*fs
    Hwindow=hann(Hlength)
    #Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    #Compute differential entropy
    n=data.shape[0]
    #Compute fft
    Hdata = data*Hwindow
    FFTdata=fft(Hdata,STFTN)
    magFFTdata=abs(FFTdata[:, 0:int(STFTN/2)])
    #Compute energy and de per frecuancy band
    de = np.zeros([n,len(fStart)])
    for i, (start, end) in enumerate(zip(fStartNum, fEndNum)):
        band = magFFTdata[:, start-1:end]  # (n_channels, ancho de banda)
        E = np.sum(band**2, axis=1) / (end - start + 1)  # (n_channels,)
        de[:, i] = np.log2(100 * E)

    return de