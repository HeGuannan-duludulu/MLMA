# discrete wavelet transform

import pywt
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

"""
Input: Audio file path
Output: The denoise audio file 
"""


def read_audio_file(file_path):
    """
    Reads an audio file and returns the sampling frequency and the audio data
    """
    sampling_freq, audio = wavfile.read(file_path)
    return sampling_freq, audio


def calc_dwt(audio, level):
    """
    Calculates the discrete wavelet transform of the given audio data
    """
    coeffs = pywt.wavedec(audio, 'db1', level=level)

    return coeffs


def Dwt_denoise(file_path, level):
    """
    Driver function for the program
    """
    # Read the audio file
    sampling_freq, audio = read_audio_file(file_path)
    coeffs = calc_dwt(audio, level)
    cA2 = coeffs[0]
    zero_coeffs = [np.zeros_like(c) for c in coeffs[1:]]
    reconstructed_audio = pywt.waverec([cA2] + zero_coeffs, 'db1')

    if reconstructed_audio.shape[0] != audio.shape[0]:
        if reconstructed_audio.shape[0] > audio.shape[0]:
            reconstructed_audio = reconstructed_audio[:audio.shape[0]]
        else:
            pad_length = audio.shape[0] - reconstructed_audio.shape[0]
            reconstructed_audio = np.pad(reconstructed_audio, (0, pad_length), 'constant')

    plt.subplot(2, 1, 1)
    plt.plot(audio[1000: 3000])
    plt.title("Original audio")
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_audio[1000: 3000])
    plt.title("Reconstructed audio")
    plt.show()
    print(audio.shape, reconstructed_audio.shape)
    return reconstructed_audio


if __name__ == '__main__':
    test_file_path = "data/training-e/e00054.wav"
    from utils import dwt_denoise
    _, audio = wavfile.read(test_file_path)
    reconstructed_audio = dwt_denoise(audio, 2)
    plt.subplot(2, 1, 1)
    # plot the spectrogram
    plt.specgram(audio, Fs=2000)
    plt.title("Original audio")
    plt.subplot(2, 1, 2)
    plt.specgram(reconstructed_audio, Fs=2000)
    plt.title("Reconstructed audio")
    plt.show()

