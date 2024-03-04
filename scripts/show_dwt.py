

from utils import dwt_denoise
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

test_file_path = "../data/training-e/e00012.wav"
sampling_rate, audio = wavfile.read(test_file_path)
reconstructed_audio = dwt_denoise(audio, 2)

fft_original = fftpack.fft(audio)
fft_reconstructed = fftpack.fft(reconstructed_audio)


freqs = fftpack.fftfreq(len(fft_original)) * sampling_rate

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(freqs, np.abs(fft_original))
plt.title('Original Audio Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(freqs, np.abs(fft_reconstructed))
plt.title('Reconstructed Audio Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()




plt.figure(figsize=(11, 5))
plt.subplot(2, 1, 1)
plt.plot(audio[2000: 3000])
plt.xticks([])
plt.yticks([])
plt.title("Original audio", fontsize=16)
plt.subplot(2, 1, 2)
plt.plot(reconstructed_audio[2000: 3000])
plt.title("Reconstructed audio", fontsize=16)
plt.tight_layout()
plt.xticks([])
plt.yticks([])
save_path = "../images/dwt_reconstructed.svg"
plt.savefig(save_path, dpi=300)
plt.show()

wavfile.write("reconstructed_audio.wav", 2000, reconstructed_audio.astype(np.int16))
