import re

import librosa
import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import get_window

import utils


class FreqFeature:

    def __init__(self, audio_path, sample_freq=2000, csv_path="data/training-a/REFERENCE-SQI.csv", use_dwt=False):
        self.audio_path = audio_path
        self.label_dict = utils.read_csv(csv_path)

        self.sample_freq, self.audio = wavfile.read(self.audio_path)  # 2000Hz
        if use_dwt:
            self.audio = utils.dwt_denoise(self.audio)
        self.audio = self.normalize_audio()
        self.freq_min = 0
        self.freq_max = self.sample_freq / 2  # 0-1000Hz
        self.num_mel_filters = 13
        self.fft_size = 2048  # 2048 FFT

        self.frame_time = 0.040  # 40ms, 50% overlap
        self.overlap_time = 0.020  # 10ms
        self.frame_len = int(self.frame_time * self.sample_freq)  # points: 80， 40ms
        self.frame_step = int(self.overlap_time * self.sample_freq)  # points: 20， 10ms overlap
        self.win_func = get_window("hamm", int(self.frame_len), fftbins=True)

    def enframe(self, input_audio, frame_len, frame_step, win_func):
        wave_len = len(input_audio)

        if wave_len < frame_len:
            frame_num = 1
        else:
            frame_num = int(np.ceil((wave_len - frame_len + frame_step) / frame_step))

        pad_len = int((frame_num - 1) * frame_step + frame_len)

        zeros = np.zeros((pad_len - wave_len,))
        pad_signal = np.concatenate((input_audio, zeros[:, np.newaxis]))

        frame_idx = np.tile(np.arange(0, frame_len), (frame_num, 1)) + \
                    np.tile(np.arange(0, frame_num * frame_step, frame_step), (frame_len, 1)).T

        frames = pad_signal[frame_idx].squeeze()
        win = np.tile(win_func, (frame_num, 1))
        return frames * win

    def normalize_audio(self):
        self.audio = self.audio.astype(np.float32)
        self.audio = self.audio / np.max(np.abs(self.audio))
        self.audio = self.audio.reshape(-1, 1)
        return self.audio

    def freq_to_mel(self, freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_freq(self, mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def compute_mfcc(self, num_filters, power_spectrum, num_coeffs):
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sample_freq / 2.0) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bins = np.floor((self.fft_size + 1) * hz_points / self.sample_freq)

        fbank = np.zeros((num_filters, int(np.floor(self.fft_size / 2 + 1))))
        for m in range(1, num_filters + 1):
            f_m_minus = int(bins[m - 1])
            f_m = int(bins[m])
            f_m_plus = int(bins[m + 1])
            fbank[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - bins[m - 1]) / (bins[m] - bins[m - 1])
            fbank[m - 1, f_m:f_m_plus] = (bins[m + 1] - np.arange(f_m, f_m_plus)) / (bins[m + 1] - bins[m])

        filter_banks = np.dot(power_spectrum, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        mfcc = dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_coeffs + 1)]
        ncoeff = mfcc.shape[0]
        cep_lifter = ncoeff
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        return mfcc

    def compute_delta_mfcc(self, mfcc):
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)

    def hamming_win(self):
        win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(self.fft_size) / (self.fft_size - 1))
        return win

    def compute_each_frame_mfcc(self):
        whole_audio_feature = []
        seg_audio_list = utils.segment_audio(self.audio, self.sample_freq)

        for each_seg in seg_audio_list:
            each_seg_frame = self.enframe(each_seg, self.frame_len, self.frame_step, self.win_func)
            each_seg_feature = []
            for each_frame in each_seg_frame:
                fft_result = np.absolute(np.fft.rfft(each_frame, self.fft_size))
                power_spectrum = (1.0 / self.fft_size) * (fft_result ** 2)
                mel_coeff = self.compute_mfcc(self.sample_freq, power_spectrum, num_coeffs=self.num_mel_filters)
                mfcc_all = self.compute_delta_mfcc(mel_coeff)
                each_seg_feature.append(mfcc_all)

            each_seg_feature = each_seg_feature[:-1]
            each_seg_feature = np.array(each_seg_feature)
            whole_audio_feature.append(each_seg_feature)

        whole_audio_feature = whole_audio_feature[:-1]
        whole_audio_feature = np.array(whole_audio_feature)

        return whole_audio_feature

    def get_labels(self):
        # audio_name = self.audio_path.split("//")[-1].split(".")[0]
        pattern = r'[a-f]\d{4,5}'

        # Searching for the pattern in the string
        match = re.search(pattern, self.audio_path)

        # Extracting the match, if found
        found = match.group() if match else "No match found"
        labels = self.label_dict[found]
        if labels < 0:
            labels = 0
        return labels

    def get_mfccs_features_and_labels(self):
        mfccs = self.compute_each_frame_mfcc()
        labels = self.get_labels()
        labels = np.array([labels] * len(mfccs))
        return mfccs, labels[:, None]


if __name__ == "__main__":
    num = 67
    audio_path = "data/training-a/a00{}.wav".format(num)
    freq_feature = FreqFeature(audio_path)
    this_wav_mfccs, labels = freq_feature.get_mfccs_features_and_labels()

    print("this_wav_mfccs.shape", this_wav_mfccs.shape, labels.shape)
