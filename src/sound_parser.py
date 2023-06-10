from pathlib import Path

import librosa
import numpy as np
from numpy import ndarray

from constants import extracted_samples_path
from utils import Utils


class SoundParser:
    sr = 22050

    @classmethod
    def parse(cls, filepath: Path):
        signal = cls.load_sample(filepath)
        foreground_signal = cls.extract_foreground(signal)
        cls.split_foreground(foreground_signal)

    @classmethod
    def load_sample(cls, filepath: Path) -> ndarray:
        signal, _ = librosa.load(filepath, sr=cls.sr)
        signal, _ = librosa.effects.trim(signal)
        Utils.save_waveform(signal, cls.sr, "initial_waveform")
        return signal

    @classmethod
    def extract_foreground(
        cls, signal: ndarray, loss_margin: int = 10, power: int = 2
    ) -> ndarray:
        spectogram = librosa.stft(signal)
        Utils.save_spectogram(spectogram, "sample_spectogram")
        signal_magnitude, phase = librosa.magphase(spectogram)

        filtered_magnitude = librosa.decompose.nn_filter(signal_magnitude)
        filtered_magnitude = np.minimum(filtered_magnitude, signal_magnitude)
        mangitude_diff = np.subtract(signal_magnitude, filtered_magnitude)

        foreground_mask = librosa.util.softmask(
            mangitude_diff, loss_margin * filtered_magnitude, power=power
        )

        foreground_magnitude = foreground_mask * signal_magnitude
        foreground_magnitude = foreground_magnitude * phase
        foreground_signal = librosa.istft(foreground_magnitude)
        Utils.save_waveform(foreground_signal, cls.sr, "foreground_waveform")
        return foreground_signal

    @classmethod
    def split_foreground(cls, signal: ndarray):
        segments_matrix = librosa.effects.split(signal, top_db=40)

        for index in range(segments_matrix.shape[0]):
            start = segments_matrix[index][0]
            end = segments_matrix[index][1]
            segmented_signal = signal[start:end]
            filename = f"{index + 1}_word.wav"
            filepath = extracted_samples_path / filename
            Utils.signal_to_file(filepath, segmented_signal, cls.sr)

            base_freq = cls.get_base_freq(segmented_signal)
            print(f"Base frequency of {index} word: {base_freq}")

    @classmethod
    def get_base_freq(cls, signal: ndarray) -> float:
        frames_base_freq, _, _ = librosa.pyin(
            signal,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=cls.sr,
        )
        filtered_frames_base_freq = frames_base_freq[~np.isnan(frames_base_freq)]
        return np.mean(filtered_frames_base_freq)
