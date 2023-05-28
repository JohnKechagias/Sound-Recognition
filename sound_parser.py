from pathlib import Path
from typing import Tuple
import librosa
from utils import Utils
from numpy import ndarray
import numpy as np


class SoundParser:
    sr = 48000

    def __init__(self, filepath: Path):
        self.filepath = filepath

        signal = self.load_sample(filepath)
        foreground_signal = self.extract_foreground(signal)
        self.split_foreground(foreground_signal)

    def load_sample(self, filepath: Path) -> ndarray:
        signal, sr = librosa.load(filepath, sr=self.sr)
        # Utils.print_waveform(signal, sr)
        return signal

    def extract_foreground(
        self, signal: ndarray, loss_margin: int = 2, power: int = 2
    ) -> ndarray:
        spectogram = librosa.stft(signal)
        # Utils.print_spectogram(spectogram)
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
        Utils.print_waveform(foreground_signal, self.sr, "black")

        return foreground_signal

    def split_foreground(self, signal: ndarray):
        segments_matrix = librosa.effects.split(signal)

        for index in range(segments_matrix.shape[0]):
            start = segments_matrix[index][0]
            end = segments_matrix[index][1]
            segmented_signal = signal[start:end]
            filename = f"number{index + 1}.wav"
            Utils.signal_to_file(filename, segmented_signal, self.sr)
 