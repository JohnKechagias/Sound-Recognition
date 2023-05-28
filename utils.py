import librosa
import soundfile
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt


class Utils:
    @staticmethod
    def signal_to_file(filename: str, signal: ndarray, sr: int):
        soundfile.write(filename, signal, sr)

    @staticmethod
    def print_waveform(signal: ndarray, sr: int, color: str = "blue"):
        librosa.display.waveshow(signal, sr=sr, alpha=0.5, color=color)

    @staticmethod
    def print_spectogram(spectogram: ndarray):
        db = librosa.amplitude_to_db(np.abs(spectogram), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(db, x_axis="time", y_axis="log", ax=ax)

        ax.set(title="Using a logarithmic frequency axis")
        fig.colorbar(img, ax=ax, format="%+2.f dB")

    @staticmethod
    def show_graphs():
        plt.show()
