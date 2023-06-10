import librosa
import numpy as np
import soundfile
from matplotlib import pyplot as plt
from numpy import ndarray


class Utils:
    @staticmethod
    def signal_to_file(filename: str, signal: ndarray, sr: int):
        soundfile.write(filename, signal, sr)

    @staticmethod
    def save_waveform(signal: ndarray, sr: int, filename: str, color: str = "r"):
        fig, ax = plt.subplots()
        librosa.display.waveshow(y=signal, sr=sr, alpha=0.5, color=color)
        fig.savefig(f"{filename}.png", bbox_inches="tight")

    @staticmethod
    def save_spectogram(spectogram: ndarray, filename: str):
        db = librosa.amplitude_to_db(np.abs(spectogram), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(db, x_axis="time", y_axis="log", ax=ax)

        ax.set(title="Using a logarithmic frequency axis")
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        fig.savefig(f"{filename}.png", bbox_inches="tight")

    @staticmethod
    def save_mfcc(mfcc: ndarray, sr: int, filename: str):
        fig, ax = plt.subplots()
        librosa.display.specshow(mfcc, x_axis="time", y_axis="mel", sr=sr, ax=ax)
        fig.savefig(f"{filename}.png", bbox_inches="tight")
