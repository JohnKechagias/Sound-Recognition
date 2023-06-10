import numpy as np
import librosa
from numpy import ndarray
from sklearn.preprocessing import scale


class ProcessPipeline:
    sr = 22050
    audio_duration = 0.5
    audio_sample_size = int(sr // audio_duration)

    @classmethod
    def get_mfcc_from_file(cls, filepath: str):
        signal, sr = librosa.load(filepath, sr=cls.sr)
        signal = librosa.util.fix_length(data=signal, size=cls.audio_sample_size)
        return cls._get_mfcc(signal, sr)

    @classmethod
    def _get_mfcc(cls, signal: ndarray, sr: int) -> ndarray:
        mfcc = librosa.feature.mfcc(y=signal, sr=sr)
        mfcc = scale(mfcc, axis=1)
        return mfcc
