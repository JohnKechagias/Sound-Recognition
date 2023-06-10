import os
import random
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from feature_extractor import ProcessPipeline


class DatasetFormatter:
    sample_size_per_class = 50

    @staticmethod
    def download_dataset():
        return tfds.load("spoken_digit", data_dir="dataset/", download=True)

    @classmethod
    def get_datasets(cls, path: str, split_ratio: float = 0.8):
        test_features, test_labels = [], []
        train_features, train_labels = [], []

        files = [file for file in os.scandir(path)]
        random.shuffle(files)

        train_num = split_ratio * cls.sample_size_per_class

        for file in files:
            filename_parts = Path(file).stem.split("_")
            label, _, file_number = filename_parts
            label, file_number = int(label), int(file_number)

            features = train_features if file_number < train_num else test_features
            labels = train_labels if file_number < train_num else test_labels

            labels.append(label)
            features.append(ProcessPipeline.get_mfcc_from_file(file))

        test_dataset = (np.stack(np.asarray(test_features)), np.asarray(test_labels))
        train_dataset = (np.stack(np.asarray(train_features)), np.asarray(train_labels))

        return test_dataset, train_dataset
