import os
from matplotlib import pyplot as plt

import numpy as np

from constants import extracted_samples_path, dataset_path, sample_path
from dataset_formatter import DatasetFormatter
from feature_extractor import ProcessPipeline
from neural_network import FeedForwardNetwork
from sound_parser import SoundParser


def main():
    SoundParser.parse(sample_path)

    test_features = []
    for file in os.listdir(extracted_samples_path):
        filepath = extracted_samples_path / file
        mfcc = ProcessPipeline.get_mfcc_from_file(filepath)
        test_features.append(mfcc)

    epochs = 50
    _, train = DatasetFormatter.get_datasets(dataset_path)
    train_features, train_labels = train
    model = FeedForwardNetwork(batch_size=32)
    test_history = model.train(train_features, train_labels, epochs=epochs)
    model.predict(np.asarray(test_features))

    acc = test_history.history['accuracy']
    loss = test_history.history['loss']

    epochs_range = range(epochs)
    print("The results are being visualized")
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)

    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("nice.png")


if __name__ == "__main__":
    main()
