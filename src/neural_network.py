from matplotlib import pyplot as plt
from numpy import ndarray
from tensorflow import keras, optimizers


class FeedForwardNetwork:
    classes = 10

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.model = keras.Sequential(
            [
                keras.layers.Reshape(target_shape=(20 * 87,), input_shape=(20, 87)),
                keras.layers.Dense(units=512, activation="relu"),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(units=128, activation="relu"),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(units=64, activation="relu"),
                keras.layers.Dense(units=24, activation="relu"),
                keras.layers.Dense(units=self.classes, activation="softmax"),
            ]
        )

        lr_schedule = optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.001,
            decay_steps=1e3 * self.batch_size,
            decay_rate=1,
            staircase=False,
        )

        self.model.compile(
            optimizer=optimizers.Adam(lr_schedule),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, features: ndarray, labels: ndarray, epochs: int = 100):
        history = self.model.fit(features, labels, epochs=epochs, batch_size=self.batch_size)
        self.model.save("number_to_output.h5", overwrite=True, include_optimizer=True)
        return history

    def predict(self, features: ndarray):
        predictions = self.model.predict(features)
        for index in range(features.shape[0]):
            prediction = predictions[index].argmax()
            print(f"Predicted label for the {index} input: {prediction}")
