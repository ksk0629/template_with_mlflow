# Reference: https://toukei-lab.com/mnist
import argparse
from typing import List, Tuple

import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

FLATTENED_SIZE: int = 784  # dimension of flattened one mnist data


def load_scaled_mnist(validation_size: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load and separate scaled MNIST dataset into ones for training, evaluating, and testing.

    :param float validation_size: ratio of validation dataset size
    :return List[numpy.ndarray] train_dataset: dataset for training whose length is 2, the first element is data and the second one is the labels
    :return List[numpy.ndarray] validation_dataset: dataset for evaluating whose length is 2, the first element is data and the second one is the labels
    :return List[numpy.ndarray] test_dataset: dataset for testing whose length is 2, the first element is data and the second one is the labels
    """
    # load mnist data
    (x, y), (test_data, test_labels) = mnist.load_data()

    # Separate x into a training dataset and a validation dataset
    train_data, validation_data, train_labels, validation_labels = train_test_split(x, y, test_size=validation_size)

    # Flatten all of data
    train_data = train_data.reshape(-1, FLATTENED_SIZE)
    validation_data = validation_data.reshape(-1, FLATTENED_SIZE)
    test_data = test_data.reshape(-1, FLATTENED_SIZE)

    # Scale those data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data, sclaed_validation_data, scaled_test_data = scaler.fit_transform([train_data, validation_data, test_data])

    # Make list whose length two, the first one is data and another is label
    train_dataset = [scaled_train_data, train_labels]
    validation_dataset = [sclaed_validation_data, validation_labels]
    test_dataset = [scaled_test_data, test_labels]

    return train_dataset, validation_dataset, test_dataset


def train_and_evaluate_cnn(experiment_name: str, run_name: str,
                           seed: int, validation_size: float,
                           n_hidden: int, n_features: int,
                           epochs: int, batch_size: int, learning_rate: float) -> None:
    """Train and evaluate CNN on sclaed MNIST datasets.

    :param str experiment_name: experiment name, which is logged to mlflow
    :param str run_name: run name, which is logged to mlflow
    :param int seed: random seed to fix random values
    :param float validation_size: ratio of validation dataset size
    :param int n_hidden: number of hidden layers
    :param int n_features: number of features
    :param int epochs: number of epochs
    :param int batch_size: training batch size
    :param float learning_rate: leraning rate
    """
    # Set an experiment name
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log automatically
        mlflow.keras.autolog()

        # Fix and log seed value
        np.random.seed(seed)
        mlflow.log_param("numpy seed", seed)

        # Get scaled mnist dataset and log the validation size
        train_dataset, validation_dataset, test_dataset = load_scaled_mnist(validation_size=validation_size)
        mlflow.log_param("validation size", validation_size)

        # Expand those dataset into data and labels
        train_data, train_labels = train_dataset
        validation_data, validation_labels = validation_dataset
        test_data, test_labels = test_dataset

        # Convert numerical labels into categorical labels
        train_labels = to_categorical(train_labels)
        validation_labels = to_categorical(validation_labels)

        # Create the model
        model = Sequential()
        model.add(Dense(n_hidden, activation="relu", input_shape=(n_features,)))
        model.add(Dense(n_hidden, activation="relu"))
        model.add(Dense(n_hidden, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=["mae", "accuracy"])

        # Fit the model
        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=True,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)],
                  validation_data=(validation_data, validation_labels))

        # Evaluate the model and log the accuracy
        test_predicted = np.argmax(model.predict(test_data), axis=1)
        accuracy = sum(test_predicted == test_labels) / len(test_predicted)
        mlflow.log_metric("accuracy", accuracy)

    print(f"accuracy is {accuracy}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN on MNIST dataset as how to use mlflow example")

    parser.add_argument("experiment_name", type=str, default="mnist with cnn")
    parser.add_argument("run_name", type=str, default=None)
    parser.add_argument("seed", type=int, default=57)
    parser.add_argument("validation_size", type=float, default=0.2)
    parser.add_argument("epochs", type=int, default=1000)
    parser.add_argument("batch_size", type=int, default=8)
    parser.add_argument("n_features", type=int, default=784)
    parser.add_argument("n_hidden", type=int, default=100)
    parser.add_argument("learning_rate", type=float, default=0.01)

    args = parser.parse_args()

    train_and_evaluate_cnn(args.experiment_name, args.run_name,
                           args.seed, args.validation_size,
                           args.n_hidden, args.n_features,
                           args.epochs, args.batch_size, args.learning_rate)
