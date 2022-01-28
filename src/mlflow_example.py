# Reference: [https://toukei-lab.com/mnist]
# Author: Keisuke Sato

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import mlflow


def load_normalized_mnist(validation_size):
    """
    Load normalized MNIST dataset.

    Parameter
    ---------
    validation_size : float
        validation size

    Returns
    -------
    train_dataset : list
        training dataset whose length is 2, the first element is data and the second one is the labels
    validation_dataset : list
        validation dataset whose length is 2, the first element is data and the second one is the labels
    test_dataset : list
        test dataset whose length is 2, the first element is data and the second one is the labels
    """
    # load mnist data
    (x, y), (test_data, test_labels) = mnist.load_data()

    # Separate x into a training dataset and a validation dataset
    train_data, validation_data, train_labels, validation_labels = train_test_split(
        x, y, test_size=validation_size)

    train_data = train_data.reshape(-1, 784)
    validation_data = validation_data.reshape(-1, 784)
    test_data = test_data.reshape(-1, 784)

    train_data = normalize(train_data)
    validation_data = normalize(validation_data)
    test_data = normalize(test_data)

    train_dataset = [train_data, train_labels]
    validation_dataset = [validation_data, validation_labels]
    test_dataset = [test_data, test_labels]

    return train_dataset, validation_dataset, test_dataset


def normalize(data):
    """
    Normalize data.

    Parameter
    ---------
    data : numpy.ndarray
    """
    normalized_data = data.astype("float32")
    normalized_data /= 255

    return normalized_data


def train_cnn(run_name, validation_size, epochs, batch_size, n_features, n_hidden, bias_init, learning_rate, seed):
    """
    Train CNN with MNIST data.

    Parameters
    ----------
    run_name : str
        run-name of mlflow
    validation_size : float
    epochs : int
    batch_size : int
    n_fetures : int
        the number of features
    n_hidden : int
        the number of hidden layers
    bias_init : float
        initial bias
    learning_rate : float
    seed : int
    """
    # Set an experiment name
    mlflow.set_experiment("CNN")

    with mlflow.start_run(run_name=run_name) as run:
        # Log automatically
        mlflow.keras.autolog()

        np.random.seed(seed)
        mlflow.log_param("numpy seed", seed)

        train_dataset, validation_dataset, test_dataset = load_normalized_mnist(
            validation_size)
        mlflow.log_param("validation size", validation_size)

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
        model.compile(optimizer=tf.optimizers.Adam(learning_rate),
                      loss="categorical_crossentropy", metrics=["mae", "accuracy"])

        # Fit the model
        log = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=True, callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, verbose=1)], validation_data=(validation_data, validation_labels))

    test_predicted = np.argmax(model.predict(test_data), axis=1)
    precision = sum(test_predicted == test_labels) / len(test_predicted)

    print(f"recision is {precision}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow example program")

    # Add arguments: [https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0]
    parser.add_argument("run_name", type=str, default=None)
    parser.add_argument("validation_size", type=float, default=0.2)
    parser.add_argument("epochs", type=int, default=1000)
    parser.add_argument("batch_size", type=int, default=8)
    parser.add_argument("n_features", type=int, default=784)
    parser.add_argument("n_hidden", type=int, default=100)
    parser.add_argument("bias_init", type=float, default=0.1)
    parser.add_argument("learning_rate", type=float, default=0.01)
    parser.add_argument("seed", type=int, default=57)

    args = parser.parse_args()

    train_cnn(args.run_name, args.validation_size, args.epochs, args.batch_size,
              args.n_features, args.n_hidden, args.bias_init, args.learning_rate, args.seed)
