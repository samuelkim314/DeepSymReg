import numpy as np
import tensorflow as tf
import os


def batch_generator(datasets, N, batch_size=128):
    """Returns a generator that produces shuffled batches of the data by calling next().

    Arguments:
        datasets: List of data that should be batched. The first dimension corresponds to the datapoints, and the sizes
                    should match
        N:  Size of data
        batch_size:     Batch size

    Returns:
        Generator that yields tuples of batched data
    """
    batch_ind = range(0, N, batch_size)
    while True:
        i_arr = np.arange(0, N)
        np.random.shuffle(i_arr)
        data_shuffled = [data_i[i_arr] for data_i in datasets]
        for i_batch in batch_ind:
            yield [data_i[i_batch:i_batch + batch_size] for data_i in data_shuffled]


class Encoder:
    """Encodes dynamics and produces a single-dimensional latent variable.
    Consists of convolutional layers followed by fully-connected layers and batch norm."""
    def __init__(self, n_filters=None, n_dense=None):
        self.input = None
        self.output = None
        self.bn = None
        self.n_filters = [16, 16] if n_filters is None else n_filters
        self.n_dense = [16] if n_dense is None else n_dense

    def __call__(self, input, training):
        with tf.name_scope("encoder"):
            self.input = input
            h = self.input
            for n in self.n_filters:
                h = tf.keras.layers.Conv1D(filters=n, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(h)
            h = tf.keras.layers.Flatten()(h)
            for n in self.n_dense:
                h = tf.keras.layers.Dense(units=n, activation=tf.nn.relu)(h)
            h = tf.keras.layers.Dense(units=1)(h)
            bn = tf.keras.layers.BatchNormalization()
            h = bn(h, training=training) / 2
            self.bn = bn
            self.output = h
            return self.output


def get_trial_path(path):
    """Get the path for the directory to save the trial results in.

    Arguments:
        path: parent directory in which to save trial results.
    Returns:
        path for the trial results.
    """
    # Create the results directory if it doesn't already exist
    if not os.path.exists(path):
        os.makedirs(path)
    # Because TensorFlow models cannot be saved in an existing directory, we need to iterate to find a new directory
    # in which to save the model
    dir_format = os.path.join(path, 'trial%d')
    i = 0
    while True:
        if os.path.isdir(dir_format % i):
            i += 1
        else:
            break
    return dir_format % i
