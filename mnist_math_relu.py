"""
Convert MNIST digits into a number and perform artihmetic operations on the number(s).
"""

import tensorflow as tf
import numpy as np
import os
import mnist_math
import argparse

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
BATCH_SIZE = 500
n_train = 10000


class AddNet:
    """Conventional fully-connected neural network with ReLU activation functions.
    Replaces the EQL network in the MNIST architecture so that we can compare extrapolation results."""
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, input):
        self.input = input
        h = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(input)
        h = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=1)(h)
        self.output = h
        return self.output


class SymbolicDigit(mnist_math.SymbolicDigit):
    """MNIST architecture but using the AddNet instead of the EQL network.
    Just changes how we save results since we no longer need to save the equation."""
    def save_result(self, sess, results_dir, result_str):
        """Save results to file"""
        # Because tf will not save the model if the directory exists, we append a number to the directory and save. This
        # allows us to run multiple trials
        dir_format = os.path.join(results_dir, "trial%d")
        i = 0
        while True:
            if os.path.isdir(dir_format % i):
                i += 1
            else:
                break

        # Save TensorFlow graph
        input_dict = {"x%d" % i: self.x[i] for i in range(len(self.x))}
        if isinstance(self.encoder, mnist_math.Encoder):
            input_dict['training'] = self.encoder.training
        output_dict = {"z%d" % i: tf.identity(self.z[i], name='z%d' % i) for i in range(len(self.z))}
        output_dict["y_hat"] = self.y_hat
        tf.saved_model.simple_save(sess, dir_format % i, inputs=input_dict, outputs=output_dict)

        # Save test accuracy inside the project directory
        file = open(os.path.join(dir_format, 'equation.txt') % i, 'w+')
        file.write("\n")
        file.write(result_str)
        file.close()

        # Save test accuracy in the higher-level directory
        file = open(os.path.join(results_dir, 'overview.txt'), 'a+')
        file.write('%d\n' % i)
        file.write(result_str)
        file.write("\n")
        file.close()


def train_add(func=lambda a, b: a+b, results_dir=None, learning_rate=1e-2, n_epochs=10001):
    """Addition of two MNIST digits using the AddNet network"""

    sr_net = AddNet()  # Symbolic regression network
    sym_digit_network = SymbolicDigit(sr_net=sr_net, normalize=mnist_math.normalize)    # Overall architecture
    sym_digit_network.set_training()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    sess = tf.Session(config=config)

    batch = mnist_math.batch_generator(batch_size=500)

    # Train
    loss_i = np.nan
    while np.isnan(loss_i):
        sess.run(tf.global_variables_initializer())
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate)
        if np.isnan(loss_i):
            continue
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate/10)

    # Calculate accuracy on test dataset
    acc_test, error_test = sym_digit_network.calc_accuracy(X_test, y_test, func, sess)
    result_str = 'Test accuracy: %g\n' % acc_test
    print(result_str)

    sym_digit_network.save_result(sess, results_dir, result_str)


def train_add_test(func=lambda a, b: a+b, results_dir=None, learning_rate=1e-2, n_epochs=10001):
    """Addition of two MNIST digits using the AddNet network.
    Withhold sums > 15 for test data"""

    sr_net = AddNet()  # Symbolic regression network
    sym_digit_network = SymbolicDigit(sr_net=sr_net, normalize=mnist_math.normalize)    # Overall architecture
    sym_digit_network.set_training()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    sess = tf.Session(config=config)

    batch = mnist_math.batch_generator(batch_size=100)

    def train_fun(y):
        return y < 15
        # return y % 2 == 0
        # return np.logical_or(y == 5, y == 15)

    def test_fun(y):
        return np.logical_not(train_fun(y))

    # Train, and restart training if loss goes to NaN
    loss_i = np.nan
    while np.isnan(loss_i):
        sess.run(tf.global_variables_initializer())
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate, train_fun=train_fun)
        if np.isnan(loss_i):
            continue
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate/10, train_fun=train_fun)

    # Calculate accuracy on test dataset
    acc_train, error_train = sym_digit_network.calc_accuracy(X_train, y_train, func, sess)
    acc_train1, error_train1 = sym_digit_network.calc_accuracy(X_train, y_train, func, sess, filter_fun=train_fun)
    acc_train2, error_train2 = sym_digit_network.calc_accuracy(X_train, y_train, func, sess, filter_fun=test_fun)
    acc_test, error_test = sym_digit_network.calc_accuracy(X_test, y_test, func, sess)
    acc_test1, error_test1 = sym_digit_network.calc_accuracy(X_test, y_test, func, sess, filter_fun=train_fun)
    acc_test2, error_test2 = sym_digit_network.calc_accuracy(X_test, y_test, func, sess, filter_fun=test_fun)
    result_str = "Train digits overall accuracy: %.3f\ttrain sum accuracy: %.3f\t test sum accuracy: %.3f\n" \
                 "Train digits overall error: %.3f\ttrain sum error: %.3f\t test sum error: %.3f\n" \
                 "Test digits overall accuracy: %.3f\ttrain sum accuracy: %.3f\t test sum accuracy: %.3f\n" \
                 "Test digits overall error: %.3f\ttrain sum error: %.3f\t test sum error: %.3f\n" % \
                 (acc_train, acc_train1, acc_train2, error_train, error_train1, error_train2,
                  acc_test, acc_test1, acc_test2, error_test, error_test1, error_test2)
    print(result_str)

    sym_digit_network.save_result(sess, results_dir, result_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on MNIST arithmetic task.")
    parser.add_argument("--results-dir", type=str, default='results/mnist/test_relu')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument("--n-epochs", type=int, default=10001, help="Number of epochs to train in each stage")
    parser.add_argument('--trials', type=int, default=1, help="Number of trials to train.")
    parser.add_argument('--filter', action='store_true', help="Train only on y<15 data.")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    import json

    meta.write(json.dumps(kwargs))
    meta.close()

    trials = kwargs['trials']
    use_filter = kwargs['filter']
    del kwargs['trials']
    del kwargs['filter']

    if use_filter:
        for _ in range(trials):
            train_add_test(**kwargs)
    else:
        for _ in range(trials):
            train_add(**kwargs)
