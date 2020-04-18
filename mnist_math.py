"""
Train a network to perform arithmetic operations on MNIST numbers.
EQL Network is used to back out the operation.
"""

import tensorflow as tf
import numpy as np
import os
from utils import functions, regularization, symbolic_network, pretty_print, helpers
import argparse

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
BATCH_SIZE = 100    # BATCH_SIZE*2 needs to divide evenly into n_train
n_train = 10000     # Size of MNIST training dataset


def batch_generator(batch_size=BATCH_SIZE):
    return helpers.batch_generator((X_train, y_train), n_train, batch_size)


def normalize(y):
    return 9 * y + 9


class Encoder:
    """Network that takes in MNIST digit and produces a single-dimensional latent variable.
    Consists of several convolution layers followed by fully-connected layers"""
    def __init__(self, training):
        """
        Arguments:
            training: Boolean of whether to use training mode or not. Matters for Batch norm layer
        """
        self.y_hat = None
        self.training = training
        self.bn = None

    def build(self, x, n_latent=1, name='y_hat'):
        """Convolutional and fully-connected layers to extract a latent variable from an MNIST image

        Arguments:
            x: Batch of MNIST images with dimension (n_batch, width, height)
            n_latent: Dimension of latent variable
            name: TensorFlow name of output
        """
        h = tf.expand_dims(x, axis=-1)  # Input to Conv2D needs dimension (n_batch, width, height, n_channels)
        h = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(h)
        h = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(h)
        h = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(h)
        h = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(h)
        h = tf.keras.layers.Flatten()(h)
        # h = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=n_latent, name=name)(h)
        self.bn = tf.keras.layers.BatchNormalization()
        # Divide by 2 to make std dev close to 0.5 because distribution is uniform
        h = self.bn(h, training=self.training) / 2
        y_hat = tf.identity(h, name=name)
        return y_hat

    def __call__(self, x, n_latent=1, name='y_hat'):
        if self.y_hat is None:
            self.y_hat = self.build(x, n_latent, name)
        return self.y_hat


class SymbolicDigit:
    """Architecture for MNIST arithmetic. Takes care of initialization, training, and saving."""
    def __init__(self, sr_net, x=None, encoder=None, normalize=None):
        """Set up the MNIST arithmetic architecture

        Arguments:
            sr_net: EQL Network, SymbolicNet instance
        """

        n_digits = 2    # Number of inputs to the arithmetic function.
        if x is None:
            x1 = tf.placeholder(tf.float32, [None, 28, 28], name='x1')
            x2 = tf.placeholder(tf.float32, [None, 28, 28], name='x2')
            x = [x1, x2]
        self.x = x

        # Encoder for each MNIST digit into latent variable (conv layers with batch norm at output)
        if encoder is None:
            self.training = tf.placeholder_with_default(True, [])
            encoder = Encoder(self.training)
        else:
            self.training = encoder.training
        self.encoder = encoder

        # We want to feed multiple digits into the same CNN, so we flatten the input first and then reshape the output
        x_full = tf.stack(self.x)    # shape = (n_digits, batch_size, 28, 28)
        batch_size = tf.shape(x_full)[1]
        x_flat = tf.reshape(x_full, [n_digits*batch_size, 28, 28])  # Flatten to (n_digits*batch_size, 28, 28)
        z_flat = self.encoder(x_flat)     # shape = (n_digits*batch_size, 1)
        z = tf.reshape(z_flat, [n_digits, batch_size])
        self.z = tf.unstack(z, axis=0, name='z')  # List of size n_digits. This gets saved.
        z = tf.transpose(z)     # reshape to shape = (batch_size, n_digits)
        self.y_hat = tf.squeeze(sr_net(z))
        if normalize is not None:
            self.y_hat = normalize(self.y_hat)

        self.y_ = tf.placeholder(tf.float32, [None])  # Placeholder for true labels
        self.lr = tf.placeholder(tf.float32)  # Learning rate of gradient descent
        self.loss = None
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.trainer = None

        self.reg = tf.constant(0.0)
        self.loss_total = None

        correct_prediction = tf.equal(tf.round(self.y_hat), tf.round(self.y_))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.error_avg = tf.reduce_mean(tf.abs(self.y_hat - self.y_))

    def set_training(self, reg=0):
        """Set up the remainder of the Tensorflow graph for training. Call set_reg before this. This must be called
        before training the network."""
        self.loss = tf.losses.mean_squared_error(self.y_, self.y_hat)
        self.reg = reg
        self.loss_total = self.loss + reg

        self.trainer = self.optimizer.minimize(self.loss_total)
        self.trainer = tf.group([self.trainer, self.encoder.bn.updates])

    def save_result(self, sess, results_dir, eq, result_str):
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
        if isinstance(self.encoder, Encoder):
            input_dict['training'] = self.encoder.training
        output_dict = {"z%d" % i: tf.identity(self.z[i], name='z%d' % i) for i in range(len(self.z))}
        output_dict["y_hat"] = self.y_hat
        tf.saved_model.simple_save(sess, dir_format % i, inputs=input_dict, outputs=output_dict)

        # Save equation and test accuracy inside the project directory
        file = open(os.path.join(dir_format, 'equation.txt') % i, 'w+')
        file.write(str(eq))
        file.write("\n")
        file.write(result_str)
        file.close()

        # Save equation and test accuracy in the higher-level directory
        file = open(os.path.join(results_dir, 'overview.txt'), 'a+')
        file.write('%d: \t%s\n' % (i, eq))
        file.write(result_str)
        file.write("\n")
        file.close()

    def train(self, sess, n_epochs, batch, func, epoch=None, lr_val=1e-3, train_fun=None):
        """Training step of the digit extractor + symbolic network"""

        if epoch is None:
            epoch = tf.placeholder_with_default(0.0, [])    # dummy variable

        loss_i = None
        for i in range(n_epochs):
            batch_x1, batch_y1 = next(batch)
            batch_x2, batch_y2 = next(batch)
            batch_y = func(batch_y1, batch_y2)

            # Filtering out the batch. This lets us train on a subset of data (e.g. y<15) and then test
            # on the rest of the data (e.g. y>=15) to evaluate extrapolation
            if train_fun is not None:
                ind_train = train_fun(batch_y)  # Indices for data matching the condition
                batch_x1 = batch_x1[ind_train]
                batch_x2 = batch_x2[ind_train]
                batch_y = batch_y[ind_train]

            if i % 1000 == 0:
                train_accuracy, loss_i, reg_i = \
                    sess.run((self.accuracy, self.loss_total, self.reg),
                             feed_dict={self.x[0]: batch_x1, self.x[1]: batch_x2, self.y_: batch_y,
                                        epoch: i, self.training: False})
                print("Step %d\t Training accuracy %.3f\tFit loss %.3f\tReg loss %.3f" %
                      (i, train_accuracy, loss_i-reg_i, reg_i))

                if np.isnan(loss_i):  # If loss goes to NaN, restart training
                    break

            sess.run(self.trainer, feed_dict={self.x[0]: batch_x1, self.x[1]: batch_x2, self.y_: batch_y,
                                              self.lr: lr_val, epoch: i, self.encoder.training: True})
        return loss_i

    def calc_accuracy(self, X, y, func, sess, filter_fun=None):
        """Calculate accuracy over a given dataset"""
        # Grab test data, split it into two halves, and then apply function to y-values to create new dataset
        n_test = y.shape[0]
        n2 = int(n_test / 2)
        X1 = X[:n2, :, :]
        X2 = X[n2:, :, :]
        y1 = y[:n2]
        y2 = y[n2:]
        y_test = func(y1, y2)

        # To calculate test accuracy, we split it up into batches to avoid overflowing the memory
        acc_batch = []
        error_batch = []
        batch_test_ind = range(0, n2, BATCH_SIZE)
        for i_batch in batch_test_ind:
            X_batch1 = X1[i_batch:i_batch + BATCH_SIZE]
            X_batch2 = X2[i_batch:i_batch + BATCH_SIZE]
            Y_batch = y_test[i_batch:i_batch + BATCH_SIZE]

            # Only pick out data (X1, X2, y) that match a condition on y.
            if filter_fun is not None:
                ind_train = filter_fun(Y_batch)
                X_batch1 = X_batch1[ind_train]
                X_batch2 = X_batch2[ind_train]
                Y_batch = Y_batch[ind_train]

            acc_i, error_i = sess.run((self.accuracy, self.error_avg),
                                      feed_dict={self.x[0]: X_batch1, self.x[1]: X_batch2, self.y_: Y_batch,
                                                 self.training: False})
            acc_batch.append(acc_i)
            error_batch.append(error_i)
        acc_total = np.mean(acc_batch)
        error_total = np.mean(error_batch)

        return acc_total, error_total

    @staticmethod
    def normalize(y):
        return y * 9 + 9


class SymbolicDigitMasked(SymbolicDigit):
    def __init__(self, sym_digit_net, sr_net_masked, normalize=None):
        super().__init__(sr_net_masked, x=sym_digit_net.x, encoder=sym_digit_net.encoder, normalize=normalize)
        self.sym_digit_net = sym_digit_net

        self.y_ = sym_digit_net.y_  # Placeholder for true labels
        self.lr = sym_digit_net.lr  # Learning rate of gradient descent

    def set_training(self, reg=0):
        """Set up the remainder of the Tensorflow graph for training. Call set_reg before this. This must be called
        before training the network."""
        self.loss = tf.losses.mean_squared_error(self.y_, self.y_hat)
        self.loss_total = self.loss + self.reg + reg

        self.trainer = self.sym_digit_net.optimizer.minimize(self.loss_total)
        self.trainer = tf.group([self.trainer, self.encoder.bn.updates])

        correct_prediction = tf.equal(tf.round(self.y_hat), tf.round(self.y_))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.error_avg = tf.reduce_mean(tf.abs(self.y_hat - self.y_))


def train_add(func=lambda a, b: a + b, results_dir=None, reg_weight=5e-2, learning_rate=1e-2, n_epochs=10001):
    """Addition of two MNIST digits with a symbolic regression network."""
    tf.reset_default_graph()

    # Symbolic regression network to combine the conv net outputs
    PRIMITIVE_FUNCS = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        *[functions.Product()] * 2,
    ]
    sr_net = symbolic_network.SymbolicNet(2, funcs=PRIMITIVE_FUNCS, init_stddev=0.1)  # Symbolic regression network
    # Overall architecture
    sym_digit_network = SymbolicDigit(sr_net=sr_net, normalize=normalize)
    # Set up regularization term and training
    penalty = regularization.l12_smooth(sr_net.get_weights())
    penalty = reg_weight * penalty
    sym_digit_network.set_training(reg=penalty)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    sess = tf.Session(config=config)

    batch = batch_generator(batch_size=100)

    # Train, and restart training if loss goes to NaN
    loss_i = np.nan
    while np.isnan(loss_i):
        sess.run(tf.global_variables_initializer())
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate)
        if np.isnan(loss_i):
            continue

        # Freezing weights
        sr_net = symbolic_network.MaskedSymbolicNet(sess, sr_net, threshold=0.01)
        sym_digit_network = SymbolicDigitMasked(sym_digit_network, sr_net, normalize=normalize)
        sym_digit_network.set_training()

        # Training with frozen weights. Regularization is 0
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate/10)

    # Print out human-readable equation (with regularization)
    weights = sess.run(sr_net.get_weights())
    expr = pretty_print.network(weights, PRIMITIVE_FUNCS, ["z1", "z2"])
    expr = normalize(expr)
    print(expr)

    # Calculate accuracy on test dataset
    acc_test, error_test = sym_digit_network.calc_accuracy(X_test, y_test, func, sess)
    result_str = 'Test accuracy: %g\n' % acc_test
    print(result_str)

    sym_digit_network.save_result(sess, results_dir, expr, result_str)


def train_add_l0(func=lambda a, b: a+b, results_dir=None, reg_weight=5e-2, learning_rate=1e-2, n_epochs=10001):
    """Addition of two MNIST digits with a symbolic regression network. Uses L0 regularizatoin"""
    tf.reset_default_graph()

    # EQL network to combine the conv net outputs
    PRIMITIVE_FUNCS = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        *[functions.Product()] * 2,
    ]
    sr_net = symbolic_network.SymbolicNetL0(2, funcs=PRIMITIVE_FUNCS, init_stddev=0.5)  # Symbolic regression network

    # Overall architecture
    sym_digit_network = SymbolicDigit(sr_net=sr_net, normalize=normalize)

    # Set up regularization term and training
    penalty = reg_weight * sr_net.get_loss()
    sym_digit_network.set_training(reg=penalty)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    sess = tf.Session(config=config)

    batch = batch_generator(batch_size=100)

    # Train, and restart training if loss goes to NaN
    loss_i = np.nan
    while np.isnan(loss_i):
        sess.run(tf.global_variables_initializer())
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate)
        if np.isnan(loss_i):
            continue
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate/10)
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate/100)

    # Print out human-readable equation
    weights = sess.run(sr_net.get_weights())
    expr = pretty_print.network(weights, PRIMITIVE_FUNCS, ["z1", "z2"])
    expr = normalize(expr)
    print(expr)

    # Calculate accuracy on test dataset
    acc_test, error_test = sym_digit_network.calc_accuracy(X_test, y_test, func, sess)
    result_str = 'Test accuracy: %g\n' % acc_test
    print(result_str)

    sym_digit_network.save_result(sess, results_dir, expr, result_str)


def train_add_test(func=lambda a, b: a+b, results_dir=None, reg_weight=5e-2, learning_rate=1e-2, n_epochs=10001):
    """Addition of two MNIST digits with a symbolic regression network.
    Withold sums > 15 for test data"""
    tf.reset_default_graph()

    # Symbolic regression network to combine the conv net outputs
    PRIMITIVE_FUNCS = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        # *[functions.Product()] * 2,
    ]
    sr_net = symbolic_network.SymbolicNet(2, funcs=PRIMITIVE_FUNCS)  # Symbolic regression network
    # Overall architecture
    sym_digit_network = SymbolicDigit(sr_net=sr_net, normalize=normalize)
    # Set up regularization term and training
    penalty = regularization.l12_smooth(sr_net.get_weights())

    epoch = tf.placeholder_with_default(0.0, [])
    penalty = tf.sin(np.pi / n_epochs / 1.1 * epoch) ** 2 * regularization.l12_smooth(sr_net.get_weights())
    penalty = reg_weight * penalty
    sym_digit_network.set_training(reg=penalty)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    sess = tf.Session(config=config)

    batch = batch_generator(batch_size=100)

    def train_fun(y):
        return y < 15

    def test_fun(y):
        return np.logical_not(train_fun(y))

    # Train, and restart training if loss goes to NaN
    loss_i = np.nan
    while np.isnan(loss_i):
        sess.run(tf.global_variables_initializer())
        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, epoch, lr_val=learning_rate, train_fun=train_fun)
        if np.isnan(loss_i):
            continue

        # Freezing weights
        sr_net_masked = symbolic_network.MaskedSymbolicNet(sess, sr_net, threshold=0.01)
        sym_digit_network = SymbolicDigitMasked(sym_digit_network, sr_net_masked, normalize=normalize)
        sym_digit_network.set_training()

        loss_i = sym_digit_network.train(sess, n_epochs, batch, func, lr_val=learning_rate/10, train_fun=train_fun)

    # Print out human-readable equation (with regularization)
    weights = sess.run(sr_net.get_weights())
    expr = pretty_print.network(weights, PRIMITIVE_FUNCS, ["z1", "z2"])
    expr = normalize(expr)
    print(expr)

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

    sym_digit_network.save_result(sess, results_dir, expr, result_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on MNIST arithmetic task.")
    parser.add_argument("--results-dir", type=str, default='results/mnist/test')
    parser.add_argument("--reg-weight", type=float, default=5e-2, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument("--n-epochs", type=int, default=10001, help="Number of epochs to train in each stage")
    parser.add_argument('--trials', type=int, default=1, help="Number of trials to train.")
    parser.add_argument('--l0', action='store_true', help="Use relaxed L0 regularization instead of L0.5")
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
    use_l0 = kwargs['l0']
    use_filter = kwargs['filter']
    del kwargs['trials']
    del kwargs['l0']
    del kwargs['filter']

    if use_l0:
        for _ in range(trials):
            train_add_l0(**kwargs)
    elif use_filter:
        for _ in range(trials):
            train_add_test(**kwargs)
    else:
        for _ in range(trials):
            train_add(**kwargs)
