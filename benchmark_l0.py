"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import pickle
import tensorflow as tf
import numpy as np
import os
from utils import functions, pretty_print
from utils.symbolic_network import SymbolicNetL0
from inspect import signature
import benchmark
import argparse


N_TRAIN = 256       # Size of training dataset
N_VAL = 100         # Size of validation dataset
DOMAIN = (-1, 1)    # Domain of dataset
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])   # Use this format if each input variable has a different domain
N_TEST = 100        # Size of test dataset
DOMAIN_TEST = (-2, 2)   # Domain of test dataset - should be larger than training domain to test extrapolation
NOISE_SD = 0        # Standard deviation of noise for training dataset
var_names = ["x", "y", "z"]

# Standard deviation of random distribution for weight initializations.
init_sd_first = 0.5
init_sd_last = 0.5
init_sd_middle = 0.5


generate_data = benchmark.generate_data


class Benchmark(benchmark.Benchmark):
    """Benchmark object just holds the results directory (results_dir) to save to and the hyper-parameters. So it is
    assumed all the results in results_dir share the same hyper-parameters. This is useful for benchmarking multiple
    functions with the same hyper-parameters."""
    def __init__(self, results_dir, n_layers=2, reg_weight=1e-2, learning_rate=1e-2,
                 n_epochs1=20001, n_epochs2=10001):
        """Set hyper-parameters"""
        self.activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            *[functions.Exp()] * 2,
            *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2
        ]

        self.n_layers = n_layers              # Number of hidden layers
        self.reg_weight = reg_weight     # Regularization weight
        self.learning_rate = learning_rate
        self.summary_step = 1000    # Number of iterations at which to print to screen
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        # Save hyperparameters to file
        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
            pickle.dump(result, f)

    def train(self, func, func_name='', trials=1, func_dir='results/test'):
        """Train the network to find a given function"""

        x_dim = len(signature(func).parameters)  # Number of input arguments to the function
        # Generate training data and test data
        x, y = generate_data(func, N_TRAIN)
        # x_val, y_val = generate_data(func, N_VAL)
        x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])

        # Setting up the symbolic regression network
        x_placeholder = tf.placeholder(shape=(None, x_dim), dtype=tf.float32)
        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)
        sym = SymbolicNetL0(self.n_layers, funcs=self.activation_funcs,
                            initial_weights=[
                                                 tf.truncated_normal([x_dim, width + n_double], stddev=init_sd_first),
                                                 tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                                 tf.truncated_normal([width, width + n_double], stddev=init_sd_middle),
                                                 tf.truncated_normal([width, 1], stddev=init_sd_last)
                                             ], )
        y_hat = sym(x_placeholder)

        # Label and errors
        error = tf.losses.mean_squared_error(labels=y, predictions=y_hat)
        error_test = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat)
        # Regularization oscillates as a function of epoch.
        reg_loss = sym.get_loss()
        loss = error + self.reg_weight * reg_loss

        # Training
        learning_rate = tf.placeholder(tf.float32)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train = opt.minimize(loss)

        loss_list = []  # Total loss (MSE + regularization)
        error_list = []     # MSE
        reg_list = []       # Regularization
        error_test_list = []    # Test error

        error_test_final = []
        eq_list = []

        # Only take GPU memory as needed - allows multiple jobs on a single GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for trial in range(trials):
                print("Training on function " + func_name + " Trial " + str(trial+1) + " out of " + str(trials))

                loss_val = np.nan
                # Restart training if loss goes to NaN (which happens when gradients blow up)
                while np.isnan(loss_val):
                    sess.run(tf.global_variables_initializer())
                    # 1st stage of training with oscillating regularization weight
                    for i in range(self.n_epochs1):
                        feed_dict = {x_placeholder: x, learning_rate: self.learning_rate}
                        _ = sess.run(train, feed_dict=feed_dict)
                        if i % self.summary_step == 0:
                            loss_val, error_val, reg_val, = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                            error_test_val = sess.run(error_test, feed_dict={x_placeholder: x_test})
                            print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (i, loss_val, error_test_val))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            reg_list.append(reg_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                break

                # Print the expressions
                weights = sess.run(sym.get_weights())
                expr = pretty_print.network(weights, self.activation_funcs, var_names[:x_dim])
                print(expr)

                # Save results
                trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)

                results = {
                    "weights": weights,
                    "loss_list": loss_list,
                    "error_list": error_list,
                    "reg_list": reg_list,
                    "error_test": error_test_list,
                    "expr": expr
                }

                with open(trial_file, "wb+") as f:
                    pickle.dump(results, f)

                error_test_final.append(error_test_list[-1])
                eq_list.append(expr)

        return eq_list, error_test_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=1e-2, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=20001, help="Number of epochs to train the first stage")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    import json

    meta.write(json.dumps(kwargs))
    meta.close()

    bench = Benchmark(**kwargs)

    bench.benchmark(lambda x: x, func_name="x", trials=5)
    bench.benchmark(lambda x: x**2, func_name="x^2", trials=20)
    bench.benchmark(lambda x: x**3, func_name="x^3", trials=20)
    bench.benchmark(lambda x: np.sin(2*np.pi*x), func_name="sin(2pix)", trials=20)
    bench.benchmark(lambda x: np.exp(x), func_name="e^x", trials=20)
    bench.benchmark(lambda x, y: x*y, func_name="xy", trials=5)
    bench.benchmark(lambda x, y: np.sin(2 * np.pi * x) + np.sin(4*np.pi * y),
                    func_name="sin(2pix)+sin(2py)", trials=20)
    bench.benchmark(lambda x, y, z: 0.5*x*y + 0.5*z, func_name="0.5xy+0.5z", trials=5)
    bench.benchmark(lambda x, y, z: x**2 + y - 2*z, func_name="x^2+y-2z", trials=20)
    bench.benchmark(lambda x: np.exp(-x**2), func_name="e^-x^2", trials=20)
    bench.benchmark(lambda x: 1 / (1 + np.exp(-10*x)), func_name="sigmoid(10x)", trials=20)
    bench.benchmark(lambda x, y: x**2 + np.sin(2*np.pi*y), func_name="x^2+sin(2piy)", trials=20)
    #
    # # 3-layer functions
    # bench.benchmark(lambda x, y, z: (x + y * z) ** 3, func_name="(x+yz)^3", trials=20)
