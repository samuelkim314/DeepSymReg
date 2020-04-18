"""Use a decoder-propagating encoder architecture to predict kinematics dynamics. The propagating decoder is
a fully-connected network (not symbolic regression)."""
import tensorflow as tf
import numpy as np
import os
import pickle
from utils import helpers
import argparse


def main(results_dir='results/kinematics/test', learning_rate=1e-3, n_epochs=5001, timesteps=5):
    # Hyperparameters
    summary_step = 1000

    class Propagator:
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

    # Import parabola data
    data = np.load('dataset/kinematic.npz')
    x_d = np.asarray(data["x_d"])
    x_v = np.asarray(data["x_v"])
    y_d = np.asarray(data["y_d"])
    y_v = np.asarray(data["y_v"])
    a_data = np.asarray(data["g"])

    # Prepare data
    # The first few time steps are reserved for the symbolic regression propagator
    x = np.stack((x_d, x_v), axis=2)     # Data fed into the encoder
    y0 = np.stack((y_d[:, 0], y_v[:, 0]), axis=1)  # Input into the symbolic propagator
    y_data = np.stack((y_d[:, 1:timesteps + 1], y_v[:, 1:timesteps + 1]), axis=2)     # shape(NG, LENGTH, 2)

    # Encoder
    enc = helpers.Encoder()     # layer should end with 1, which is the output
    x_input = tf.placeholder(shape=(None, x.shape[1], x.shape[2]), dtype=tf.float32, name="enc_input")
    training = tf.placeholder_with_default(False, [])
    z = enc(x_input, training=training)

    prop_d = Propagator()
    prop_v = Propagator()
    prop_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="prop_input")  # input is d, v

    rec_input = [prop_input]
    for i in range(timesteps):
        full_input = tf.concat([rec_input[i], z], axis=1, name="full_input")    # d, v, z
        rec_input.append(tf.concat([prop_d(full_input), prop_v(full_input)], axis=1, name="c_prop_input"))
    y_hat = tf.stack(rec_input[1:], axis=1)   # Ignore initial conditions

    # Label and errors
    y = tf.placeholder(shape=(None, timesteps, 2), dtype=tf.float32, name="label_input")
    loss = tf.losses.mean_squared_error(labels=y, predictions=y_hat)

    # Training
    learning_rate_ph = tf.placeholder(tf.float32)
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ph)
    train = opt.minimize(loss)

    # Training session
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(n_epochs):
            feed_dict = {x_input: x, prop_input: y0, y: y_data,
                         learning_rate_ph: learning_rate, training: True}
            _ = sess.run(train, feed_dict=feed_dict)
            if i % summary_step == 0:
                loss_i = sess.run(loss, feed_dict=feed_dict)
                loss_list.append(loss_i)
                print(loss_i)

        # Save results
        results = {
            "timesteps": timesteps,
            "summary_step": summary_step,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "loss_list": loss_list,
        }

        trial_dir = helpers.get_trial_path(results_dir)

        tf.saved_model.simple_save(sess, trial_dir,
                                   inputs={"x": x_input, "y0": prop_input, "training": training, "z": z},
                                   outputs={"z": z, "y": y_hat})

        with open(os.path.join(trial_dir, 'summary.pickle'), "wb+") as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on kinematics task.")
    parser.add_argument("--results-dir", type=str, default='results/kinematics/test_relu')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Base learning rate for training')
    parser.add_argument("--n-epochs", type=int, default=10001, help="Number of epochs to train in each stage")
    parser.add_argument("--timesteps", type=int, default=5, help="Number of epochs to train in each stage")
    parser.add_argument('--trials', type=int, default=1, help="Number of trials to train.")

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
    del kwargs['trials']

    for _ in range(trials):
        main(**kwargs)
