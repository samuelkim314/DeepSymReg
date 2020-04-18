import tensorflow as tf
import numpy as np
import os
import pickle
from utils import helpers
import argparse


class Propagator:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, input):
        self.input = input
        h = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(input)
        h = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=1)(h)
        self.output = h
        return self.output


def main(results_dir='results/sho/test', trials=1, learning_rate=1e-4, timesteps=25, batch_size=128, n_epochs=20000):
    # Hyperparameters
    summary_step = 2000

    # Import parabola data
    data = np.load('dataset/sho.npz')
    x_d = np.asarray(data["x_d"])
    x_v = np.asarray(data["x_v"])
    y_d = np.asarray(data["y_d"])
    y_v = np.asarray(data["y_v"])
    omega2_data = data["omega2"]
    N = data["N"]

    # Prepare data
    x = np.stack((x_d, x_v), axis=2)    # Shape (N, NT, 2)
    y0 = np.stack((y_d[:, 0], y_v[:, 0]), axis=1)   # Initial conditions for prediction y, fed into propagator
    y_data = np.stack((y_d[:, 1:timesteps + 1], y_v[:, 1:timesteps + 1]), axis=2)     # shape(NT, timesteps, 2)

    # Tensorflow placeholders for x, y0, y
    x_input = tf.placeholder(shape=(None, x.shape[1], x.shape[2]), dtype=tf.float32, name="enc_input")
    y0_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="prop_input")  # input is d, v
    y_input = tf.placeholder(shape=(None, timesteps, 2), dtype=tf.float32, name="label_input")

    # Dynamics encoder
    encoder = helpers.Encoder()
    training = tf.placeholder_with_default(False, [])
    enc_output = encoder(x_input, training=training)
    z_input = tf.placeholder(shape=(None, 1), dtype=tf.float32)     # For when we want to bypass encoder
    z_data = omega2_data[:, np.newaxis]
    # enc_output = z_input  # Uncomment to bypass encoder

    # Propagating decoders
    prop_d = Propagator()
    prop_v = Propagator()

    rec_input = [y0_input]
    for i in range(timesteps):
        full_input = tf.concat([rec_input[i], enc_output, tf.ones_like(enc_output)], axis=1, name="full_input")  # d, v, z1, 1
        rec_input.append(tf.concat([prop_d(full_input), prop_v(full_input)], axis=1, name="c_prop_input"))
    y_hat = tf.stack(rec_input[1:], axis=1)  # Ignore initial conditions

    # Training
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    loss = tf.losses.mean_squared_error(labels=y_input, predictions=y_hat)
    train = tf.group([opt.minimize(loss), encoder.bn.updates])

    batch = helpers.batch_generator([x, y_data, y0, z_data], N, batch_size=batch_size)

    # Training session
    with tf.Session() as sess:
        for _ in range(trials):
            loss_i = np.nan

            while np.isnan(loss_i):
                loss_list = []
                sess.run(tf.global_variables_initializer())

                for i in range(n_epochs):
                    x_batch, y_batch, y0_batch, z_batch = next(batch)
                    feed_dict = {x_input: x_batch, y0_input: y0_batch, y_input: y_batch, z_input: z_batch, training: True}
                    _ = sess.run(train, feed_dict=feed_dict)
                    if i % summary_step == 0 or i == n_epochs - 1:
                        # print(sess.run(y_hat, feed_dict=feed_dict)[0])
                        feed_dict[training] = False
                        loss_i, z_arr = sess.run((loss, enc_output), feed_dict=feed_dict)
                        r = np.corrcoef(z_batch[:, 0], z_arr[:, 0])[1, 0]   # Correlation coefficient
                        loss_list.append(loss_i)
                        print("Epoch %d\tTotal loss: %f\tCorrelation: %f"
                              % (i, loss_i, r))
                        if np.isnan(loss_i):
                            break

            print("Done. Saving results.")

            # Save results
            results = {
                "summary_step": summary_step,
                "learning_rate": learning_rate,
                "n_epochs": n_epochs,
                "timesteps": timesteps,
                "loss_plot": loss_list,
            }

            trial_dir = helpers.get_trial_path(results_dir)  # Get directory in which to save trial results

            tf.saved_model.simple_save(sess, trial_dir, inputs={"x": x_input, "y0": y0_input, "training": training},
                                       outputs={"z": enc_output, "y": y_hat})

            # Save a summary of the parameters and results
            with open(os.path.join(trial_dir, 'summary.pickle'), "wb+") as f:
                pickle.dump(results, f)

            with open(os.path.join(results_dir, 'summary.txt'), 'a') as f:
                f.write("Error: %f\n\n" % loss_list[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on simple harmonic oscillator (SHO) task.")
    parser.add_argument("--results-dir", type=str, default='results/sho/test_relu')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Base learning rate for training')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=20000, help="Number of epochs to train in 1st stage")
    parser.add_argument("--timesteps", type=int, default=25, help="Number of time steps to predict")
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

    main(**kwargs)
