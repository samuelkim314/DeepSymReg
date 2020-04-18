import tensorflow as tf
import numpy as np
import os
import pickle
from utils.symbolic_network import SymbolicNet, MaskedSymbolicNet, SymbolicCell
from utils import functions, regularization, helpers, pretty_print
import argparse


def main(results_dir='results/sho/test', trials=1, learning_rate=1e-2, reg_weight=2e-4, timesteps=25, batch_size=129,
         n_epochs1=2001, n_epochs2=5001, n_epochs3=5001):
    # Hyperparameters
    summary_step = 500
    timesteps0 = 1

    primitive_funcs = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        *[functions.Product(norm=0.1)] * 2,
    ]

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
    y_data = np.stack((y_d[:, 1:timesteps + 1], y_v[:, 1:timesteps + 1]), axis=2)     # shape(NG, LENGTH, 2)

    # Tensorflow placeholders for x, y0, y
    x_input = tf.placeholder(shape=(None, x.shape[1], x.shape[2]), dtype=tf.float32, name="enc_input")
    y0_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="prop_input")  # input is d, v
    y_input = tf.placeholder(shape=(None, timesteps, 2), dtype=tf.float32, name="label_input")
    length_input = tf.placeholder(dtype=tf.int32, shape=())

    # Dynamics encoder
    encoder = helpers.Encoder()
    training = tf.placeholder_with_default(False, [])
    z = encoder(x_input, training=training)
    z_data = omega2_data[:, np.newaxis]

    # Propagating decoders
    prop_d = SymbolicNet(2, funcs=primitive_funcs)
    prop_v = SymbolicNet(2, funcs=primitive_funcs)
    prop_d.build(4)
    prop_v.build(4)
    # Building recurrent structure
    rnn = tf.keras.layers.RNN(SymbolicCell(prop_d, prop_v), return_sequences=True)
    y0_rnn = tf.concat([tf.expand_dims(y0_input, axis=1), tf.zeros((tf.shape(y0_input)[0], length_input - 1, 2))],
                       axis=1)
    prop_input = tf.concat([y0_rnn, tf.keras.backend.repeat(z, length_input),
                            tf.ones((tf.shape(y0_input)[0], length_input, 1))], axis=2)
    prop_output = rnn(prop_input)

    epoch = tf.placeholder(tf.float32)
    reg_freq = np.pi / (n_epochs1 + n_epochs2) / 1.1
    reg_loss = tf.sin(reg_freq * epoch) ** 2 * regularization.l12_smooth(prop_d.get_weights()) + \
               tf.sin(reg_freq * epoch) ** 2 * regularization.l12_smooth(prop_v.get_weights())
    # reg_loss = regularization.l12_smooth(prop_d.get_weights()) + regularization.l12_smooth(prop_v.get_weights())

    # Training
    learning_rate_ph = tf.placeholder(tf.float32)
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ph)
    reg_weight_ph = tf.placeholder(tf.float32)
    error = tf.losses.mean_squared_error(labels=y_input[:, :length_input, :], predictions=prop_output)
    loss = error + reg_weight_ph * reg_loss
    train = tf.group([opt.minimize(loss), encoder.bn.updates])

    batch = helpers.batch_generator([x, y_data, y0, z_data], N=N, batch_size=batch_size)

    # Training session
    with tf.Session() as sess:
        for _ in range(trials):
            loss_i = np.nan

            while np.isnan(loss_i):
                loss_list = []
                error_list = []
                reg_list = []

                sess.run(tf.global_variables_initializer())

                for i in range(n_epochs1 + n_epochs2):
                    if i < n_epochs1:
                        reg_weight_i = reg_weight / 5
                        learning_rate_i = learning_rate
                        length_i = min(i // 500 * 2 + timesteps0, timesteps)
                    else:
                        reg_weight_i = reg_weight
                        learning_rate_i = learning_rate / 5
                        length_i = timesteps

                    x_batch, y_batch, y0_batch, z_batch = next(batch)
                    feed_dict = {x_input: x_batch, y0_input: y0_batch, y_input: y_batch,
                                 epoch: i, learning_rate_ph: learning_rate_i, training: True,
                                 reg_weight_ph: reg_weight_i, length_input: length_i}
                    _ = sess.run(train, feed_dict=feed_dict)

                    if i % summary_step == 0 or i == n_epochs1 - 1:
                        feed_dict[training] = False
                        loss_i, error_i, reg_i = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                        z_arr = sess.run(z, feed_dict=feed_dict)
                        r = np.corrcoef(z_batch[:, 0], z_arr[:, 0])[1, 0]
                        loss_list.append(loss_i)
                        error_list.append(error_i)
                        reg_list.append(reg_i)
                        print("Epoch %d\tTotal loss: %f\tError: %f\tReg loss: %f\tCorrelation: %f"
                              % (i, loss_i, error_i, reg_i, r))
                        if np.isnan(loss_i):
                            break

            # Setting small weights to 0 and freezing them
            prop_d_masked = MaskedSymbolicNet(sess, prop_d, threshold=0.01)
            prop_v_masked = MaskedSymbolicNet(sess, prop_v, threshold=0.01)
            # Keep track of currently existing variables. When we rebuild the rnn, it makes new variables that we need
            # to initialize. Later, we will use this to figure out what the uninitialized variables are.
            temp = set(tf.global_variables())
            # Rebuilding the decoding propagator. Remove regularization
            rnn = tf.keras.layers.RNN(SymbolicCell(prop_d_masked, prop_v_masked), return_sequences=True)
            prop_output = rnn(prop_input)
            loss = tf.losses.mean_squared_error(labels=y_input[:, :length_input, :], predictions=prop_output)
            train = tf.group([opt.minimize(loss), encoder.bn.updates])

            weights_d = sess.run(prop_d_masked.get_weights())
            expr_d = pretty_print.network(weights_d, primitive_funcs, ["d", "v", "z", 1])
            print(expr_d)
            weights_v = sess.run(prop_v_masked.get_weights())
            expr_v = pretty_print.network(weights_v, primitive_funcs, ["d", "v", "z", 1])
            print(expr_v)

            print("Frozen weights. Next stage of training.")

            # Initialize only the uninitialized variables.
            sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))

            for i in range(n_epochs3):
                x_batch, y_batch, y0_batch, z_batch = next(batch)
                feed_dict = {x_input: x_batch, y0_input: y0_batch, y_input: y_batch,
                             epoch: 0, learning_rate_ph: learning_rate / 10, training: True, reg_weight_ph: 0,
                             length_input: length_i}
                _ = sess.run(train, feed_dict=feed_dict)
                if i % summary_step == 0:
                    feed_dict[training] = False
                    loss_i, error_i, reg_i = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                    z_arr = sess.run(z, feed_dict=feed_dict)
                    r = np.corrcoef(z_batch[:, 0], z_arr[:, 0])[1, 0]
                    loss_list.append(loss_i)
                    error_list.append(error_i)
                    reg_list.append(reg_i)
                    print("Epoch %d\tError: %g\tCorrelation: %f" % (i, error_i, r))

            weights_d = sess.run(prop_d_masked.get_weights())
            expr_d = pretty_print.network(weights_d, primitive_funcs, ["d", "v", "z", 1])
            print(expr_d)
            weights_v = sess.run(prop_v_masked.get_weights())
            expr_v = pretty_print.network(weights_v, primitive_funcs, ["d", "v", "z", 1])
            print(expr_v)

            # Save results
            results = {
                "summary_step": summary_step,
                "learning_rate": learning_rate,
                "n_epochs1": n_epochs1,
                "n_epochs2": n_epochs2,
                "reg_weight": reg_weight,
                "timesteps": timesteps,
                "timesteps0": timesteps0,
                "weights_d": weights_d,
                "weights_v": weights_v,
                "loss_plot": loss_list,
                "error_plot": error_list,
                "reg_plot": reg_list,
                "expr_d": expr_d,
                "expr_v": expr_v
            }

            trial_dir = helpers.get_trial_path(results_dir)  # Get directory in which to save trial results

            tf.saved_model.simple_save(sess, trial_dir,
                                       inputs={"x": x_input, "y0": y0_input, "training": training},
                                       outputs={"z": z, "y": prop_output})

            # Save a summary of the parameters and results
            with open(os.path.join(trial_dir, 'summary.pickle'), "wb+") as f:
                pickle.dump(results, f)

            with open(os.path.join(results_dir, 'eq_summary.txt'), 'a') as f:
                f.write(str(expr_d) + "\n")
                f.write(str(expr_v) + "\n")
                f.write("Error: %f\n\n" % error_list[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on simple harmonic oscillator (SHO) task.")
    parser.add_argument("--results-dir", type=str, default='results/sho/test')
    parser.add_argument("--reg-weight", type=float, default=2e-4, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument("--n-epochs1", type=int, default=2001, help="Number of epochs to train in 1st stage")
    parser.add_argument("--n-epochs2", type=int, default=5001, help="Number of epochs to train in 2nd stage")
    parser.add_argument("--n-epochs3", type=int, default=5001, help="Number of epochs to train in 3rd stage")
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
