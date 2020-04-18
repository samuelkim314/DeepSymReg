import tensorflow as tf
import numpy as np
import os
import pickle
import argparse
from utils.symbolic_network import SymbolicNetL0, SymbolicCell
from utils import functions, pretty_print, helpers


def main(results_dir='results/kinematics/test', learning_rate=1e-2, reg_weight=1e-3, n_epochs=10001,
         timesteps=5):
    tf.reset_default_graph()

    # Hyperparameters
    summary_step = 1000
    # tf.set_random_seed(0)

    # Import parabola data
    data = np.load('dataset/kinematic.npz')
    x_d = np.asarray(data["x_d"])
    x_v = np.asarray(data["x_v"])
    y_d = np.asarray(data["y_d"])
    y_v = np.asarray(data["y_v"])
    a_data = np.asarray(data["g"])

    # Prepare data
    # The first few time steps are reserved for the symbolic regression propagator
    x = np.stack((x_d, x_v), axis=2)    # Shape (N, NT, 2)
    y0 = np.stack((y_d[:, 0], y_v[:, 0]), axis=1)  # Input into the symbolic propagator
    y_data = np.stack((y_d[:, 1:timesteps + 1], y_v[:, 1:timesteps + 1]), axis=2)     # shape(NG, LENGTH, 2)

    # Encoder
    encoder = helpers.Encoder()     # layer should end with 1, which is the output
    x_input = tf.placeholder(shape=(None, x.shape[1], x.shape[2]), dtype=tf.float32, name="enc_input")
    y_input = tf.placeholder(shape=(None, timesteps, 2), dtype=tf.float32, name="label_input")
    y0_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="y_input")  # input is d, v
    length_input = tf.placeholder(dtype=tf.int32, shape=())
    training = tf.placeholder_with_default(False, [])
    z = encoder(x_input, training=training)
    # enc_output = np.array(g_data)[:, np.newaxis]  # uncomment to ignore the autoencoder

    # Build EQL network for the propagating decoder
    primitive_funcs = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        *[functions.Product(norm=0.1)] * 2,
    ]
    prop_d = SymbolicNetL0(2, funcs=primitive_funcs)
    prop_v = SymbolicNetL0(2, funcs=primitive_funcs)
    prop_d.build(4)
    prop_v.build(4)
    # Build recurrent structure
    rnn = tf.keras.layers.RNN(SymbolicCell(prop_d, prop_v), return_sequences=True)
    y0_rnn = tf.concat([tf.expand_dims(y0_input, axis=1), tf.zeros((tf.shape(y0_input)[0], length_input - 1, 2))], axis=1)
    prop_input = tf.concat([y0_rnn, tf.keras.backend.repeat(z, length_input),
                            tf.ones((tf.shape(y0_input)[0], length_input, 1))], axis=2)
    y_hat = rnn(prop_input)

    # Label and errors
    reg_loss = prop_d.get_loss() + prop_v.get_loss()

    # Training
    learning_rate_ph = tf.placeholder(tf.float32)
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ph)
    error = tf.losses.mean_squared_error(labels=y_input[:, :length_input, :], predictions=y_hat)
    loss = error + reg_weight * reg_loss
    train = opt.minimize(loss)
    train = tf.group([train, encoder.bn.updates])

    # Training session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    with tf.Session(config=config) as sess:
        loss_i = np.nan
        while np.isnan(loss_i):
            loss_list = []
            error_list = []
            reg_list = []

            sess.run(tf.global_variables_initializer())
            length_i = 1

            for i in range(n_epochs):
                lr_i = learning_rate

                feed_dict = {x_input: x, y0_input: y0, y_input: y_data,
                             learning_rate_ph: lr_i, training: True, length_input: length_i}
                _ = sess.run(train, feed_dict=feed_dict)
                if i % summary_step == 0:
                    feed_dict[training] = False
                    loss_val, error_val, reg_val = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                    loss_list.append(loss_val)
                    error_list.append(error_val)
                    reg_list.append(reg_val)
                    print("Epoch %d\tTotal loss: %f\tError: %f\tReg loss: %f" % (i, loss_val, error_val, reg_val))
                    loss_i = loss_val

                    if i > 3000:
                        length_i = timesteps
                    if np.isnan(loss_i):
                        break

        weights_d = sess.run(prop_d.get_weights())
        expr_d = pretty_print.network(weights_d, primitive_funcs, ["d", "v", "z", 1])
        print(expr_d)
        weights_v = sess.run(prop_v.get_weights())
        expr_v = pretty_print.network(weights_v, primitive_funcs, ["d", "v", "z", 1])
        print(expr_v)

        # z_arr = sess.run(enc_output, feed_dict=feed_dict)

        # Save results
        results = {
            "timesteps": timesteps,
            "summary_step": summary_step,
            "learning_rate": learning_rate,
            "N_EPOCHS": n_epochs,
            "reg_weight": reg_weight,
            "weights_d": weights_d,
            "weights_v": weights_v,
            "loss_plot": loss_list,
            "error_plot": error_list,
            "l12_plot": reg_list,
            "expr_d": expr_d,
            "expr_v": expr_v
        }

        trial_dir = helpers.get_trial_path(results_dir)     # Get directory in which to save trial results
        tf.saved_model.simple_save(sess, trial_dir,
                                   inputs={"x": x_input, "y0": y0_input, "training": training},
                                   outputs={"z": z, "y": y_hat})

        # Save a summary of the parameters and results
        with open(os.path.join(trial_dir, 'summary.pickle'), "wb+") as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on kinematics task.")
    parser.add_argument("--results-dir", type=str, default='results/kinematics/test_l0')
    parser.add_argument("--reg-weight", type=float, default=1e-3, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
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
