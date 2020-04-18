import tensorflow as tf
import numpy as np
import os
import pickle
import argparse
from utils.symbolic_network import SymbolicNet, MaskedSymbolicNet
from utils import functions, regularization, helpers, pretty_print


def main(results_dir='results/kinematics/test', learning_rate=1e-2, reg_weight=1e-3, n_epochs1=5001, n_epochs2=5001,
         timesteps=5):
    # Hyperparameters
    summary_step = 500
    timesteps0 = 1

    # Import kinematics data
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
    label_data = np.stack((y_d[:, 1:timesteps+1], y_v[:, 1:timesteps+1]), axis=2)     # shape(NG, timesteps, 2)

    # Encoder
    encoder = helpers.Encoder()     # layer should end with 1, which is the output
    x_input = tf.placeholder(shape=(None, x.shape[1], x.shape[2]), dtype=tf.float32, name="enc_input")
    y_input = tf.placeholder(shape=(None, timesteps, 2), dtype=tf.float32, name="label_input")
    training = tf.placeholder_with_default(False, [])
    z = encoder(x_input, training=training)
    # z = np.array(a_data)[:, np.newaxis]  # uncomment to ignore the autoencoder

    # Propagating decoder
    primitive_funcs = [
        *[functions.Constant()] * 2,
        *[functions.Identity()] * 4,
        *[functions.Square()] * 4,
        *[functions.Sin()] * 2,
        *[functions.Exp()] * 2,
        *[functions.Sigmoid()] * 2,
        *[functions.Product(norm=0.1)] * 2,
    ]
    prop_d = SymbolicNet(2, funcs=primitive_funcs)
    prop_v = SymbolicNet(2, funcs=primitive_funcs)
    prop_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="prop_input")  # input is d, v

    def rec_sr(y0_input, enc_output, length, prop1=prop_d, prop2=prop_v):
        rec_input = [y0_input]
        for i in range(length):
            full_input = tf.concat([rec_input[i], enc_output, tf.ones_like(enc_output)], axis=1, name="full_input")  # d, v, z
            rec_input.append(tf.concat([prop1(full_input), prop2(full_input)], axis=1, name="c_prop_input"))
        output = tf.stack(rec_input[1:], axis=1)  # Ignore initial conditions
        return output

    y_hat_start = rec_sr(prop_input, z, timesteps0, prop_d, prop_v)
    y_hat_full = rec_sr(prop_input, z, timesteps, prop_d, prop_v)

    # Label and errors
    epoch = tf.placeholder(tf.float32)
    reg_weight_ph = tf.placeholder(tf.float32)
    reg_loss = regularization.l12_smooth(prop_d.get_weights()) + regularization.l12_smooth(prop_v.get_weights())

    # Training
    learning_rate_ph = tf.placeholder(tf.float32)
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ph)

    def define_loss(prop_output, length):
        error = tf.losses.mean_squared_error(labels=y_input[:, :length, :], predictions=prop_output[:, :length, :])
        loss = error + reg_weight_ph * reg_loss
        train = opt.minimize(loss)
        train = tf.group([train, encoder.bn.updates])
        return error, loss, train

    error_start, loss_start, train_start = define_loss(y_hat_start, timesteps0)
    error_full, loss_full, train_full = define_loss(y_hat_full, timesteps)

    # Training session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # Take up variable amount of memory on GPU
    with tf.Session(config=config) as sess:

        loss_i = np.nan
        while np.isnan(loss_i):

            loss_list = []
            error_list = []
            reg_list = []
            error, loss, train = error_start, loss_start, train_start

            sess.run(tf.global_variables_initializer())

            for i in range(n_epochs1):
                feed_dict = {x_input: x, prop_input: y0, y_input: label_data,
                             epoch: 0, learning_rate_ph: learning_rate, training: True, reg_weight_ph: reg_weight}
                _ = sess.run(train, feed_dict=feed_dict)
                if i % summary_step == 0:
                    feed_dict[training] = False
                    print_loss, print_error, print_l12 = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                    loss_list.append(print_loss)
                    error_list.append(print_error)
                    reg_list.append(print_l12)
                    print("Epoch %d\tTotal loss: %f\tError: %f\tReg loss: %f" % (i, print_loss, print_error, print_l12))
                    loss_i = print_loss
                    if i > 2000:
                        error, loss, train = error_full, loss_full, train_full
                    if np.isnan(loss_i):
                        break

        # Setting small weights to 0 and freezing them
        prop_d_masked = MaskedSymbolicNet(sess, prop_d, threshold=0.1)
        prop_v_masked = MaskedSymbolicNet(sess, prop_v, threshold=0.1)

        # Rebuilding the decoding propagator
        prop_output_masked = rec_sr(prop_input, z, timesteps, prop_d_masked, prop_v_masked)
        error, loss, train = define_loss(prop_output_masked, timesteps)

        weights_d = sess.run(prop_d_masked.get_weights())
        expr_d = pretty_print.network(weights_d, primitive_funcs, ["d", "v", "z", 1])
        print(expr_d)
        weights_v = sess.run(prop_v_masked.get_weights())
        expr_v = pretty_print.network(weights_v, primitive_funcs, ["d", "v", "z", 1])
        print(expr_v)

        print("Frozen weights. Next stage of training.")

        for i in range(n_epochs2):
            feed_dict = {x_input: x, prop_input: y0, y_input: label_data,
                         epoch: 0, learning_rate_ph: learning_rate / 10, training: True, reg_weight_ph: 0}
            _ = sess.run(train, feed_dict=feed_dict)
            if i % summary_step == 0:
                feed_dict[training] = False
                print_loss, print_error, print_l12 = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                loss_list.append(print_loss)
                error_list.append(print_error)
                reg_list.append(print_l12)
                print("Epoch %d\tError: %g" % (i, print_error))

        weights_d = sess.run(prop_d_masked.get_weights())
        expr_d = pretty_print.network(weights_d, primitive_funcs, ["d", "v", "z", 1])
        print(expr_d)
        weights_v = sess.run(prop_v_masked.get_weights())
        expr_v = pretty_print.network(weights_v, primitive_funcs, ["d", "v", "z", 1])
        print(expr_v)

        # Save results
        results = {
            "timesteps": timesteps,
            "summary_step": summary_step,
            "learning_rate": learning_rate,
            "n_epochs1": n_epochs1,
            "n_epochs2": n_epochs2,
            "reg_weight_ph": reg_weight,
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
                                   inputs={"x": x_input, "y0": prop_input, "training": training},
                                   outputs={"z": z, "y": y_hat_full})

        # Save a summary of the parameters and results
        with open(os.path.join(trial_dir, 'summary.pickle'), "wb+") as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EQL network on kinematics task.")
    parser.add_argument("--results-dir", type=str, default='results/kinematics/test')
    parser.add_argument("--reg-weight", type=float, default=1e-3, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=5001, help="Number of epochs to train in each stage")
    parser.add_argument("--n-epochs2", type=int, default=5001, help="Number of epochs to train in each stage")
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
