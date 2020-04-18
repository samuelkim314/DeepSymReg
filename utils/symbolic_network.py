"""Contains the symbolic regression neural network architecture."""

import tensorflow as tf
from utils import functions

# Constants for L0 Regularization
BETA = 2 / 3
GAMMA = -0.1
ZETA = 1.1
EPSILON = 1e-6


class SymbolicLayer:
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""
    def __init__(self, funcs=None, initial_weight=None, variable=False, init_stddev=0.1):
        """

        funcs: List of activation functions, using utils.functions
        initial_weight: (Optional) Initial value for weight matrix
        variable: Boolean of whether initial_weight is a variable or not
        init_stddev: (Optional) if initial_weight isn't passed in, this is standard deviation of initial weight
        """

        if funcs is None:
            funcs = functions.default_func
        self.initial_weight = initial_weight
        self.W = None       # Weight matrix
        self.built = False  # Boolean whether weights have been initialized
        if self.initial_weight is not None:     # use the given initial weight
            with tf.name_scope("symbolic_layer"):
                if not variable:
                    self.W = tf.Variable(self.initial_weight)
                else:
                    self.W = self.initial_weight
            self.built = True

        self.output = None  # Tensorflow tensor for layer output
        self.init_stddev = init_stddev
        self.n_funcs = len(funcs)           # Number of activation functions (and number of layer outputs)
        self.funcs = [func.tf for func in funcs]        # Convert functions to list of Tensorflow functions
        self.n_double = functions.count_double(funcs)   # Number of activation functions that take 2 inputs
        self.n_single = self.n_funcs - self.n_double    # Number of activation functions that take 1 input

        self.out_dim = self.n_funcs + self.n_double

    def build(self, in_dim):
        """Initialize weight matrix"""
        self.W = tf.Variable(tf.random_normal(shape=[in_dim, self.out_dim], stddev=self.init_stddev))
        self.built = True

    def __call__(self, x):
        """Multiply by weight matrix and apply activation units"""
        with tf.name_scope("symbolic_layer"):
            if not self.built:
                self.build(x.shape[1].value)    # First dimension is batch size
            g = tf.matmul(x, self.W)  # shape = (?, self.size)
            self.output = []

            in_i = 0    # input index
            out_i = 0   # output index
            # Apply functions with only a single input
            while out_i < self.n_single:
                self.output.append(self.funcs[out_i](g[:, in_i]))
                in_i += 1
                out_i += 1
            # Apply functions that take 2 inputs and produce 1 output
            while out_i < self.n_funcs:
                self.output.append(self.funcs[out_i](g[:, in_i], g[:, in_i+1]))
                in_i += 2
                out_i += 1
            self.output = tf.stack(self.output, axis=1)
            return self.output

    def get_weight(self):
        return self.W


class SymbolicLayerBias(SymbolicLayer):
    """SymbolicLayer with a bias term"""
    def __init__(self, funcs=None, initial_weight=None, variable=False, init_stddev=0.1):
        super().__init__(funcs, initial_weight, variable, init_stddev)
        self.b = None

    def build(self, in_dim):
        super().build(in_dim)
        self.b = tf.Variable(tf.ones(shape=self.n_funcs) * 0.01)

    def __call__(self, x):
        """Multiply by weight matrix and apply activation units"""
        super().__call__(x)
        self.output += self.b
        return self.output


class SymbolicNet:
    """Symbolic regression network with multiple layers. Produces one output."""
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None,
                 variable=False, init_stddev=0.1):
        self.depth = symbolic_depth     # Number of hidden layers
        self.funcs = funcs
        self.shape = (None, 1)
        if initial_weights is not None:
            self.symbolic_layers = [SymbolicLayer(funcs=funcs, initial_weight=initial_weights[i], variable=variable)
                                    for i in range(self.depth)]
            if not variable:
                self.output_weight = tf.Variable(initial_weights[-1])
            else:
                self.output_weight = initial_weights[-1]
        else:
            # Each layer initializes its own weights
            if isinstance(init_stddev, list):
                self.symbolic_layers = [SymbolicLayer(funcs=funcs, init_stddev=init_stddev[i]) for i in range(self.depth)]
            else:
                self.symbolic_layers = [SymbolicLayer(funcs=funcs, init_stddev=init_stddev) for _ in range(self.depth)]
            # Initialize weights for last layer (without activation functions)
            self.output_weight = tf.Variable(tf.random_uniform(shape=(self.symbolic_layers[-1].n_funcs, 1)))

    def build(self, input_dim):
        in_dim = input_dim
        for i in range(self.depth):
            self.symbolic_layers[i].build(in_dim)
            in_dim = self.symbolic_layers[i].n_funcs

    def __call__(self, input):
        self.shape = (int(input.shape[1]), 1)     # Dimensionality of the input
        h = input
        # Building hidden layers
        for i in range(self.depth):
            h = self.symbolic_layers[i](h)
        # Final output (no activation units) of network
        h = tf.matmul(h, self.output_weight)
        return h

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [self.symbolic_layers[i].W for i in range(self.depth)] + [self.output_weight]


class MaskedSymbolicNet(SymbolicNet):
    """Symbolic regression network where weights below a threshold are set to 0 and frozen. In other words, we apply
    a mask to the symbolic network to fine-tune the non-zero weights."""
    def __init__(self, sess, sr_unit, threshold=0.01):
        # weights = sr_unit.get_weights()
        # masked_weights = []
        # for w_i in weights:
        #     # w_i = tf.where(tf.abs(w_i) < threshold, tf.zeros_like(w_i), w_i)
        #     # w_i = tf.where(tf.abs(w_i) < threshold, tf.stop_gradient(w_i), w_i)
        #     masked_weights.append(w_i)

        weights = sr_unit.get_weights()
        masked_weights = []
        for w_i in weights:
            mask = tf.constant(sess.run(tf.abs(w_i) > threshold), dtype=tf.float32)
            masked_weights.append(tf.multiply(w_i, mask))

        super().__init__(sr_unit.depth, funcs=sr_unit.funcs, initial_weights=masked_weights, variable=True)
        self.sr_unit = sr_unit


class SymbolicLayerL0(SymbolicLayer):
    def __init__(self, funcs=None, initial_weight=None, variable=False, init_stddev=0.1,
                 bias=False, droprate_init=0.5, lamba=1.):
        super().__init__(funcs, initial_weight, variable, init_stddev)
        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.qz_log_alpha = None
        self.in_dim = None
        self.eps = None

    def build(self, in_dim):
        with tf.name_scope("symbolic_layer"):
            self.in_dim = in_dim
            if self.W is None:
                self.W = tf.Variable(tf.random_normal(shape=[in_dim, self.out_dim], stddev=self.init_stddev))
            if self.use_bias:
                self.bias = tf.Variable(0.1*tf.ones((1, self.out_dim)))
            self.qz_log_alpha = tf.Variable(tf.random_normal((in_dim, self.out_dim),
                                                             mean=tf.log(1-self.droprate_init) - tf.log(self.droprate_init),
                                                             stddev=1e-2))

    def quantile_concrete(self, u):
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = tf.sigmoid((tf.log(u) - tf.log(1.0-u) + self.qz_log_alpha) / BETA)
        return y * (ZETA - GAMMA) + GAMMA

    def sample_u(self, shape, reuse_u=False):
        """Uniform random numbers for concrete distribution"""
        # print("Hello")
        if self.eps is None or not reuse_u:
            self.eps = tf.random.uniform(shape=shape, minval=EPSILON, maxval=1.0 - EPSILON)
        return self.eps

    def sample_z(self, batch_size, sample=True):
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return tf.clip_by_value(z, 0, 1)
        else:   # Mean of the hard concrete distribution
            pi = tf.sigmoid(self.qz_log_alpha)
            return tf.clip_by_value(pi * (ZETA - GAMMA) + GAMMA, clip_value_min=0.0, clip_value_max=1.0)

    def get_z_mean(self):
        """Mean of the hard concrete distribution"""
        pi = tf.sigmoid(self.qz_log_alpha)
        return tf.clip_by_value(pi * (ZETA - GAMMA) + GAMMA, clip_value_min=0.0, clip_value_max=1.0)

    def sample_weights(self, reuse_u=False):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim), reuse_u=reuse_u))
        mask = tf.clip_by_value(z, clip_value_min=0.0, clip_value_max=1.0)
        return mask * self.W

    def get_weight(self):
        """Deterministic value of weight based on mean of z"""
        return self.W * self.get_z_mean()

    def loss(self):
        """Regularization loss term"""
        return tf.reduce_sum(tf.sigmoid(self.qz_log_alpha - BETA * tf.log(-GAMMA / ZETA)))

    def __call__(self, x, sample=True, reuse_u=False):
        """Multiply by weight matrix and apply activation units"""
        with tf.name_scope("symbolic_layer"):
            if self.W is None or self.qz_log_alpha is None:
                self.build(x.shape[1].value)

            if sample:
                h = tf.matmul(x, self.sample_weights(reuse_u=reuse_u))
            else:
                w = self.get_weight()
                h = tf.matmul(x, w)

            if self.use_bias:
                h = h + self.bias

            # shape of h = (?, self.n_funcs)

            self.output = []
            # apply a different activation unit to each column of h
            in_i = 0    # input index
            out_i = 0   # output index
            # Apply functions with only a single input
            while out_i < self.n_single:
                self.output.append(self.funcs[out_i](h[:, in_i]))
                in_i += 1
                out_i += 1
            # Apply functions that take 2 inputs and produce 1 output
            while out_i < self.n_funcs:
                self.output.append(self.funcs[out_i](h[:, in_i], h[:, in_i+1]))
                in_i += 2
                out_i += 1
            self.output = tf.stack(self.output, axis=1)
            return self.output


class SymbolicNetL0(SymbolicNet):
    """Symbolic regression network with multiple layers. Produces one output."""
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None,
                 variable=False, init_stddev=0.1):
        super().__init__(symbolic_depth, funcs, initial_weights, initial_bias, variable, init_stddev)
        if initial_weights is not None:
            self.symbolic_layers = [SymbolicLayerL0(funcs=funcs, initial_weight=initial_weights[i], variable=variable)
                                    for i in range(self.depth)]
            if not variable:
                self.output_weight = tf.Variable(initial_weights[-1])
            else:
                self.output_weight = initial_weights[-1]
        else:
            # Each layer initializes its own weights
            if isinstance(init_stddev, list):
                self.symbolic_layers = [SymbolicLayerL0(funcs=funcs, init_stddev=init_stddev[i])
                                        for i in range(self.depth)]
            else:
                self.symbolic_layers = [SymbolicLayerL0(funcs=funcs, init_stddev=init_stddev)
                                        for _ in range(self.depth)]
            # Initialize weights for last layer (without activation functions)
            self.output_weight = tf.Variable(tf.random_uniform(shape=(self.symbolic_layers[-1].n_funcs, 1)))

    def __call__(self, input, sample=True, reuse_u=False):
        self.shape = (int(input.shape[1]), 1)     # Dimensionality of the input
        # connect output from previous layer to input of next layer
        h = input
        for i in range(self.depth):
            h = self.symbolic_layers[i](h, sample=sample, reuse_u=reuse_u)
        # Final output (no activation units) of network
        h = tf.matmul(h, self.output_weight)
        return h

    def get_loss(self):
        return tf.reduce_sum([self.symbolic_layers[i].loss() for i in range(self.depth)])

    def get_weights(self):
        return self.get_symbolic_weights() + [self.get_output_weight()]

    def get_symbolic_weights(self):
        return [self.symbolic_layers[i].get_weight() for i in range(self.depth)]

    def get_output_weight(self):
        return self.output_weight


class SymbolicCell(tf.keras.layers.SimpleRNNCell):
    """cell for use with tf.keras.layers.RNN, allowing us to build a recurrent network with the EQL network.
    This is used for the propagating decoder in the dynamics architecture.
    Assume two state variables: position and velocity."""
    state_size = 2
    output_size = 2

    def __init__(self, sym1, sym2):
        # units: dimensionality of output space
        super().__init__(units=self.output_size)
        self.sym1 = sym1
        self.sym2 = sym2

    def call(self, inputs, state, training=None):
        """
        Arguments:
            inputs (at time t): shape (batch, feature)
            state: [x], shape(x)=(batch, feature)
            training:   Ignore this
        """
        full_input = state[0] + inputs[:, :2]
        full_input = tf.concat([full_input, inputs[:, 2:4]], axis=1)
        output = tf.concat([self.sym1(full_input), self.sym2(full_input)], axis=1)
        next_state = output
        return output, next_state
