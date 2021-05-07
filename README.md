# Integration of Neural Network-Based Symbolic Regression in Deep Learning for Scientific Discovery

This repository is the official implementation of 
[Integration of Neural Network-Based Symbolic Regression in Deep Learning for Scientific Discovery](https://arxiv.org/abs/1912.04825)

Please cite the above paper if you use this code for your work.

Symbolic regression, in which a model discovers an analytical equation describing a dataset as opposed to finding 
the weights of pre-defined features, is normally implemented using genetic programming.
Here, we demonstrate symbolic regression using neural networks,
which allows symbolic regression to be performed using gradient-based optimization techniques,
i.e., backpropagation.
It can be integrated with other deep learning architectures, allowing for end-to-end training of a system that produces
interpretable and generalizable results. This repository implements symbolic regression with neural networks and
demonstrates its integration with deep learning for multiple tasks,
including arithmetic on MNIST digits and extracting the equations of kinematic and differential equation datasets.

The PyTorch implementation can be found at [this repository](https://github.com/samuelkim314/DeepSymRegTorch)

## Requirements

* Python 3.5
* TensorFlow 1.15
* NumPy 1.16 (does not work on 1.17)
* Scipy 1.3
* Sympy 1.6
* Matplotlib (optional)

All dependencies are in [requirements.txt](requirements.txt). 
To install required packages, you can simply run the following code in your shell.
```
pip install -r requirements.txt
```

Note that the pretty_print functions in SymPy 1.4 only works with TensorFlow <=1.13.

## Package Description

The core code is all contained inside the utils/ directory

* `functions.py` contains different primitives, or activation functions. The primitives are built as classes so that different parts of the code (TensorFlow versus NumPy versus SymPy) have a unified way of addressing the functions.

* `pretty_print.py` contains functions to print out the equations in the end in a human-readable format from a trained EQL network.

* `symbolic_network.py` contains the core code of the EQL network, including methods for L0 regularization.

## Quick Intro

This demonstrates a minimal example for how to use this library for training the EQL network.

```python
import numpy as np
import tensorflow as tf
from utils import functions, pretty_print
from utils.symbolic_network import SymbolicNetL0
from utils.regularization import l12_smooth


funcs = functions.default_func
x_dim = 1
# Random data for a simple function
x = np.random.rand(100, x_dim) * 2 - 1
y = x ** 2

# Set up TensorFlow graph for the EQL network
x_placeholder = tf.placeholder(shape=(None, x_dim), dtype=tf.float32)
sym = SymbolicNetL0(symbolic_depth=2, funcs=funcs, init_stddev=0.5)
y_hat = sym(x_placeholder)

# Set up loss function with L0.5 loss
mse = tf.losses.mean_squared_error(labels=y, predictions=y_hat)
loss = mse + 1e-2 * sym.get_loss()

# Set up TensorFlow graph for training
opt = tf.train.RMSPropOptimizer(learning_rate=1e-2)
train = opt.minimize(loss)

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train, feed_dict={x_placeholder: x})

    # Print out the expression
    weights = sess.run(sym.get_weights())
    expr = pretty_print.network(weights, funcs, ['x'])
    print(expr)

```

For a more complete example with training stages or L0 regularization, see below.

## Training

Each task are trained independently. 
Refer to the paper https://arxiv.org/abs/1912.04825 for a description of each of the tasks.

### Benchmark
`benchmark_accuracy.py`/`benchmark_accuracy_l0.py`: 
Run EQL benchmarks on various functions using smoothed L0.5 and relaxed L0 regularization, respectively.

### MNIST
`mnist_math.py`: Learn arithmetic operations on MNIST digits. 

`mnist_math_relu.py`: Same as `mnist_math.py`, 
but using a conventional neural network with ReLU activation functions instead of the EQL network.

### Kinematics
`kinematics_data.py`: Generate data for the kinematics task. This must be run before training the model.

`kinematics_sr.py`/`kinematics_sr_l0.py`: Dynamics encoder combined with a recurrent EQL network for the kinematics task
using smoothed L0.5 and relaxed L0 regularization, respectively. 
`kinematics_sr.py` implements an unrolled RNN to demonstrate the internal architecture, while 
`kinematics_sr_l0.py` implements the RNN using the built-in TensorFlow libraries.

`kinematics_relu.py`: Same as `kinematics_sr.py`
but using a conventional neural network with ReLU activation functions instead of the EQL network.

### Simple Harmonic Oscillator (SHO)
`sho_data.py`: Generatedata for the SHO task. This must be run before training the model.

`sho_sr.py`/`sho_sr_l0.py`: Dynamics encoder combined with a recurrent EQL network for the kinematics task
using smoothed L0.5 and relaxed L0 regularization, respectively. 
Both implement the RNN using the built-in TensorFlow libraries.

`sho_relu.py`: Same as `sho_sr.py`
but using a conventional neural network with ReLU activation functions instead of the EQL network.

### Authors
Samuel Kim, Peter Lu, Srijon Mukherjee, Michael Gilbert, Li Jing, Vladimir Ceperic, Marin Soljacic

### Contributing
If you'd like to contribute, or have any suggestions for these guidelines, 
you can contact Samuel Kim at samkim (at) mit (dot) edu or open an issue on this GitHub repository.

All content in this repository is licensed under the MIT license.
