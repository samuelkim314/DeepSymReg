# Integration of Neural Network-Based Symbolic Regression in Deep Learning for Scientific Discovery

This repository is the official implementation of 
[Integration of Neural Network-Based Symbolic Regression in Deep Learning for Scientific Discovery](https://arxiv.org/abs/1912.04825)

Symbolic regression, in which a model discovers an analytical equation describing a dataset as opposed to finding 
the weights of pre-defined features, is normally implemented using genetic programming.
Here, we demonstrate symbolic regression using neural networks,
which allows symbolic regression to be performed using gradient-based optimization techniques,
i.e., backpropagation.
It can be integrated with other deep learning architectures, allowing for end-to-end training of a system that produces
interpretable and generalizable results. This repository implements symbolic regression with neural networks and
demonstrates its integration with deep learning for multiple tasks,
including arithmetic on MNIST digits and extracting the equations of kinematic and differential equation datasets.

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