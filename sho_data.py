"""
Generate data for the simple harmonic oscillator (SHO)

Equation: x'' + w0^2*x=0

Arguments:
[-N]: integer number of different constant g's to be generated
[-T]: integer number of timesteps to be generated
output: string destination of file output
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-N', nargs='?', type=int, default=1000)     # Number of data points
parser.add_argument('--t-max', nargs='?', type=int, default=50)
parser.add_argument('-T', nargs='?', type=int, default=500)
parser.add_argument('--output', metavar='f', default='dataset/sho')
args = parser.parse_args()

test = False    # Used for debugging

NT = args.T     # Number of time steps
N = args.N     # Number of data points (number of g values to sample)
T_MAX = args.t_max

# Min/max limits on parameters
OMEGA2_MIN = 0.1
OMEGA2_MAX = 1
X0_MIN = -1
X0_MAX = 1
V0_MIN = -0.5
V0_MAX = 0.5

# Draw from uniform distribution for each parameter.
# Initial conditions have shape (2, NG, 1), where first axis corresponds to x and y datasets.
# Parameters have shape (1, NG, 1)
# Last dimension is for numpy broadcasting for calculating time-dependent x and v
x0 = (X0_MAX - X0_MIN) * np.random.random((2, N, 1)) + X0_MIN
v0 = (V0_MAX - V0_MIN) * np.random.random((2, N, 1)) + V0_MIN
omega2 = (OMEGA2_MAX - OMEGA2_MIN) * np.random.random((1, N, 1)) + OMEGA2_MIN

t = np.linspace(0, T_MAX, NT)
dt = t[1] - t[0]
t = t[np.newaxis, np.newaxis, :]    # For broadcasting

omega = np.sqrt(omega2)
x = x0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)
v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)

t = np.ravel(t)

if test:
    # Calculate using finite-difference
    x_fd = np.zeros((2, N, NT))
    v_fd = np.zeros((2, N, NT))
    x_fd[:, :, 0] = x0[:, :, 0]
    v_fd[:, :, 0] = v0[:, :, 0]

    for i in range(NT-1):
        x_fd[:, :, i+1] = v_fd[:, :, i] * dt + x_fd[:, :, i]
        v_fd[:, :, i+1] = -omega2[:, :, 0] * x_fd[:, :, i] * dt + v_fd[:, :, i]

    plt.figure()
    plt.plot(t, x[0, 0, :])
    plt.plot(t, v[0, 0, :])

    plt.figure()
    for i in range(10):
        plt.plot(t, x[0, i, :])

    plt.figure()
    for i in range(10):
        plt.plot(t, v[0, i, :])

    plt.figure()
    for i in range(3):
        plt.plot(t, x[0, i, :])
        plt.plot(t, x_fd[0, i, :], ':')
    plt.xlim([0, 20])

    plt.show()

else:
    outfile = args.output

    np.savez(outfile,
             NT=NT,
             N=N,
             x_d=x[0],
             x_v=v[0],
             y_d=x[1],
             y_v=v[1],
             omega2=np.ravel(omega2))
