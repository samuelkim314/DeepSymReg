"""
Usage:
python parabola_network output [-N] [-T]

Arguments:
[-N]: integer number of different constant a's to be generated
[-T]: integer number of timesteps to be generated
output: string destination of file output
"""

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-N', nargs='?', type=int, default=100)
parser.add_argument('-T', nargs='?', type=int, default=100)
parser.add_argument('--output', metavar='f', default='dataset/kinematic')
args = parser.parse_args()

NT = args.T     # Number of time steps
NG = args.G     # Number of data points (number of g values to sample)

# Min/max limits on parameters
D0_MIN = -1
D0_MAX = 1
V0_MIN = -1
V0_MAX = 1
A_MIN = -1
A_MAX = 1

# Draw from uniform distribution for each parameter. Shape [2, NG], where first axis corresponds to x and y datasets
# and second axis correspond to number of data points.
d0 = (D0_MAX - D0_MIN) * np.random.random((2, NG)) + D0_MIN
v0 = (V0_MAX - V0_MIN) * np.random.random((2, NG)) + V0_MIN
a = (A_MAX - A_MIN) * np.random.random((1, NG)) + A_MIN

t = np.arange(NT)
d = np.zeros((2, NG, NT))
v = np.zeros((2, NG, NT))

# Propagate forward in time
for t_i in range(NT):
    d[:, :, t_i] = d0 + v0 * t_i + 0.5 * a * t_i ** 2
    v[:, :, t_i] = v0 + a * t_i

outfile = args.output

np.savez(outfile,
         NT=NT,
         NG=NG,
         x_d=d[0],
         x_v=v[0],
         y_d=d[1],
         y_v=v[1],
         g=a[:])

