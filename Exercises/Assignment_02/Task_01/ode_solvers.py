#! /usr/bin/env python3
"""Numerical ODE solvers

    Author:     David Hernandez
    Matr. Nr.:  01601331
    e-Mail:     david.hernandez@univie.ac.at

This module implements four different solver for ordinary differential
equations.
    • Forward Euler Method
    • Calssical Runge-Kutta Method
    • Backward Euler Method
    • Crank-Nicolson Method
"""

#######################################################################
#   Import packages.
#######################################################################

import warnings

import numpy as np

from matplotlib import rc
from matplotlib import pyplot as plt

#######################################################################
#   Define functions used in this file or package.
#######################################################################

def forward_euler(derivative,
                  init_cond=0.,
                  t_start=0.,
                  t_end=1.,
                  n_steps=100):
    """Forward Euler ODE solver.

    Parameters:
    -----------
    derivative : function
        This is the derivative of the function that shall be
        determined.
    init_cond : scalar
        The initial condition of the ODE
    t_start : scalar
        The start time of the integration.
    t_end : scalar
        The end time of the integration.
    n_steps : scalar
        The number of integration time steps.

    Returns:
    result : float
        The approximated next function value y_(n+1).
    """
    t_step = (t_end - t_start) / n_steps
    result = np.zeros(n_steps)
    t = t_start

    for i in range(len(result)):
        buffer = result[i-1]
        if t == t_start:
            result[i] = init_cond
        else:
            result[i] = buffer + t_step * derivative(t, buffer)
        t += t_step
    return result


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""
    y_prime = lambda t, y : -2 * t * y
    y_fe = forward_euler(y_prime,
                         init_cond=1,
                         t_start=0,
                         t_end=3,
                         n_steps=100)

    y_crk = classical_runge_kutta()



if __name__ == '__main__':
    main()
    exit(0)

