#! /usr/bin/env python3
"""Numerical ODE solvers

    Author:     David Hernandez
    Matr. Nr.:  01601331
    e-Mail:     david.hernandez@univie.ac.at

This module implements four different solver for ordinary differential
equations.
    • Forward Euler Method
    • Classical Runge-Kutta Method
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

def newton(init=0,
           function=lambda x : np.sin(x),
           derivative=lambda x : np.cos(x),
           tolerance=1e-7,
           interim_res=None,
           iters=1,
           verbosity=False):
    """Newton method for equation solving.

    The Newton method is an iterative approach to function solving.
    A different name for this method is tangent method because it
    uses the first derivative of the function in the equation. Each
    following estimate of the solution can be computed as follows:

                    f (x_i)
    x_(i+1) = x_i - ———————
                    f'(x_i)

    Since this method is iterative a recursive approach can be
    implemented to get the result within a userdefined tolerance.

    Parameters
    ----------
    init : scalar
        The previous estimate or the initial guess of the solution.
    function : lambda function
        The function in the equation.
    derivative : lambda function
        The first derivative of the function.
    tolerance : scalar
        The tolerance within the approximation my differ from the true
        solution.
    iters : scalar
        The number of iterations before the result converges.
    verbosity : boolean
        If true, the number of iterations will be printed for each run.

    Returns
    -------
    result : scalar
        The new estimate for the result of the equation.
    """

    ###################################################################
    #   Maximum recursion depth break condition.
    ###################################################################
    max_depth = 1000

    ###################################################################
    #   Compute the next approximation of the solution.
    ###################################################################
    next_approx = init - np.divide(function(init), derivative(init))

    ###################################################################
    #   Save interim results to a list.
    ###################################################################
    if interim_res is not None and isinstance(interim_res, list):
        interim_res.append(next_approx)
    else:
        wrn_msg = 'Warning: interim_res is not a list. Interim '\
                  + 'results will not be saved for later use!'
        warnings.warn(wrn_msg)

    ###################################################################
    #   Break condition:
    #       Either the next approximation is within the user defined
    #       tolerance or the maximum recursion depth is reached with
    #       no convergence. If the solution does not converge a
    #       RecursionError is raised.
    ###################################################################
    if np.abs(next_approx - init) < tolerance:
        if verbosity:
            print(f'Solution converged after {iters} iterations.\n')
        print(next_approx)
        exit(0)
        return next_approx
    elif iters > max_depth:
        err_msg = f'Maximum of {max_depth} iterations passed and no '\
                  + f'convergence occured.'
        raise RecursionError(err_msg)
    else:
        # print(next_approx)                # For debugging
        return newton(init=next_approx,
                      function=function,
                      derivative=derivative,
                      tolerance=tolerance,
                      interim_res=interim_res,
                      iters = iters + 1)


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
    result : np.array
        The results of each integration step. This is an approximation
        to y(t) for each time step t.
    """
    ###################################################################
    #   t_step : The integration time step length derived from start
    #            time, end time and the number of integration steps.
    #   result : A numpy array that will hold the approximation for
    #            each function value y(t).
    #   t : The parameter t which will be updated in each iteration.
    ###################################################################
    t_step = (t_end - t_start) / n_steps
    result = np.empty(n_steps)
    t = t_start

    for i in range(n_steps):
        if t == t_start:
            result[i] = init_cond
        else:
            result[i] = result[i-1] + t_step * derivative(t, result[i-1])
        t += t_step

    return result


def classical_runge_kutta(derivative,
                          init_cond=0.,
                          t_start=0.,
                          t_end=3.,
                          n_steps=100):
    """description"""
    t_step = (t_end - t_start) / n_steps
    result = np.empty(n_steps)
    t = t_start

    for i in range(n_steps):
        if t == t_start:
            result[i] = init_cond
        else:
            k_1 = derivative(t, result[i-1])
            k_2 = derivative(t + 1/2 * t_step,
                             result[i-1] + 1/2 * t_step * k_1)
            k_3 = derivative(t + 1/2 * t_step,
                             result[i-1] + 1/2 * t_step * k_2)
            k_4 = derivative(t + t_step,
                             result[i-1] + t_step * k_3)
            result[i] = result[i-1]\
                        + 1/6 * t_step * (k_1 + 2 * (k_2 + k_3) + k_4)
        t += t_step

    return result


def backward_euler(f,
                   f_prime,
                   init_cond=0.,
                   t_start=0.,
                   t_end=0.,
                   n_steps=100):
    """description"""
    t_step = (t_end - t_start) / n_steps
    result = np.empty(n_steps)
    t = t_start

    for i in range(n_steps):
        if t == t_start:
            result[i] = init_cond
        else:
            G = lambda y_next : y_next - result[i]\
                                - t_step * f(t + t_step, y_next)
            G_prime = lambda y_next : 1 - t_step * f_prime(t + t_step, y_next)

            y_next = newton(init=0,
                            function=G,
                            derivative=G_prime)
            result[i] = result[i-1] + t_step * f(t + t_step, y_next)

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

    y_crk = classical_runge_kutta(y_prime,
                                  init_cond=1,
                                  t_start=0,
                                  t_end=3,
                                  n_steps=100)

    y_double_prime = lambda t, y : y * (-2 + 4 * t**2)

    y_be = backward_euler(f=y_prime,
                          f_prime=y_double_prime,
                          init_cond=1,
                          t_start=0,
                          t_end=3,
                          n_steps=100)

    print(y_fe - y_be)

if __name__ == '__main__':
    main()
    exit(0)

