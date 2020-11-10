#! /usr/bin/env python3
"""Runge-Kutta-Fehlberg ODE solver

    Author:     David Hernandez
    Matr. Nr.:  01601331
    e-Mail:     david.hernandez@univie.ac.at

Description...
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

def get_k(f, dt, y_n, t):
    """Get k_i coefficients."""
    k = np.empty(6)

    k[0] = f(t, y_n)
    k[1] = f(t + 1/4*dt, y_n + dt/4 * k[0])
    k[2] = f(t + 3/8*dt, y_n
                         + 3/32*dt*k[0]
                         + 9/32*dt*k[1])
    k[3] = f(t + 12/13*dt, y_n
                           + 1932/2197*dt*k[0]
                           - 7200/2197*dt*k[1]
                           + 7296/2197*dt*k[2])
    k[4] = f(t + dt, y_n
                     + 439/216*dt*k[0]
                     - 8*dt*k[1]
                     + 3680/513*dt*k[2]
                     - 845/4104*dt*k[3])
    k[5] = f(t + dt/2, y_n
                       - 8/27*dt*k[0]
                       - 2*dt*k[1]
                       + 3544/2565*dt*k[2]
                       + 1859/4104*dt*k[3]
                       - 11/40*dt*k[4])
    return k


def optimize(k, y_n, dt, tol):
    """Optimizes time step."""
    a = 1 - 1e-3

    y5 = y_n + dt * (16/135 * k[0]
                     + 6656/12825 * k[2]
                     + 28561/56430* k[3]
                     - 9/50 * k[4]
                     - 2/55 * k[5])
    y4 = y_n + dt * (25/216 * k[0]
                     + 1408/2565 * k[2]
                     + 2197/4104 * k[3]
                     - 1/5 * k[4])

    lte = y5 - y4
    lte_tol = tol * y_n
    # delta = lte_tol / lte
    # print(delta)
    # exit(0)

    dt_opt = a * dt * (np.abs(lte_tol/lte))**0.2
    print(f'dt_opt:\t{dt_opt}')

    if dt_opt < dt:
        return optimize(k, y_n, dt_opt, tol)
    else:
        return dt_opt, y5


def rk45(f,
         init_cond=0.,
         tolerance=1e-3,
         init_t_step=1e-1,
         t_start=0.,
         t_end=20.):
    """Runge-Kutta-Fehlberg method.

    description...

    Parameters:
    -----------
    f : function
        The derivative y'(t) of the function y(t) that shall be
        determined.
    init_cond : scalar
        The initial condition of the differential equation.
    tolerance : scalar
        The local truncation error tolerance LTE_tol that shall be used
        to work out the optimal time step t_opt.
    t_init : scalar
        The initial time step guess.

    Returns:
    --------
    result : numpy.array
        The approximated function values y(t).
    """
    result = []
    t = t_start
    dt = init_t_step
    i = 0

    while t < t_end:
        if t == t_start:
            result.append(init_cond)
        else:
            y_n = result[i-1]
            # print(y_n)
            k = get_k(f, dt, result[i-1], t)

            dt, y_next = optimize(k, result[i-1], dt, tolerance)
            # test = optimize(k, result[i-1], dt, tolerance)
            result.append(y_next)

        i += 1
        t += dt

    print('finished one xD')
    return(np.array(result))

#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""
    y0 = 1
    y_prime = lambda t, y : 0.1 * y + np.sin(t)

    t_start = 0
    t_end = 20

    lte_tolerances = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    # result = 

    results = np.array([rk45(f=y_prime,
                             init_cond=y0,
                             tolerance=lte_tol,
                             init_t_step=1,
                             t_start=t_start,
                             t_end=t_end) for lte_tol in lte_tolerances])
    print(results)
    print('Done!')

    ###################################################################
    #   Set LaTeX font to default used in LaTeX documents.
    ###################################################################
    rc('font',
       **{'family':'serif',
          'serif':['Computer Modern Roman']})
    rc('text', usetex=True)

    ###################################################################
    #   Create new matplotlib.pyplot figure with subplots.
    ###################################################################
    fig = plt.figure(figsize=(6.3, 3.54))       #   figsize in inches

    ###################################################################
    #   Plot the data.
    ###################################################################
    ax1 = fig.add_subplot(111)

    ax1.grid(True, which='major', linewidth=0.5)

    ax1.plot(np.linspace(start=0, stop=20, num=len(results[0])),
             results[0])
    ax1.plot(np.linspace(start=0, stop=20, num=len(results[1])),
             results[1])
    ax1.plot(np.linspace(start=0, stop=20, num=len(results[2])),
             results[2])
    ax1.plot(np.linspace(start=0, stop=20, num=len(results[3])),
             results[3])
    ax1.plot(np.linspace(start=0, stop=20, num=len(results[4])),
             results[4])

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax1.set_title('title')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y(t)')
    ax1.legend()

    ###################################################################
    #   Save the Figure as vector graphic in the current working
    #   directory.
    ###################################################################
    plt.savefig('runge-kutta-fehlberg.pdf', format='pdf')

    ###################################################################
    #   Show the plot in in a popup window.
    ###################################################################
    plt.show()

if __name__ == '__main__':
    main()
    exit(0)

