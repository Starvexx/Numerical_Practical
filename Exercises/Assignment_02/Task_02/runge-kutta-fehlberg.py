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

def get_k(function,
          time,
          y_n,
          delta_t,
          butcher_tableau):
    """Compute k_i coefficients.

    This function computes the k coefficients from a Butcher Tableau.

    Parameters:
    -----------
    function : function
        the derivative y'(t, y) of the function y(t) that is to be
        determined.
    time : scalar
        The current time t of the time step Δt.
    y_n : scalar
        The current function value from which the function value
        y_(n+1) will be derived.
    delta_t : scalar
        The time step length of the current time step.
    butcher_tableau : numpy.array
        This array contains the Butcher Tableau.

    Returns:
    k : numpy.array
        An array containing all k values computed from the Butcher
        Tableau and y'(t, y)

    """
    ###################################################################
    #   Initialize a new array that will hold the values for k_i. The
    #   length of the array is determined from the Butcher Tableau.
    ###################################################################
    k = np.zeros(len(butcher_tableau[0]))

    ###################################################################
    #   Compute the individual values of k.
    ###################################################################
    for i, line in enumerate(butcher_tableau[2]):
        k[i] = function(time + delta_t*butcher_tableau[0][i],
                        y_n + delta_t*np.dot(line, k))

    return k


def optimize(function,
             time,
             t_end,
             y_n,
             delta_t,
             tolerance,
             butcher_tableau,
             logfile,
             rec_depth=0):
    """Optimizes time step.

    This function optimizes length of the time step depending on the
    chosen tolerance. This is done by computing the Local Truncation
    Error (LTE) and comparing it to the LTE tolerance as follows:

        Δt_opt = a * Δt * abs(LTE_tol/LTE)^(1/5)                (1)

    Where LTE_tol is the user specified tolerance times y_n:

        LTE_tol = tol * y_n

    To get the LTE, it is necessary to determine the 4th and 5th order
    next approximations y4_(n+1) and y5_(n+1), assuming the
    Runge-Kutta-Fehlberg method. They are defined as is shown below:

        y4_(n+1) = y_n + Δt * (     25/216   * k1               (2)
                               +  1408/2565  * k3
                               +  2197/4104  * k4
                               -     1/5     * k5)

        y5_(n+1) = y_n + Δt * (     16/135   * k1               (3)
                               +  6656/12825 * k3
                               + 28561/56430 * k4
                               -     9/50    * k5
                               +     2/55    * k6)

    Obviously for a different Runge-Kutta method equations (2) and (3)
    look different, as each Butcher Tableau is unique to its method.
    However, from Equations (2) and (3) the LTE is calculated:

        LTE = y5_(n+1) - y4_(n+1)

    Finally if Δt_opt is less than Δt the optimization is repeated
    with Δt = Δt_opt, until Δt_opt is greater than Δt and the code can
    advance to the next time step with Δt = Δt_opt and y_n = y5_(n+1).
    If no repeated optimization takes place the code advances
    immediately with Δt = Δt_opt and y_n = y5_(n+1).

    Parameters:
    -----------
    function : function
        The derivative y'(t, y) of the function y(t) that is to be
        determined.
    time : scalar
        The current time t of the time step.
    t_end : scalar
        The end time of the total integration interval.
    y_n : scalar
        Current function value y(t) from which the next function value
        y_(n+1)(t) is derived.
    delta_t : scalar
        The current time step length Δt.
    tolerance : scalar
        The tolerance factor used for the LTE_tol.
        LTE_tol = tolerance * y_n
    butcher_tableau : numpy.array
        A numpy array containing the Butcher Tableau of the RK-method.
    logfile : string
        The file path / file name of the logfile. All intermediate
        results are dumped into that file.
    rec_depth : scalar
        The number of function calls.

    Returns:
    --------
    t : scalar
        The updated time time for the next time step.
    y_new : scalar
        The next function value y_(n+1).
    delta_t : scalar
        The optimized time step.
    lte : scalar
        The Local Truncation Error after the last optimization.
    """

    ###################################################################
    #   Get the k coefficients for the computation.
    ###################################################################
    k = get_k(function=function,
              time=time,
              y_n=y_n,
              delta_t=delta_t,
              butcher_tableau=butcher_tableau)

    ###################################################################
    #   Compute y_(n+1). For RK45 this are the fourth and fifth order
    #   approximations. They are stored on a numpy.array where the
    #   first entry is the fifth order and the second entry the fourth
    #   order.
    ###################################################################
    y_npo = y_n + delta_t * np.dot(butcher_tableau[1], k)

    ###################################################################
    #   Compute the LTE and LTE_tol.
    ###################################################################
    lte = y_npo[0] - y_npo[1]
    lte_tol = tolerance * y_n

    ###################################################################
    #   If the end step has been reached go ahead and return the final
    #   results.
    ###################################################################
    if time == t_end:
        return time, y_npo[0], delta_t, lte

    ###################################################################
    #   Optimize the time step.
    ###################################################################
    dt_opt = 0.999 * delta_t * (np.abs(lte_tol/lte))**(1/5)

    ###################################################################
    #   Write everything to logfile.
    ###################################################################
    if logfile is not None:
        with open(logfile, 'a') as log:
            log.write(f'\nt:\t\t  {time}\n')
            log.write(f'y_n:\t  {y_n}\n')
            log.write(f'Δt:\t\t  {delta_t}\n')
            log.write(f'tol:\t  {tolerance}\n')
            log.write(f'k:\t\t  {k}\n')
            log.write(f'y5_(n+1): {y_npo[0]}\n')
            log.write(f'y4_(n+1): {y_npo[1]}\n')
            log.write(f'lte:\t  {lte}\n')
            log.write(f'lte_tol:  {lte_tol}\n')
            log.write(f'Δt_opt:\t  {dt_opt}\n')

    ###################################################################
    #   Check if the optimization was successful, if the optimization
    #   criterion is not met, keep optimizing. This is done
    #   recursively. Also if the time step update exceeds the end time
    #   set delta_t to t_end - time and set the time to the end time.
    #   Then compute y_(n+1) for the end time.
    ###################################################################
    if delta_t < dt_opt and (time + dt_opt) > t_end:
        ###############################################################
        #   Compute final result.
        ###############################################################
        delta_t = t_end - time
        time = t_end

        return optimize(function=function,
                        time=time,
                        t_end=t_end,
                        y_n=y_n,
                        delta_t=delta_t,
                        tolerance=tolerance,
                        logfile=logfile,
                        butcher_tableau=butcher_tableau,
                        rec_depth=rec_depth)

    elif delta_t > dt_opt:
        ###############################################################
        #   Keep optimizing.
        ###############################################################
        rec_depth += 1

        return optimize(function=function,
                        time=time,
                        t_end=t_end,
                        y_n=y_n,
                        delta_t=dt_opt,
                        tolerance=tolerance,
                        logfile=logfile,
                        butcher_tableau=butcher_tableau,
                        rec_depth=rec_depth)

    else:
        ###############################################################
        #   Return results and move on to the next time step.
        ###############################################################
        if logfile is not None:
            with open(logfile, 'a') as log:
                st = '#' * 79
                log.write(f'\n{st}\nOptimized Δt {rec_depth} times...\n{st}\n')
        time += dt_opt      #   Update time for the next time step.

        ###############################################################
        #   Return results from optimization
        ###############################################################
        return time, y_npo[0], dt_opt, lte


def rk45(function,
         init_cond=0.,
         tolerance=1e-3,
         init_t_step=1e-1,
         t_start=0.,
         t_end=20.,
         logfile='rk45.log'):
    """Runge-Kutta-Fehlberg method.

    description...

    Parameters:
    -----------
    function : function
        The derivative y'(t) of the function y(t) that shall be
        determined.
    init_cond : scalar
        The initial condition of the differential equation.
    tolerance : scalar
        The local truncation error tolerance LTE_tol that shall be used
        to work out the optimal time step t_opt.
    init_t_step : scalar
        The initial time step guess.
    t_start : scalar
        Start time for the integration.
    t_end : scalar
        End time for the integration.
    logfile : string
        The name of the logfile to store intermediate results to.

    Returns:
    --------
    result : numpy.array
        This is the results array. It holds the pairs of approximated
        function values and the corresponding times, as well as the
        time step lengths and the Local Truncation Errors.
    """
    ###################################################################
    #   Initialize result array as list with the initial conditions.
    #   Computational results will be appended to it. It will be
    #   converted to a numpy.array in the end.
    ###################################################################
    result = [np.array([t_start, init_cond, init_t_step, 0])]

    ###################################################################
    #   Define the Butcher Tableau for the Runge-Kutta_Fehlberg
    #   method.
    ###################################################################
    A = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    B = np.array([[16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],
                  [25/216, 0,  1404/2565,   2197/4104,  -1/5,    0]])
    C = np.array([np.zeros(6),
                  [      1/4,          0,          0,         0,     0, 0],
                  [     3/32,       9/32,          0,         0,     0, 0],
                  [1932/2197, -7200/2197,  7296/2197,         0,     0, 0],
                  [  439/216,         -8,   3680/513, -845/4104,     0, 0],
                  [    -8/27,          2, -3544/2565, 1859/4104,-11/40, 0]])

    butcher_tableau = np.array([A, B, C])

    ###################################################################
    #   Compute the time steps until the end time is reached.
    ###################################################################
    i = 1
    while result[i-1][0] < t_end:
        t = result[i-1][0]
        y_n = result[i-1][1]
        dt = result[i-1][2]

        ################################################################
        #   Write current time of the time step to the logfile.
        ################################################################
        if logfile is not None:
            st = 79*'#'
            with open(logfile, 'a') as log:
                log.write(f'\n{st}\nCurrent time of time step:\t{t}\n{st}\n')

        step_results = optimize(function=function,
                                time=t,
                                t_end=t_end,
                                y_n=y_n,
                                delta_t=dt,
                                tolerance=tolerance,
                                butcher_tableau=butcher_tableau,
                                logfile=logfile)

        result.append(np.array(step_results))
        i += 1

    ###################################################################
    #   Return result list as numpy.array.
    ###################################################################
    return np.array(result)

#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""

    log = False
    ###################################################################
    #   Set up the differential equation and the initial condition.
    ###################################################################
    y0 = 1
    y_dot = lambda t, y : 0.1 * y + np.sin(t)

    ###################################################################
    #   Set up the time interval within the solution is approximated.
    ###################################################################
    t_start = 0
    t_end = 20

    ###################################################################
    #   The tolerances for which the algorithm is tested for.
    ###################################################################
    tolerances = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    ###################################################################
    #   Apply the ODE solver with different tolerances.
    ###################################################################
    for tol in tolerances:
        ###############################################################
        #   Create log files for the different runs.
        ###############################################################
        if log is not False:
            with open(f'tol_{tol}.log', 'w') as log:
                st = '#' * 79
                log.write(f'{st}\n{st}\n')
                log.write(f'\nRK45 LOGFILE\n\nTolerance:\t{tol} * y_n\n\n')
                log.write(f'{st}\n{st}\n')

    results = np.array([rk45(function=y_dot,
                             init_cond=y0,
                             tolerance=tol,
                             init_t_step=0.5,
                             t_start=t_start,
                             t_end=t_end,
                             logfile=None) for tol in tolerances])
                             # logfile=f'tol_{tol}.log') for tol in tolerances])

    ###################################################################
    #   Write results from the different runs to individual text files.
    ###################################################################
    for tol, result in zip(tolerances, results):
        with open(f'tol_{tol}.txt', 'w') as output:
            st = '#' * 79
            output.write(f'{st}\n{st}\n')
            output.write(f'\nRK45 RESULTS\n\nTolerance:\t{tol} * y_n\n\n')
            output.write(f'{st}\n{st}\n\n')
            output.write(f't\t, y(t)\t, Δt\t, LTE\n')
            for line in result:
                output.write(f'{line[0]:.6f}, ')
                output.write(f'{line[1]:.6f}, ')
                output.write(f'{line[2]:.6f}, ')
                output.write(f'{line[3]:.6f}\n')


    ###################################################################
    #   Set LaTeX font to default used in LaTeX documents.
    ###################################################################
    rc('font',
       **{'family':'serif',
          'serif':['Computer Modern Roman']},
       size = 9)
    rc('text', usetex=True)

    ###################################################################
    #   Create new matplotlib.pyplot figure with subplots.
    ###################################################################
    fig = plt.figure(figsize=(3.55659, 8))       #   figsize in inches

    ###################################################################
    #   Plot the data.
    ###################################################################
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.grid(True, which='major', linewidth=0.5)
    ax2.grid(True, which='major', linewidth=0.5)
    ax3.grid(True, which='major', linewidth=0.5)

    for result, tol in zip(results, tolerances):
        ax1.plot(result[:, 0],
                 result[:, 1],
                 label=str(tol))
        ax2.plot(result[:, 0],
                 result[:, 2],
                 label=str(tol))

    n_steps = np.array([len(a) for a in results])
    ax3.plot(n_steps[:6], tolerances[:6])

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax1.set_title('RK45 approximations')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y(t)')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.48), ncol=3)

    ax2.set_xlabel('t')
    ax2.set_ylabel('\(\Delta\)t')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.48), ncol=3)

    ax3.set_xlabel('steps')
    ax3.set_ylabel('\(\mathrm{LTE}_\mathrm{tol}\)')
    ax3.set_yscale('log')

    plt.subplots_adjust(top=0.965,
                        bottom=0.055,
                        left=0.17,
                        right=0.97,
                        hspace=0.485,
                        wspace=0.2)

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

