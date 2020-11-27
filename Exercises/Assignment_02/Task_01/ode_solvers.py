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

The main() function is a test suite that compares and benchmarks the
different methods with regard to the number of time steps used in
the integration. The differential equation and the initial condition
used for testing can be seen below:

    y'(t)  = -2 * t * y
    y(t=0) = 1
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
#   Functions:
#       newton(init, function, derivative, tolerance, interim_res,
#              iters, verbosity)
#       forward_euler(derivative, init_cond, t_start, t_end, n_steps)
#       classical_runge_kutta(derivative, init_cond, t_start, t_end,
#                             n_steps)
#       backward_euler(f, f_prime, init_cond, t_start, t_end, n_steps)
#       crank_nicolson(f, f_prime, init_cond, t_start, t_end, n_steps)
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
    implemented to get the result within a user defined tolerance.

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
    max_depth = 500
    # init = 0
    ###################################################################
    #   Compute the next approximation of the solution.
    ###################################################################
    next_approx = init - (function(init) / derivative(init))
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
        return next_approx
    elif iters > max_depth:
        err_msg = f'Maximum of {max_depth} iterations passed and no '\
                  + f'convergence occurred.'
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

    This method is a simple explicit ODE solver. It computes the
    following result for the time step from the current approximation:

        y_(n+1) = y_n + Δt * y'_n

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
    result = np.empty(n_steps + 1)
    t = t_start

    for i in range(n_steps + 1):
        if t == t_start:
            result[i] = init_cond
        else:
            result[i] = result[i-1] + t_step * derivative(t - t_step, result[i-1])
        t += t_step
    return result


def classical_runge_kutta(derivative,
                          init_cond=0.,
                          t_start=0.,
                          t_end=3.,
                          n_steps=100):
    """Classical Runge Kutta Method.

    This method is also explicit and computes the next approximation
    from the current value.

        y_(n+1) = y_n + Δt/6 * (k_1 + 2 * (k_2 + k_3) + k_4)

    Where k_i are coefficients that correspond to the slopes of y(t)
    at different places in the time step. k_1 is at the beginning,
    k_2 and k_3 are in the middle and k_4 is at the end. They can be
    determined as follows:

        k_1 = y'(t, y_n)
        k_2 = y'(t + Δt/2, y_n + (Δt + k_1)/2)
        k_3 = y'(t + Δt/2, y_n + (Δt + k_2)/2)
        k_4 = y'(t + Δt, y_n + Δt * k3)

    Parameters:
    -----------
    derivative : function
        The derivative y'(t) of the function y(t) that shall be
        determined,
    init_cond : scalar
        The initial condition of the differential equation.
    t_start : scalar
        The integration start time.
    t_end : scalar
        The integration end time.
    n_steps : scalar
        The number of integration time steps.

    Returns:
    --------
    result : numpy.array
        The result of the integration. This are the approximated
        function values y(t).

    """
    t_step = (t_end - t_start) / n_steps
    result = np.empty(n_steps + 1)
    t = t_start

    for i in range(n_steps + 1):
        if t == t_start:
            result[i] = init_cond
        else:
            ###########################################################
            #   Compute the Coefficients k_1 to k_4
            ###########################################################
            k_1 = derivative(t - t_step, result[i-1])
            k_2 = derivative(t - t_step + 1/2 * t_step,
                             result[i-1] + 1/2 * t_step * k_1)
            k_3 = derivative(t - t_step + 1/2 * t_step,
                             result[i-1] + 1/2 * t_step * k_2)
            k_4 = derivative(t - t_step + t_step,
                             result[i-1] + t_step * k_3)

            ###########################################################
            #   Compute the result for this time step.
            ###########################################################
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
    """Backward Euler Method.

    This is an implicit method, meaning that it estimates the next
    result from an approximation of the next function value. This can
    be achieved by utilizing a Newton Iterator. The next value is
    computed as follows:

        y_(n+1) = y_n + Δt * y'(t + Δt, y_(n+1))                (1)

    To get y_(n+1) above equation can be rearranged as such:

        y_(n+1) - y_n - Δt * y'(t + Δt, y_(n+1)) = 0            (2)

    Now y_(n+1) can be estimated by using the Newton equation solver.
    This, however, needs the derivative of above equation (2) with
    respect to y_(n+1) resulting in:

        1 - Δt * d/dy_(n+1) (y'(t + Δt, y_(n+1))) = 0           (3)

    Assuming that the left side of Equation (2) is G(y) and the left
    side of Equation (3) is G'(y), the estimator for y_(n+1) can be
    determined by using Newtons method.

    Parameters:
    -----------
    f : function
        This function is the derivative of the actual function that is
        to be determined. f is in essence y'(t).
    f_prime : function
        This is the derivative of f. It is needed for the newton solver
        used in the process of determining y(t).
    init_cond : scalar
        This is the initial condition of the differential equation.
    t_start : scalar
        The start time of the integration.
    t_end : scalar
        The end time of the integration.
    n_steps : scalar
        The number of integration steps used.

    Returns:
    --------
    result : numpy.array
        This are the results y(t) for the individual time steps.
    """
    t_step = (t_end - t_start) / n_steps
    result = np.empty(n_steps + 1)
    t = t_start

    for i in range(n_steps + 1):
        if t == t_start:
            result[i] = init_cond
        else:
            ###########################################################
            #   Define G(y) and G'(y) that are needed for the Newton
            #   iterator.
            ###########################################################
            G = lambda y_next : y_next - result[i-1]\
                                - t_step * f(t, y_next)
            G_prime = lambda y_next : 1 - t_step * f_prime(t, y_next)

            ###########################################################
            #   Approximate y_(n+1) using the Newton iterator. y_next
            #   will be used to determine the result for the next time
            #   step.
            ###########################################################
            y_next = newton(init=result[i-1],
                            function=G,
                            derivative=G_prime,
                            tolerance=1e-5)

            ###########################################################
            #   Compute the function value of the next time step.
            ###########################################################
            result[i] = result[i-1] + t_step * f(t, y_next)

        t += t_step

    return result


def crank_nicolson(f,
                   f_prime,
                   init_cond=0.,
                   t_start=0.,
                   t_end=0.,
                   n_steps=100):
    """Crank-Nicolson Method.

    This implicit method can be interpreted as the mean of the Forward
    and Backward Euler methods. The next time step can be computed as
    follows:

        y_(n+1) = y_n + Δt/2 * (f(t, y_n) + f(t + Δt, y_(n+1)))

    Parameters:
    -----------
    f : function
        The derivative y'(t) of the y(t) that is to be approximated.
    f_prime : function
        The derivative f'(t, y) of f(t, y) with respect to y. Expressed
        differently:
            f'(t,y) = d/dy y'(t)
    init_cond : scalar
        The initial condition of the differential equation.
    t_start : scalar
        The start time of the integration.
    t_end : scalar
        The end time of the integration.
    n_steps : scalar
        The number of time steps used in the integration.

    Returns:
    --------
    result : numpy.array
        The approximated function values y(t) for the time steps
        specified in n_steps.
    """
    t_step = (t_end - t_start) / n_steps
    result = np.empty(n_steps + 1)
    t = t_start

    for i in range(n_steps + 1):
        if t == t_start:
            result[i] = init_cond
        else:
            ###########################################################
            #   Define G(y) and G'(y) that are needed for the Newton
            #   iterator.
            ###########################################################
            G = lambda y_next : y_next - result[i-1]\
                                - t_step * f(t, y_next)
            G_prime = lambda y_next : 1 - t_step * f_prime(t, y_next)

            ###########################################################
            #   Approximate y_(n+1) using the Newton iterator. y_next
            #   will be used to determine the result for the next time
            #   step.
            ###########################################################
            y_next = newton(init=result[i-1],
                            function=G,
                            derivative=G_prime)

            ###########################################################
            #   Compute the function value of the next time step.
            ###########################################################
            result[i] = result[i-1] + 1/2 * t_step * (f(t - t_step, result[i-1])
                                                      + f(t, y_next))

        t += t_step

    return result

#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine.

    This is a test suite to compare the performance of different ODE
    solving algorithms. Four different numerical solvers are
    implemented, the Forward Euler and Classical Runge Kutta as
    explicit methods and the Backward Euler and Crank Nicolson as
    implicit methods. These four methods are then applied to the
    following initial value problem:

        y(t=0) = 1
        dy/dt  = -2 * t * y = y_dot

    For the explicit methods that is all that is needed. However, to
    get a result from the implicit methods the value for the next time
    step has to be estimated. This can be done by using the Newton
    method, which is an iterative equation solver. For this to work
    we need a function f(y), in our case that is y_dot, and iters
    derivative with respect to y. Hence the need for a function called
    y_dot_prime. Even though in this case that derivative is -2 * t,
    the y parameter was passed to the function to maintain the general
    form of that function which may include a dependency on y.
    """

    ###################################################################
    #   y_dot is the derivative of the function that shall be deter-
    #   mined by the ODE solver.
    #   y_dot_prime is the derivative of y_dot with respect to y. This
    #   is needed for the newton iterator used by the two implicit 
    #   methods.
    ###################################################################
    y_dot = lambda t, y : -2 * t * y
    y_dot_prime = lambda t, y : -2 * t + 0 * y

    ###################################################################
    #   time_steps is a list containing the different number of time 
    #   steps to compare.
    ###################################################################
    time_steps = np.array([2, 5, 10, 30, 100])
    y_fe = []   #   forward euler results for each number of time steps
    y_crk = []  #   runge kutta results for each number of time steps
    y_be = []   #   backwrd euler results for each number of time steps
    y_cn = []   #   crank nicolsn results for each number of time steps

    ###################################################################
    #   Here the different methods are applied to the differential
    #   equation using different numbers of time steps so that they
    #   can be compared to each other.
    ###################################################################
    for steps in time_steps:
        y_fe.append(forward_euler(y_dot,
                                  init_cond=1,
                                  t_start=0,
                                  t_end=3,
                                  n_steps=steps))

        y_crk.append(classical_runge_kutta(y_dot,
                                           init_cond=1,
                                           t_start=0,
                                           t_end=3,
                                           n_steps=steps))

        y_be.append(backward_euler(f=y_dot,
                                   f_prime=y_dot_prime,
                                   init_cond=1,
                                   t_start=0,
                                   t_end=3,
                                   n_steps=steps))

        y_cn.append(crank_nicolson(f=y_dot,
                                   f_prime=y_dot_prime,
                                   init_cond=1,
                                   t_start=0,
                                   t_end=3,
                                   n_steps=steps))

    ###################################################################
    #   Join the results from the different methods with different time
    #   time steps into one list for easier plotting.
    ###################################################################
    methods = [y_fe, y_crk, y_be, y_cn]

    ###################################################################
    #   Set up the analytic solution to compare it with the numerical
    #   methods.
    ###################################################################
    t_ana = np.linspace(start=0,
                        stop=3,
                        num=1000,
                        endpoint=True)
    y = lambda t : np.exp(-(t**2))
    y_ana = np.array([y(t) for t in t_ana])



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
    fig = plt.figure(figsize=(6.3, 5))       #   figsize in inches
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    axes = [ax1, ax2, ax3, ax4]
    names = ['Forward Euler',
             'Classical Runge Kutta',
             'Backward Euler',
             'Crank-Nicolson']

    ###################################################################
    #   Plot the results from the different methods in this loop.
    ###################################################################
    for axis, method, name in zip(axes, methods, names):
        axis.grid(True, which='major', linewidth=0.5)

        ###############################################################
        #   Plot the analytic solution.
        ###############################################################
        axis.plot(t_ana,
                  y_ana,
                  color='cyan',
                  label='analytic solution')

        ###############################################################
        #   Define different line styles. In general all are dash
        #   dotted, but the number of dots increase with the number
        #   of time steps.
        ###############################################################
        styles=[(0, (3, 2, 1, 2)),
                (0, (3, 2, 1, 2, 1, 2)),
                (0, (3, 2, 1, 2, 1, 2, 1, 2)),
                (0, (3, 2, 1, 2, 1, 2, 1, 2, 1, 2)),
                (0, (3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2))]

        ###############################################################
        #   Plot results for the different numbers of time steps.
        ###############################################################
        for i, n_steps in enumerate(time_steps):
            t = np.linspace(start=0,
                            stop=3,
                            num=n_steps + 1,
                            endpoint=True)
            axis.plot(t,
                     method[i],
                     ls=styles[i],
                     label=f'{n_steps} steps')

        ###############################################################
        #   Set axis limits.
        ###############################################################
        axis.set_ylim(-0.1,1.1)
        axis.set_xlim(0, 3)

        ###############################################################
        #   Label the individual plots.
        ###############################################################
        props = dict(boxstyle='round',
                     facecolor='white',
                     edgecolor='gray',
                     linewidth=0.5,
                     alpha=1)
        axis.text(0.5, 0.9, name,
                  transform=axis.transAxes,
                  verticalalignment='center',
                  horizontalalignment='center',
                  bbox=props)

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax1.set_ylabel('y(t)')
    ax1.set_xticklabels([])
    ax1.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0, 1, 2, 3])
    ax3.set_ylabel('y(t)')
    ax3.set_xlabel('t')
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels([0, 1, 2, 3])
    ax4.set_xlabel('t')
    ax4.set_yticklabels([])
    ax4.set_xticks([0, 1, 2, 3])
    ax4.set_xticklabels([0, 1, 2, 3])

    ###################################################################
    #   place legend. One for all plots is used.
    ###################################################################
    ax4.legend(loc='lower center',
               bbox_to_anchor=(-0.045, -0.51),
               ncol=3)

    ###################################################################
    #   Position the subplot within the figure and set spacing between
    #   the plots.
    ###################################################################
    plt.subplots_adjust(left=0.095,
                        bottom=0.185,
                        right=0.975,
                        top=0.94,
                        hspace=0.075,
                        wspace=0.075)
    fig.suptitle('Four different numerical ODE solvers', size = 11)
    ###################################################################
    #   Save the Figure as pdf in the current working directory.
    ###################################################################
    plt.savefig('ODE_solvers.pdf', format='pdf')

    ###################################################################
    #   Show the plot in in a pop up window.
    ###################################################################
    plt.show()


if __name__ == '__main__':
    main()
    exit(0)

