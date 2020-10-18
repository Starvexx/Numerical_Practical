#! /usr/bin/env python3
"""Simpson integrator.

   Author:     David Hernandez
   Matr. Nr.:  01601331
   e-Mail:     david.hernandez@univie.ac.at

This is a simple script that demonstrates a numerical integrator using
Simpson's rule. This method of integration is an approximation of the
definite integral of a function f(x), approximated by an easier to
integrate polynomial. One polynomial often used is the quadratic poly-
nomial, a simple parabola. There are also higher order polynomials
used, but the shall not be incorporated here.

In general, higher accurracy is achieved by dividing the interval [a,b]
within the integral shall be determined into a number of subintervals.
The number n of subdivisions determines the number of terms used in the
approximation, as the general formula is given as follows:

⌠b
⎮ f(x) dx ≈ Δx/3 * (f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + 2f(x_4) +
⌡a
           ... + 4f(x_(n-1)) + f(x_n))

Where Δx = (b - a) / n and x_i = a + i * Δx. (Source: Wikipedia, see
                                              link below for further
                                              details.)

https://en.wikipedia.org/wiki/Simpson%27s_rule
"""


#######################################################################
#   Import packages used in this script.
#######################################################################

import types

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt

#######################################################################
#   Define functions used in this script.
#######################################################################

def simpson(n=1, lower=0, upper=1, function=lambda x : np.sin(x)):
    """ Integrate numerically using the simpson rule.

    This function does the numerical integration using simpsons rule.

    Parameters
    ----------
    n : scalar
        The number of intervals or integrationsteps.
    lower : scalar
        The lower boundary of the definite integral.
    upper : scalar
        The upper boundary of the definite integral.
    function : numpy.array or function
        Holds all the function values for the function to be
        integrated. The array needs to be populated with an odd number
        of values.
        This parameter can also take a function.

    Returns
    -------
    result : scalar
        The result of the numerical integration.
    """

    ###################################################################
    #   Distinguish between numpy.array and lambda function. If none
    #   of these types are given a TypeError is raised and the program
    #   is terminated.
    ###################################################################
    if isinstance(function, types.FunctionType):
        ###############################################################
        #   Compute the points where the function has to be evaluated.
        ###############################################################
        eval_pts = 2 * n + 1

        ###############################################################
        #   Check if the number of eval_pts is even or odd. When it is
        #   even add one to make odd.
        ###############################################################
        if eval_pts % 2 == 0:
            eval_pts += 1

        ###############################################################
        #   Populate fvals with the function values at the eval_pts.
        ###############################################################
        fvals = np.array([function(x) for x in np.linspace(lower,
                                                           upper,
                                                           eval_pts,
                                                           True)])
    elif isinstance(function, np.ndarray):
        ###############################################################
        #   Populate with function values.
        ###############################################################
        fvals = function
    else:
        err_msg = 'Invalid data type. function must me a '\
                  + 'numpy.array or function.'
        raise TypeError(err_msg)

    ###################################################################
    #   Compute bin width.
    ###################################################################
    delta_x = (upper - lower) / n

    ###################################################################
    #   By slicing the fval array smart, the functionvalues with the
    #   odd and even indices are separated so that they can be used
    #   conveniently in simpsons rule.
    ###################################################################
    result = delta_x / 6 * (fvals[0]
                            + fvals[2*n]
                            + 4 * np.sum(fvals[1::2])
                            + 2 * np.sum(fvals[2:2*n-1:2]))
    return result


def pretty_print(expression=None, lower=None, upper=None, value=None,
                 steps=None):
    """Print the result of the integral in a pretty way."""

    print(f'The following integral evaluates to\n'
          + f'⌠{upper}\n'
          + f'⎮ {expression} dx = {value}\n'
          + f'⌡{lower}')


def plotting(results, steps):
    """Plot the results with respect to integrationsteps.

    Parameters
    ----------
    results : list or numpy.array
        The results that are to be plotted.
    steps : list or np.array
        The number of steps that correspond to the achived results.
    """

    ###################################################################
    #   Set LaTeX font to default used in LaTeX documents.
    ###################################################################
    rc('font',
       **{'family':'serif',
          'serif':['Computer Modern Roman']})
    rc('text', usetex=True)

    ###################################################################
    #   Create new matplotlib.pyplot figure with one subplot.
    ###################################################################
    fig = plt.figure(figsize=(6.3, 3.54))       #   figsize in inches
    ax = fig.add_subplot(111)

    ax.grid(True, which='both',     #   Draw grid
            linewidth=0.5)

    ax.axhline(linewidth=0.5,       #   Draw analytic solution
               color='black',
               y=4 + np.pi,
               label='analytic solution')
    ax.scatter(steps,               #   Plot results from numerical
               results,             #   integration
               marker='x',
               label='numeric solution')

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax.set_xscale('log')
    ax.set_xlabel('steps')
    ax.set_ylabel('result')
    ax.set_ylim(7.13, 7.35)
    ax.set_title('Results using different integration steps')
    ax.legend()

    ###################################################################
    #   Position and the subplot within the figure.
    ###################################################################
    plt.subplots_adjust(left=0.11,
                        bottom=0.12,
                        right=0.9,
                        top=.88)

    ###################################################################
    #   Save the figure as vector graphic in the current working
    #   directory.
    ###################################################################
    plt.savefig('simpson_results.svg', format='svg')
    ###################################################################
    #   Show the plot in popup window.
    ###################################################################
    plt.show()


def main():
    """ Main subroutine."""

    a = 0                           #   upper integration boundary
    b = np.pi                       #   lower integration boundary

    steps_list = [1, 2, 5,          #   list of different integration
                  10, 20, 50,       #   steps
                  100, 200, 500,
                  1000, 2000, 5000,
                  10000]

    ###################################################################
    #   The function that is to be integrated.
    ###################################################################
    f = lambda x : 2 * np.sin(x) + 1

    ###################################################################
    #   Populate the results list with the results when different
    #   numbers of integrationsteps are used.
    ###################################################################
    results = [simpson(n=steps,
                       lower=a,
                       upper=b,
                       function=f) for steps in steps_list]

    ###################################################################
    #   Compute the result of the integral using 10000 steps.
    ###################################################################
    result = simpson(n=steps_list[12],
                     lower=a,
                     upper=b,
                     function=f)

    ###################################################################
    #   Print the result using multiline unicode output.
    ###################################################################
    pretty_print(expression='2*sin(x)+1',
                 lower=0,
                 upper='π',
                 value=result)
    print(f'with {steps_list[12]} integrationsteps.')

    ###################################################################
    #   Plot the results when using different numbers of integration
    #   steps.
    ###################################################################
    plotting(results,
             steps_list)


if __name__ == "__main__":
    main()
    exit(0)
