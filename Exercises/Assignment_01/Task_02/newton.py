#! /usr/bin/env python3
"""Newton iterator

    Author:     David Hernandez
    Matr. Nr.:  01601331
    e-Mail:     david.hernandez@univie.ac.at

The Newton iterative method is one way to evaluate the zero points of
a function, i.e. solves the following equation incrementally.

    solve(f(x) = 0, x)

For this method to work, the user has to provide two informations.
Firstly a initial guess of the zero point, and secondly a tolerance
range within the result may lie. Typically this tolerance is of the
order of 10^-3 to 10^-7.
"""

#######################################################################
#   Import packages.
#######################################################################

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
           iters=1):
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
    if interim_res is not None:
        interim_res.append(next_approx)

    ###################################################################
    #   Break condition:
    #       Either the next approximation is within the user defined
    #       tolerance or the maximum recursion depth is reached with
    #       no convergence. If the solution does not converge a
    #       RecursionError is raised.
    ###################################################################
    if np.abs(next_approx - init) < tolerance:
        print(f'Solution converged after {iters} iterations.\n')
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


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""
    
    ###################################################################
    #   Define the first function and its derivative. They will be
    #   passed to the newton-function along with the interim_1 list.
    #   The list will hold the interim results for the praphical
    #   representation of the results.
    ###################################################################
    f_1 = lambda x : x**3 - x + 1
    df_1 = lambda x : 3*(x**2) - 1
    interim_1 = []

    ###################################################################
    #   Evaluate the result of the first equation.
    ###################################################################
    result_1 = newton(init=0,
                      function=f_1,
                      derivative=df_1,
                      tolerance=1e-5,
                      interim_res=interim_1)
    
    ###################################################################
    #   Define the second functiuon and its derivative.
    ###################################################################
    f_2 = lambda x : np.cos(x) - 2*x
    df_2 = lambda x : -np.sin(x) - 2
    interim_2 = []

    ###################################################################
    #   Evaluate the result of the second equation.
    ###################################################################
    result_2 = newton(init = 0,
                      function=f_2,
                      derivative=df_2,
                      tolerance=1e-5,
                      interim_res=interim_2)

    ###################################################################
    #   Set LaTeX font to default used in LaTeX documents.
    ###################################################################
    rc('font',
       **{'family':'serif',
          'serif':['Computer Modern Roman']})
    rc('text', usetex=True)

    ###################################################################
    #   Create new Matplotlib.pyplot figure with one subplot.
    ###################################################################
    fig = plt.figure(figsize=(6.3, 3.54))       #   figsize in inches
    ax = fig.add_subplot(111)

    ax.grid(True, which='both',     #   Draw grid
            linewidth=0.5)

    ax.axhline(linewidth=0.5,         #   Draw actual solution for the
             color='red',           #   first equation.
             y=-1.3247,
             label='solution f1')
    ax.scatter(range(len(interim_1)),   #   Plot results from numerical
               interim_1,               #   approximation.
               marker='x',
               color='red',
               label='approximation f1')

    ax.axhline(linewidth=0.5,         #   Draw actual solution for the
             color='blue',          #   second equation.
             y=0.4502,
             label='solution f2')
    ax.scatter(range(len(interim_2)),   #   Plot results from numerical
               interim_2,               #   approximation.
               marker='x',
               color='blue',
               label='approximation f2')

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax.set_xlabel('iterations')
    ax.set_ylabel('result')
    ax.set_xlim(0, 20.5)
    ax.set_ylim(-4, 4)
    ax.set_title('Results from Newton solver')
    ax.legend()

    ###################################################################
    #   Position the subplot within the figure.
    ###################################################################
    #plt.subplots_adjust(left=0.11,
    #                    bottom=0.12,
    #                    right=0.9,
    #                    top=0.88)

    ###################################################################
    #   Save the figure as vector graphic in the current working
    #   directory.
    ###################################################################
    plt.savefig('newton_results.svg', format='svg')
    ###################################################################
    #   Show the plot in a popup window.
    ###################################################################
    plt.show()


if __name__ == '__main__':
    main()
    exit(0)
