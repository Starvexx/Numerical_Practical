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


##
##  Import packages used in this script.
##

import types

import numpy as np

##
##  Define functions used in this script.
##

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
    #
    #   Distinguish between numpy.array and lambda function. If none
    #   of these types are given a TypeError is raised and the program
    #   is terminated.
    #
    ###################################################################
    if isinstance(function, types.FunctionType):
        eval_pts = 2 * n + 1
        if eval_pts % 2 == 0:
            eval_pts += 1
        fvals = np.array([function(x) for x in np.linspace(lower,
                                                           upper,
                                                           eval_pts,
                                                           True)])
    elif isinstance(function, np.ndarray):
        fvals = function
    else:
        err_msg = 'Invalid data type. function must me a '\
                  + 'numpy.array or function.'
        raise TypeError(err_msg)

    delta_x = (upper - lower) / n

    ###################################################################
    #
    #   By slicing the fval array smart, the functionvalues with the
    #   odd and even indices are separated so that they can be used
    #   conveniently in simpsons rule.
    #
    ###################################################################
    result = delta_x / 6 * (fvals[0] 
                            + fvals[2*n]
                            + 4 * np.sum(fvals[1::2])
                            + 2 * np.sum(fvals[2:2*n-1:2]))
    return result


def main():
    """ Main subroutine.."""

    a = 0
    b = np.pi
    steps = 1000

    f = lambda x : 2 * np.sin(x) + 1

    result = simpson(n=steps,
                     lower=a,
                     upper=b,
                     function=f)

    print(f'The following integral evaluates to\n'
          + f'⌠π\n'
          + f'⎮ 2*sin(x) + 1 dx = {result}\n'
          + f'⌡0\n'
          + f'using {steps} integration steps.')


if __name__ == "__main__":
    main()
    exit(0)
