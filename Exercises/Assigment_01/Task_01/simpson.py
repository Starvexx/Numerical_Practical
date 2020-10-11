#! /usr/bin/env python3

###############################################################################
#
#                               Simpson-integrator
#
#   Author:     David Hernandez
#   Matr. Nr.:  01601331
#   e-Mail:     david.hernandez@univie.ac.at
#
#   This is a simple script that demonstrates a numerical integrator using
#   Simpson's rule. This method of integration is an approximation of the
#   definite integral of a function f(x), approximated by an easier to 
#   integrate polynomial. One polynomial often used is the quadratic poly-
#   nomial, a simple parabola. There are also higher order polynomials used,
#   but the shall not be incorporated here.
#
#   In general, higher accurracy is achieved by dividing the interval [a,b]
#   within the integral shall be determined into a number of subintervals.
#   The number n of subdivisions determines the number of terms used in the
#   approximation, as the general formula is given as follows:
#
#   ⌠a
#   ⎮ f(x) dx ≈ Δx/3 * (f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + 2f(x_4) + ...
#   ⌡b
#               + 4f(x_(n-1)) + f(x_n))
#
#   Where Δx = (b - a) / n and x_i = a + i * Δx. (Source: Wikipedia, see link
#                                                 below for further details.)
#
#   https://en.wikipedia.org/wiki/Simpson%27s_rule
#
###############################################################################

##
##  Import packages used in this script.
##

import numpy as np

##
##  Define functions used in this script.
##

def integrate(n=1, upper=1, lower=0):
    """ Test 
    """
    pass

if __name__ == '__main__':
    print(f'Henlo wurld!')
