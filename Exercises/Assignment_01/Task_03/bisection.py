#! /usr/bin/env python3
"""Bisection method

    Author:     David Hernandez
    Matr. Nr.:  01601331
    e-Mail:     david.hernandez@univie.ac.at

Another method for solving equations is the bisection method. Just
like the Newtin method, it is an iterative approach that hones in on
the solution.
"""

#######################################################################
#   Import pachages.
#######################################################################

import warnings

import numpy as np

from matplotlib import rc
from matplotlib import pyplot as plt

from astropy import constants as const
from astropy import units as u

#######################################################################
#   Define functions used in this file or package.
#######################################################################
def bisection():
    """Bisection method for equation solving.

    Maybe a more detailed description of how this works.
    """
    pass


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""

    ###################################################################
    #   Define lambda functions for the speed of sound c_s, the radius
    #   of the critical point r_c and a velocity unit.
    ###################################################################
    c_s = lambda T : np.sqrt(np.divide(const.k_B * T,
                                       0.5 * const.m_p))
    r_c = lambda c_sound : np.divide(const.G * const.M_sun,
                                 2*np.square(c_sound))
    mps = u.m / u.s

    ###################################################################
    #   Define the function with dependence on the velocity and the
    #   radial distance from the star.
    ###################################################################
    parker =  lambda v, r

    ###################################################################
    #   For debugging.
    ###################################################################
    # print(r_c(c_sound(10*u.K)).to(u.au))
    # print(c_sound(10*u.K).to(mps))



if __name__ == '__main__':
    main()
    exit(0)


