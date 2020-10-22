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
    c_s = lambda T : np.sqrt((const.k_B * T) / (0.5 * const.m_p))
    r_c = lambda c_sound : (const.G * const.M_sun) / (2 * c_sound**2)
    mps = u.m / u.s

    ###################################################################
    #   For debugging.
    ###################################################################
    # print(r_c(c_sound(10*u.K)).to(u.au))
    # print(c_sound(10*u.K).to(mps))

    ###################################################################
    #   Define the function with dependence on the velocity and the
    #   radial distance from the star.
    ###################################################################
    parker =  lambda v, r, t : c_s \
                               * (r_c(c_s(t)) / r)**2 \
                               * np.exp(-(2 * r_c(c_s(t))) / r + 3 / 2) \
                               - v * np.exp(-v**2 / (2 * c_s(t)**2))

    ###################################################################
    #   Set the radial bins from the center (2R_sol) to the outer
    #   rim (1AU). This array will be iterated over to determine the
    #   solar wind for each distance.
    #   Also set the temperatures for which the the wind shall be
    #   determined.
    ###################################################################
    radii = np.linspace(2 * u.R_sun, 1 * u.au, 99, True)
    temps = np.linspace(2e6 * u.K, 10e6 * u.K, 5, True)

    ###################################################################
    #   Compute the wind velocities.
    #   TODO: Finish computation of velocities, also sth is wron with
    #         parker function .. I think. Dont know whats goin on here.
    ###################################################################
    for r in radii:
        for t in temps:
            bisection(parker(v, r, t),
                      )

    print(temps)

if __name__ == '__main__':
    main()
    exit(0)


