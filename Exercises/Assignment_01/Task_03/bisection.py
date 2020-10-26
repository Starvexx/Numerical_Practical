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
#   Import packages.
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
def bisection(function=lambda x : np.sin(x),
              lower=np.pi/2,
              upper=np.pi * 3/4,
              tolerance=10e-5,
              rec_depth=1):
    """Bisection method for equation solving.

    TODO: Adapt return value for universal use. At this moment the
          value attribute of a astropy.quantity object is returned!
          If changed adapt docstring to the changes!

    This method for equation solving depends on a function f(x) in an
    interval [a, b]. If the the function evaluated at a has a different
    sign compared to it evaluated at b, there is a solution for
    f(x) = 0 | a < x < b. In the next step the function is evaluated at
    c = a + ((a + b) / 2) and we determine if the solution lies to the
    left of c or to the right. This again is done by analyzing the
    signs of f(a), f(c) and f(b). If the solution lies in the interval
    [a, c], f(a) will have a different sign from f(c). If the solution
    is to be found in the interfal [c, b], f(c) will have a sign
    differing from f(b). This analysis is repeated until f(c) is
    smaller than a tolerance value. Usually the tolerance is a value
    between 10e-4 and 10e-7. Since this method depends on iteration, a
    recursive approach was chosen.

    Parameters:
    -----------
    function : lambda function or function
        The function f(x) in the equation f(x) = 0 for which the
        solutiopn x shall be determined.
    lower : scalar
        The lower boundary of the solution interval [a, b].
    upper : scalar
        The upper boundary of the solution interval [a, b].
    tolerance : scalar
        The maximum deviation from the true result of the equation.
    rec_depth : scalar
        Recursion counter, is only to be used within the function to
        count the function calls. This parameter shall in general not
        be changed by the user. However, if a problem may be solved
        by a larger recursion depth, a negave value may be set by the
        user in order to increase the maximum number of recursions.
        Use with caution.

    Returns:
    --------
    mid.value : Value attribute of astropy.quantity object
        The result determined by the bisection method.
    """

    ###################################################################
    #   Check if the upper and lower boundaries, a and b, were given 
    #   correctly. If not, they are switched and a warning is raised to
    #   stdout.
    ###################################################################
    if lower > upper:
        wrn_msg = 'Lower boundary is greater than upper boundary.'\
                  + 'Switching boundaries, result may be incorrect!'
        warnings.warn(wrn_msg)
        buffer = lower
        lower = upper
        upper = buffer

    ###################################################################
    #   Calculate the midpoint c of the interval.
    ###################################################################
    mid = lower + ((upper - lower) / 2)

    ###################################################################
    #   Evaluate the function at the three points a, b and c.
    ###################################################################
    f_lower = function(lower)
    f_mid = function(mid)
    f_upper = function(upper)

    ###################################################################
    #   Define a fuinction to determine the sign of a value. Returns
    #   True if the value is negative and False if the value is
    #   positive.
    ###################################################################
    sign = lambda x : (x / np.abs(x)) == -1

    ###################################################################
    #   Set up maximum recursion depth limit and count the function
    #   calls.
    ###################################################################
    rec_depth += 1
    max_depth = 100

    ###################################################################
    #   Recursion depth break condition. Raises Recursion error if the
    #   result does not converge after 100 iterations.
    ###################################################################
    if rec_depth > max_depth:
        err_msg = f'Maximum of {max_depth} iterations passed and no '\
                  + f'convergence occured.'
        raise RecursionError(err_msg)

    ###################################################################
    #   Compute the solution using a recursive approach. Main break
    #   condition is achieved when f(c) is smaller then the tolerance
    #   given by the user. If the solution is in the interval [a, c]
    #   this function is recursed with new boundaries, a = a and b = c.
    #   If the solution is in the interval [c, b], this function is
    #   recursed with a = c and b = b.
    ###################################################################
    if np.abs(f_mid) < tolerance:
        print(f'Solution converged after {rec_depth} iterations.\n')
        return mid.value
    else:
        if (sign(f_lower) != sign(f_mid)):
            return bisection(function=function,
                             lower=lower,
                             upper=mid,
                             tolerance=tolerance,
                             rec_depth=rec_depth)
        elif (sign(f_mid) != sign(f_upper)):
            return bisection(function=function,
                             lower=mid,
                             upper=upper,
                             tolerance=tolerance,
                             rec_depth=rec_depth)


def plot(x, y):
    """A simple plotting function.

    This function plots the results of the calculation of the parker
    wind.

    Parameters:
    -----------
    x : numpy array like or list
        The x values for the plot.
    y : numpy array like or list
        The y values for the plot.
    """
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
    ax = fig.add_subplot(111)

    ax.grid(True, which='both', linewidth=0.5)

    ###################################################################
    #   Define different colors, labels and linestyles for the
    #   different temperatures.
    ###################################################################
    colors=['red', 'blue', 'brown', 'magenta', 'cyan']
    labels=['2 MK', '4 MK', '6 MK', '8 MK', '10 MK']
    styles=[(0, (5, 1)), ':', '-.', (0, (3, 2, 1, 2, 1, 2)), '--']
    for temp, color, label, style in zip(y, colors, labels, styles):
        ax.plot(x, temp, color=color, label=label, ls=style)

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax.set_title('Parker wind speeds for different temperatures')
    ax.set_xlabel('distance [\(\mathrm{R}_\odot\)]')
    ax.set_ylabel('velocity [\(\mathrm{km}/\mathrm{s}\)]')
    ax.legend(ncol=2)

    ###################################################################
    #   Position the subplot within the figure.
    ###################################################################
    plt.subplots_adjust(left=0.11,
                        bottom=0.15,
                        right=0.9,
                        top=0.88)

    ###################################################################
    #   Save the Figure as vector graphic in the current working
    #   directory.
    ###################################################################
    plt.savefig('parker_results.svg', format='svg')

    ###################################################################
    #   Show the plot in in a popup window.
    ###################################################################
    plt.show()


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine to test the bisection algorithm."""

    ###################################################################
    #   Define lambda functions for the speed of sound c_s, the radius
    #   of the critical point r_c and a velocity unit.
    ###################################################################
    c_s = lambda T : np.sqrt((const.k_B * T) / (0.5 * const.m_p))
    r_c = lambda T : (const.G * const.M_sun) / (2 * c_s(T)**2)
    mps = u.m / u.s
    kps = u.km / u.s

    ###################################################################
    #   Define the function with dependence on the velocity and the
    #   radial distance from the star.
    ###################################################################
    parker= lambda v, r, t : c_s(t) \
                             * (r_c(t) / r)**2 \
                             * np.exp(-(2 * r_c(t)) / r + 3 / 2) \
                             - v * np.exp(-v**2 / (2 * c_s(t)**2))

    ###################################################################
    #   Set the radial bins from the center (2R_sol) to the outer
    #   rim (1AU). This array will be iterated over to determine the
    #   solar wind for each distance.
    #   Also set the temperatures for which the the wind shall be
    #   determined.
    ###################################################################
    r_1 = (2 * u.R_sun)
    r_2 = (1 * u.au).to(u.R_sun)

    radii = np.linspace(r_1, r_2, 100, True)
    temps = np.linspace(2e6 * u.K, 10e6 * u.K, 5, True)

    ###################################################################
    #   Fill a 2D list with the results from the bisection method for 
    #   the Parker Wind, where the first dimension are for the 
    #   different temperatures and the second dimension are for the 
    #   different radii. In essence this yields a 5×100 matrix with the
    #   lambda functions for different temperatures and radii as the
    #   matrix elements. A distinction had to be made for the case of
    #   supersonic and subsonic winds. For distances smaller than the
    #   critical radius r_c(T) we expect the solution to lie somewhere
    #   between 0 km/s and c_s(T), wheras for distances greater than 
    #   r_c the speeds lie between c_s(T) and an arbitraty high upper
    #   limit. Here c_s(T) is the speed of sound at temperature T and
    #   r_c(T) is the critical radius for certain sound speed c_s(T),
    #   hence the dependence on T for the critical radius.
    ###################################################################
    result = [[],[],[],[],[]]
    for line, t in zip(result, temps):
        for r in radii:
            if r < r_c(t):
                line.append(bisection(function=lambda v : parker(v,r,t),
                                      lower=0*kps,
                                      upper=c_s(t).to(kps),
                                      tolerance=10e-5*kps))
            else:
                line.append(bisection(function=lambda v : parker(v,r,t),
                                      lower=c_s(t).to(kps),
                                      upper=20000*kps,
                                      tolerance=10e-5*kps))

    ###################################################################
    #   Plot the results from above.
    ###################################################################
    plot(radii, result)


if __name__ == '__main__':
    main()
    exit(0)
