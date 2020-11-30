#! /usr/bin/env python3
"""

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
from numpy.random import seed
from numpy.random import randint

from matplotlib import rc
from matplotlib import pyplot as plt

#######################################################################
#   Classes used in this file.
#######################################################################

class MyPi:
    """A class to compute Pi.

    This is a class that utilizes the Direct Simulation Monte Carlo
    approach to compute the Mathematical constant Pi.
    """

    def __init__(self, size):
        if (size % 2) == 0:
            warnings.warn('Domain size is even. Adding 1 to make odd!',
                          UserWarning)
            size += 1

        self.size = size
        self.domain = np.zeros((size, size), dtype=int)
        self.mask = np.zeros((size, size), dtype=bool)

    def clear_domain(self):
        self.domain = np.zeros((self.size, self.size), dtype=int)


    def launch_particles(self, n):
        if n != 0:
            seed()
            x = randint(0, self.size, n)
            seed()
            y = randint(0, self.size, n)
            coord = np.stack([x, y], axis=1)
            coord_unique, counts = np.unique(coord, axis=0, return_counts=True)
            self.x = coord_unique.T[0]
            self.y = coord_unique.T[1]
            self.domain[self.x, self.y] += counts
            self.init=False
        else:
            self.init = True


    def compute(self):
        if self.init:
            return 0

        __radius = int(self.size/2)
        __center = (__radius, __radius)
        __dom_pos = np.ogrid[0:self.size, 0:self.size]
        __cen_dist = np.sqrt(  (__dom_pos[0] - __center[0])**2
                             + (__dom_pos[1] - __center[1])**2)
        self.mask[__cen_dist <= __radius] = True
        sum2 = np.sum(self.domain)
        sum1 = np.sum(self.domain[self.mask])
        return (sum1/sum2) * 4


#######################################################################
#   Define functions used in this file or package.
#######################################################################

def template():
    """Template function"""
    pass

#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""
    test = MyPi(1001)
    pi = []
    for n in range(500000):
        if n>=1:
            test.clear_domain()
            test.launch_particles(n)
            pi.append(test.compute())

    pi = np.array(pi)

    print(np.nanmedian(pi))

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
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.grid(True, which='major', linewidth=0.5)
    ax2.grid(True, which='major', linewidth=0.5)

    ax1.plot(np.arange(0, len(pi), 1), pi, lw=0.5)
    ax2.plot(np.arange(0, len(pi), 1), np.pi - pi, lw=0.5)

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax1.set_title('DSMC Pi')
    ax1.set_xlabel('Number of points, M')
    ax1.set_ylabel('Calculated value of \(\pi\)')
    ax1.legend()

    ###################################################################
    #   Save the Figure to a file in the current working
    #   directory.
    ###################################################################
    plt.savefig('pi.pdf', format='pdf')
    
    ###################################################################
    #   Show the plot in in a popup window.
    ###################################################################
    # plt.show()

if __name__ == '__main__':
    main()
    exit(0)

