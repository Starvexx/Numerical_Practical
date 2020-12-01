#! /usr/bin/env python3
"""Multiprocessing Direct Simulation Monte Carlo

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        30.11.2020
    Last edited:    30.11.2020

Description...
"""

#######################################################################
#   Import packages.
#######################################################################

import warnings

import sys

import numpy as np

from multiprocessing import Pool

from matplotlib import rc
from matplotlib import pyplot as plt

class MyPi:
    """A class to compute Pi.

    This is a class that utilizes the Direct Simulation Monte Carlo
    approach to compute the Mathematical constant Pi.

    Methods:
    --------
    clear_domain(self):
        Clears all particles from the domain.
    launch_particles(self, n):
        Launches n particles. If n is zero, the no_particles attribute
        flag is set to True.
    compute(self):
        Get the approximation for Pi.

    Attributes:
    self.size:
        The size of the domain.
    self.domain : 2D numpy.array
        The simulation domain is a two dimensional square of user
        defined size. The size is given when a new MyPi object is
        created and the constructor is called.
    self.mask : 2D numpy.array
        A circular boolean mask where the circles radius equals to half
        the size of the domain r = self.size / 2
    self.x : 1D numpy.array
        The x component of the launched particles.
    self.y : 1D numpy.array
        The y component of the launched particles.
    self.no_particles : scalar
        A boolean flag to determine if any particles are created.

    """

    def __init__(self, size):
        """Constructor.

        Creates a new MyPi object and initialized the simulation
        domain. The domain size is defined by the user and is passed
        as function argument. If a even number is passed, one is added
        to make for integer center values.
        In addition a circular boolean mask with the same dimensions as
        the domain is created. This will be used later to determine the
        particles within the circle.

        Parameters:
        -----------
        size : scalar
            The size of the simulation domain.
        """
        ###############################################################
        #   If the domain size is even make it odd and notify the
        #   user.
        ###############################################################
        if (size % 2) == 0:
            warnings.warn('Domain size is even. Adding 1 to make odd!',
                          UserWarning)
            size += 1

        self.size = size
        self.domain = np.zeros((size, size), dtype=int)
        self.mask = np.zeros((size, size), dtype=bool)

        ###############################################################
        #   Determine the distance of each position in the domain from
        #   the center. If the distance is less than the radius set the
        #   value in the Mask to True.
        ###############################################################
        __radius = int(self.size/2)
        __center = (__radius, __radius)
        ###############################################################
        #   __dom_pos are the indices of each position in the domain.
        ###############################################################
        __dom_pos = np.ogrid[0:self.size, 0:self.size]
        __cen_dist = np.sqrt(  (__dom_pos[0] - __center[0])**2
                             + (__dom_pos[1] - __center[1])**2)
        self.mask[__cen_dist <= __radius] = True


    def __del__(self):
        """Destructor."""
        print('MyPi destructed.')


    def clear_domain(self):
        """Reset the domain to all zeros."""
        self.domain = np.zeros((self.size, self.size), dtype=int)


    def launch_particles(self, n):
        """Launch particles.

        This method launches the user specified number of particles n.
        It uses numpy.randint() to generate random integers in the
        range of zero to self.size for x and y. A new random seed is
        used for the two components.

        Parameters:
        -----------
        n : scalar
            The number of particles generated.

        """
        if n != 0:
            ###########################################################
            #   Generate the x and y coordinates of the particles.
            ###########################################################
            seed()
            __x = randint(0, self.size, n)
            seed()
            __y = randint(0, self.size, n)
            ###########################################################
            #   Join the two components into an array where each entry
            #   holds one particles coordinates.
            ###########################################################
            __coord = np.stack([__x, __y], axis=1)
            ###########################################################
            #   Get unique coordinates and count multiplicity
            ###########################################################
            __coord_unique, __counts = np.unique(__coord, axis=0,
                                                 return_counts=True)
            ###########################################################
            #   Store the components to class attributes x and y to
            #   access them outside of the class.
            ###########################################################
            self.x = __coord_unique.T[0]
            self.y = __coord_unique.T[1]
            ###########################################################
            #   Place the particles in the domain according to their
            #   multiplicity. Each value in the Domain represents
            #   the number of particles at that location in the domain.
            ###########################################################
            self.domain[self.x, self.y] += __counts
            ###########################################################
            #   If n is not zero set this flag to False, in order to
            #   catch if zero particles are launched by the user.
            ###########################################################
            self.no_particles = False
        else:
            ###########################################################
            #   No particles generated.
            ###########################################################
            self.no_particles = True


    def compute(self):
        """compute Pi.

        Here Pi is computed by multiplying the fraction of the number
        of particles in the circle and the total number of particles by
        four.

        """
        ###############################################################
        #   If there are no particles generated, return zero.
        ###############################################################
        if self.no_particles:
            return 0
        ###############################################################
        #   Count the total number of particles in the domain.
        ###############################################################
        __sum2 = np.sum(self.domain)
        ###############################################################
        #   Count the number of particles in the circle.
        ###############################################################
        __sum1 = np.sum(self.domain[self.mask])
        ###############################################################
        #   Return the approximation for Pi.
        ###############################################################
        return (__sum1 / __sum2) * 4


def get_pi(obj, particles):
    obj.clear_domain()
    obj.launch_particles(particles)
    return obj.compute()

#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""
    test = MyPi(5)

    n_max = int(sys.argv[1])
    cores = int(sys.argv[2])

    particles = np.arange(0, n_max, 1, dtype=int)

    results = []




if __name__ == '__main__':
    main()
    exit(0)

