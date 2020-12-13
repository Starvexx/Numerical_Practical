#! /usr/bin/env python3
"""Direct Simulation Monte Carlo Demo

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        27.10.2020
    Last edited:    30.10.2020

This is a Demo-script to demonstrate the Direct Simulation Monte Carlo
Method. Here Pi is computed. This is done by creating a square domain
with a side length of l. Within this domain we define a circular disc
with radius r=l/2. Now a number of points are distributed arbitrarily
over the whole domain. By counting the number of points within the
disc, dividing that number by the total number of point and multiplying
by four, we get an approximation for Pi.

            n_total
        π ≈ -------- * 4
            n_inside

Where n_total is the total number of points and n_inside is the number
of points inside the circular disc.
"""

#######################################################################
#   Import packages.
#######################################################################

import warnings

import sys

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

    Methods:
    --------
    launch_particles(self, n):
        Launches n particles. If n is zero, the no_particles attribute
        flag is set to True.
    compute(self):
        Get the approximation for Pi.

    Attributes:
    self.size : scalar
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
        print('Object deleted.')


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
        ###############################################################
        #   Clear the domain.
        ###############################################################
        self.domain = np.zeros((self.size, self.size), dtype=int)

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


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine.

    Here the actual work is done, The computation runs ten times to
    show the random nature of the DSMC method. The mean of each run
    is stored to a numpy.array and finally printed to stdout.
    Additionally all plots are produced here.
    """

    print('Let´s have some Pi ;D')

    ###################################################################
    #   Create a new MyPi class object, get the maximum number of
    #   particles that will be generated and initialize the results
    #   array for ten individual runs.
    ###################################################################
    pi = MyPi(1001)
    n_max = int(sys.argv[1])
    pi_mean = np.zeros(10)

    ###################################################################
    #   Run the computation ten times to see the differences due to the
    #   random nature of the DSMC method.
    ###################################################################
    for i in range(10):
        ###############################################################
        #   Set up the progress bar.
        ###############################################################
        sys.stdout.write(f'Progress: [{"-" * 50}]')
        sys.stdout.flush()
        sys.stdout.write("\b" * 51)

        ###############################################################
        #   Create the result array for the individual runs. Each run
        #   will compute pi for all numbers of particles in n_max.
        ###############################################################
        results = np.zeros(n_max)
        for j, n in enumerate(range(n_max)):
            ###########################################################
            #   Update progress bar.
            ###########################################################
            if n % int(n_max / 50) == 0:
                sys.stdout.write("#")
                sys.stdout.flush()

            ###########################################################
            #   Clear domain, launch particles and compute the result.
            ###########################################################
            pi.launch_particles(n)
            results[j] = pi.compute()
        sys.stdout.write("]\n")

        ###############################################################
        #   Compute the mean of the current run.
        ###############################################################
        pi_mean[i] = np.mean(results)

    ###################################################################
    #   Compute the residual from the mean of each run.
    ###################################################################
    residual = pi_mean - np.pi

    upper_percentile = int(len(results) - len(results)/10)

    upper = np.max(results[upper_percentile:])
    lower = np.min(results[upper_percentile:])

    with open('results.txt', 'w') as out:
        out.write(f'Mean:\t{pi_mean}\nResults:\t{pi}\n'
                  + f'Residuals:\t{residual}')

    ###################################################################
    #   Clear the Domain and launch 1000 new particles. This is used
    #   for the Domain plot.
    ###################################################################
    pi.launch_particles(1000)

    print(f'Pi:\t\t{pi_mean[9]:.3f} +{(upper-pi_mean[9]):.3f} ' +
            f'-{(pi_mean[9]-lower):.3f}')
    print(f'Residual:\t{residual[9]}')

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
    fig = plt.figure(figsize=(7.27126, 2.5))       #   figsize in inches

    ###################################################################
    #   Plot the data.
    #   ax1: The computed values for pi.
    #   ax2: The residuals.
    #   ax2: The simulation domain.
    ###################################################################
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.grid(True, which='major', linewidth=0.5)
    ax2.grid(True, which='major', linewidth=0.5)
    ax3.grid(False)

    ax1.plot(np.arange(0, len(results), 1), results, lw=0.5)
    ax2.plot(np.arange(0, len(results), 1), results - np.pi, lw=0.5)
    ax3.scatter(pi.x, pi.y, s=0.2, color='red')
    ax3.imshow(pi.mask, alpha=0.5, cmap='brg')

    ###################################################################
    #   Format the subplot.
    ###################################################################

    props = dict(boxstyle='round',
                 facecolor='white',
                 edgecolor='gray',
                 linewidth=0.5,
                 alpha=1)
    ax1.text(0.5, 0.9, f'Mean: {pi_mean[9]:.3f}',
             transform=ax1.transAxes,
             verticalalignment='center',
             horizontalalignment='center',
             bbox=props)

    ax2.text(0.5, 0.9, f'Mean: {(pi_mean[9] - np.pi):.3f}',
             transform=ax2.transAxes,
             verticalalignment='center',
             horizontalalignment='center',
             bbox=props)

    ax1.set_title('DSMC Pi', size=12)
    ax1.set_xlabel('Number of points, M')
    ax1.set_ylabel('Calculated value of \(\pi\)')
    ax1.axhline(pi_mean[0], color='black', lw=0.5, alpha=0.5)

    ax2.set_title('DSMC Residuals', size=12)
    ax2.set_xlabel('Number of points, M')
    ax2.set_ylabel('Residuals: \(\pi - \mathrm{npumpy.pi}\)')
    ax2.axhline(pi_mean[0] - np.pi, color='black', lw=0.5, alpha=0.5)

    ax3.set_title('DSMC Domain', size=12)
    ax3.set_xlabel('x range')
    ax3.set_ylabel('y range')
    ax3.set_aspect(aspect='equal')
    ax3.axhline(pi.size / 2, color='black', lw=0.5, alpha=0.5)
    ax3.axvline(pi.size / 2, color='black', lw=0.5, alpha=0.5)

    fig.subplots_adjust(top=0.88,
                        bottom=0.175,
                        left=0.065,
                        right=0.965,
                        hspace=0.2,
                        wspace=0.32)

    ###################################################################
    #   Save the Figure to a file in the current working
    #   directory.
    ###################################################################
    plt.savefig('pi.pdf', format='pdf')

    ###################################################################
    #   Show the plot in in a popup window.
    ###################################################################
    plt.show()

if __name__ == '__main__':
    main()
    exit(0)

