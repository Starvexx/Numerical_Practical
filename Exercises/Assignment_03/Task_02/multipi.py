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
from numpy.random import seed
from numpy.random import randint

from time import time
from time import sleep

from multiprocessing import Pool
from multiprocessing import cpu_count

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
        # print('Object deleted.')
        pass


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



def get_pi(particles):
    """Wrapper function to compute pi using a MyPi Class Object"""
    pi = MyPi(1001)
    pi.launch_particles(particles)
    result = pi.compute()
    del pi
    return result


def parallelize(task, arguments, cores):
        pool = Pool(cores)
        results = []
        for argument in arguments:
            results.append(pool.apply_async(task, [argument]))

        sys.stdout.write(f'Progress: [{"-" * 50}]')
        sys.stdout.flush()
        sys.stdout.write("\b" * 51)

        while True:
            complete = sum(1 for result in results if result.ready())
            comp_percent = 100 * complete / len(results)

            if complete % int(len(results) / 50) == 0:
                sys.stdout.write("#" * int(comp_percent / 2))
                sys.stdout.write("\b" * int(comp_percent / 2))
                sys.stdout.flush()

            if complete == len(results):
                break

        sys.stdout.write("#" * 50 + "]\n")
        pool.close()
        pool.join()

        return results


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine.

    Three general modes of operation can be chosen.
        - Normal: Runs the code with the specified amount of CPU cores,
                  physical or logical. (CL arguments ranging from one
                  to the maximum number of cores)
        - Auto:   Runs the code with the maximum available cores,
                  physical or logical. (CL argument 0)
        - Bench:  Runs the code in benchmark mode. The code is run
                  multiple times, each time with an increased number of
                  CPU cores to assert the performance increase.
                  (CL argument is negative)
    """
    n_max = int(sys.argv[1])
    cores = int(sys.argv[2])
    try:
        max_cores = cpu_count()
    except:
        print('Unable to get maximum number of cores.')
        print('Setting max cores to 1!')
        print('Automatic and Benchmark modes compromised.')
        print('Running on single core...')
        max_cores = 1

    ###################################################################
    #   Check if a valid number of cores was selected. The number of
    #   cores defines the mode of the program.
    ###################################################################
    if cores == 0 or cores > max_cores:
        if cores > max_cores:
            warnings.warn(f'Invalid number of CPU cores chosen. '
                          + f'Using {max_cores} cores!', UserWarning)
        cores = max_cores
        benchmark = False
    elif cores < 0:
        cores = max_cores
        benchmark = True
    else:
        benchmark = False

    particles = np.arange(0, n_max, 1, dtype=int)

    if benchmark:
        print("Benchmark mode entered...")
        print("Assessing performance for various numbers of CPU cores.")
        runtimes = np.zeros(max_cores)
        for i, core in enumerate(np.arange(1, cores + 1, 1)):
            t_start = time()
            print(f'Running with {core} cores...')
            res = parallelize(get_pi, particles, core)
            runtimes[i] = time() - t_start

        for i, runtime in enumerate(runtimes):
            print(f'\nRuntime with {i + 1} cores used:\t{runtime:.3f}\n'
                    + f'\t\t  Speedup:\t{runtimes[0]/runtime:.3f}')
    else:
        print(f'Running with {cores} cores...')
        res = parallelize(get_pi, particles, cores)

    results = []
    for result in res:
        results.append(result.get())

    results = np.array(results)
    upper_percentile = int(len(results) - len(results)/10)

    upper = np.max(results[upper_percentile:])
    lower = np.min(results[upper_percentile:])
    residuals = results - np.pi
    pi_mean = np.mean(results)
    print(f'Pi is {pi_mean:.3f} +{(upper-pi_mean):.3f} -{(pi_mean-lower):.3f}')

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
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.grid(True, which='major', linewidth=0.5)
    ax2.grid(True, which='major', linewidth=0.5)

    ax1.plot(np.arange(0, len(results), 1), results, lw=0.5)
    ax2.plot(np.arange(0, len(results), 1), results - np.pi, lw=0.5)

    ###################################################################
    #   Format the subplot.
    ###################################################################

    props = dict(boxstyle='round',
                 facecolor='white',
                 edgecolor='gray',
                 linewidth=0.5,
                 alpha=1)
    ax1.text(0.5, 0.9, f'Mean: {pi_mean:.3f}',
             transform=ax1.transAxes,
             verticalalignment='center',
             horizontalalignment='center',
             bbox=props)

    ax2.text(0.5, 0.9, f'Mean: {(pi_mean - np.pi):.3f}',
             transform=ax2.transAxes,
             verticalalignment='center',
             horizontalalignment='center',
             bbox=props)

    ax1.set_title('DSMC Pi', size=12)
    ax1.set_xlabel('Number of points, M')
    ax1.set_ylabel('Calculated value of \(\pi\)')
    ax1.axhline(pi_mean, color='black', lw=0.5, alpha=0.5)

    ax2.set_title('DSMC Residuals', size=12)
    ax2.set_xlabel('Number of points, M')
    ax2.set_ylabel('Residuals: \(\pi - \mathrm{npumpy.pi}\)')
    ax2.axhline(pi_mean - np.pi, color='black', lw=0.5, alpha=0.5)

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
    plt.savefig('multipi_8_cores.pdf', format='pdf')

    ###################################################################
    #   Show the plot in in a popup window.
    ###################################################################
    # plt.show()


if __name__ == '__main__':
    main()
    exit(0)

