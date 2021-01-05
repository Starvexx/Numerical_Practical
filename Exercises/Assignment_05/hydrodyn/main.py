#! /usr/bin/env python3
"""Project Name

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        04.01.2021
    Last edited:    04.01.2021

Description...
"""

#######################################################################
#   Import packages.
#######################################################################

import warnings

import numpy as np

from matplotlib import rc
from matplotlib import pyplot as plt

#######################################################################
#   Import user defined packages.
#######################################################################

from advection.methods import CentralDifferenceScheme as cds

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
    simulation = cds(sim_time_steps=100)
    simulation.propagate(5/99, 0.5)
    print(simulation.sim_spacetime[80])

    for i, step in enumerate(simulation.sim_spacetime):
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
        ax1 = fig.add_subplot(111)

        ax1.grid(True, which='major', linewidth=0.5)

        ax1.plot(simulation.x, simulation.sim_spacetime[0], label='initial')
        ax1.plot(simulation.x, step, color='red', label='propagated')

        ###################################################################
        #   Format the subplot.
        ###################################################################
        ax1.set_title(f'CDS @ n = {i:03}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('q(x)')
        ax1.set_ylim(-2, 3)
        ax1.legend()

        ###################################################################
        #   Save the Figure to a file in the current working
        #   directory.
        ###################################################################
        plt.savefig(f'frame_{i:03}.png', format='png')
        plt.close(fig)

        ###################################################################
        #   Show the plot in in a popup window.
        ###################################################################

if __name__ == '__main__':
    main()
    exit(0)

