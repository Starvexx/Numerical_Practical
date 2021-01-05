#! /usr/bin/env python3
"""Project Name

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        04.01.2021
    Last edited:    05.01.2021

Main driver function for hydrodynamic advection simulations.
"""

#######################################################################
#   Import packages.
#######################################################################

import warnings

import os

import platform

import subprocess

import sys

import numpy as np

from matplotlib import rc
from matplotlib import pyplot as plt

#######################################################################
#   Import user defined packages.
#######################################################################

from advection.methods import CentralDifferenceScheme as cds
from advection.methods import UpstreamDifferencingScheme as uds
from advection.methods import LaxWendroffScheme as lws

#######################################################################
#   Define functions used in this file or package.
#######################################################################

def set_image_save_path(sysOS):
    """Set the image save path for any produced plots.

    This function sets the image save path and creates any missing
    directories if necessary.

    Parameters:
    -----------
    sysOS : string
        A string holding the information on which system the code is
        run.
    """

    ###################################################################
    #   Check if the script is run on a Unix or Windows machine and
    #   use correct syntax for each.
    ###################################################################
    if sysOS == 'Windows':
        ###############################################################
        #   Try to create the folder. If the folder already exists,
        #   notify the user and do nothing.
        ###############################################################
        try:
            image_save_path = os.getcwd() + '\plots'
            os.mkdir(image_save_path)
        except FileExistsError:
            print('Plot directory already exists.')
    else:
        ###############################################################
        #   Try to create the folder. If the folder already exists,
        #   notify the user and do nothing.
        ###############################################################
        try:
            image_save_path = os.getcwd() + '/plots'
            os.mkdir(image_save_path)
        except FileExistsError:
            print('Plot directory already exists.')


def parse_args():
    """A simple argument parser to read commandline arguments."""
    if len(sys.argv) > 3:
        warnings.warn('Wrong number of arguments given.')
        with open('./man/manual.txt') as helpfile:
            print(helpfile.read())
    elif len(sys.argv) == 1:
        warnings.warn('No arguments given, defaulting to Lax Wendroff.')
        return 'l'
    elif sys.argv[1] != '-m':
        with open('./man/manual.txt') as helpfile:
            print(helpfile.read())
    elif sys.argv[1] == '-m' and sys.argv[2] == 'c':
        return 'c'
    elif sys.argv[1] == '-m' and sys.argv[2] == 'u':
        return 'u'
    elif sys.argv[1] == '-m' and sys.argv[2] == 'l':
        return 'l'
    elif sys.argv[1] == '-m' and len(sys.argv) == 2:
        warnings.warn('No method chosen, defaulting to Lax Wendroff.')
        return 'l'


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""

    ###################################################################
    #   Set the SPATIAL_MODEL variable that determines which method is
    #   is going to be used in the simulation.
    ###################################################################
    SPATIAL_MODEL = parse_args()

    ###################################################################
    #   Create a method instance which is a SimulationDomain subclass.
    #   There are three different models available:
    #       c: Center Difference Scheme
    #       u: Upstream Difference Scheme
    #       l: Lax-Wendroff Scheme
    #   If for some unknown reason none of these options are chosen,
    #   the program exits with a RuntimeError.
    ###################################################################
    if SPATIAL_MODEL == 'c':
        simulation = cds(sim_time_steps=100)
    elif SPATIAL_MODEL == 'u':
        simulation = uds(sim_time_steps=100)
    elif SPATIAL_MODEL == 'l':
        simulation = lws(sim_time_steps=100)
    else:
        raise RuntimeError('Unknown simulation model. Exiting...')
        exit(1)

    ###################################################################
    #   Here the asdvection is simulated. The .propagate() method takes
    #   only two arguments, the time step length delta_t and the 
    #   velocity.
    ###################################################################
    simulation.propagate(delta_t=5/99, velocity=0.5)

    ###################################################################
    #   Get the machines operating system info and set the image save
    #   path for any plots that will be produced.
    ###################################################################
    sysOS = platform.system()
    set_image_save_path(sysOS)

    for i, step in enumerate(simulation.sim_spacetime):
        ###############################################################
        #   Set LaTeX font to default used in LaTeX documents.
        ###############################################################
        rc('font',
           **{'family':'serif',
              'serif':['Computer Modern Roman']})
        rc('text', usetex=True)

        ###############################################################
        #   Create new matplotlib.pyplot figure with subplots.
        ###############################################################
        fig = plt.figure(figsize=(6.3, 3.54))   #   figsize in inches

        ###############################################################
        #   Plot the data.
        ###############################################################
        ax1 = fig.add_subplot(111)

        ax1.grid(True, which='major', linewidth=0.5)

        ax1.plot(simulation.x, simulation.sim_spacetime[0], label='initial')
        ax1.plot(simulation.x, step, color='red', label='propagated')

        ###############################################################
        #   Format the subplot.
        ###############################################################
        ax1.set_title(f'CDS @ n = {i:03}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('q(x)')
        ax1.set_ylim(-2, 3)
        ax1.legend()

        ###############################################################
        #   Save the Figure to a file in the current working
        #   directory.
        ###############################################################
        plt.savefig(f'./plots/frame_{i:03}.png', format='png')

        ###############################################################
        #   Save one frame in the middle as a snapshot for later usage.
        ###############################################################
        if i == int(simulation.sim_time_steps / 2):
            plt.savefig(f'./plots/{type(simulation).__name__}' + \
                        f'_snapshot_{i:03}.pdf', format='pdf')
        plt.close(fig)

        ###############################################################
        #   Show the plot in in a popup window.
        ###############################################################

    ###################################################################
    #   Animate the results from the simulation using ImageMagick and
    #   delete the single frames. If ImageMagick is inot available on
    #   the machine, notify the user and keep the individual frames.
    ###################################################################
    animation_name = 'animation_' + type(simulation).__name__ + '.gif'
    if sysOS == 'Linux':
        cmd = f'convert -delay 5 ./plots/frame_*.png -loop 0 {animation_name}' \
            + '&& rm ./plots/*.png'
        try:
            _ = subprocess.check_output(cmd, shell=True,
                                        stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            warnings.warn('Unable to create animation with ImageMagick.')

if __name__ == '__main__':
    main()
    exit(0)

