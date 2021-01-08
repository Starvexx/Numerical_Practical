#! /usr/bin/env python3
"""Project Name

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        07.01.2021
    Last edited:    07-01-2021

Description...
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

from fractions import Fraction

from matplotlib import rc
from matplotlib import pyplot as plt

#######################################################################
#   Import user files.
#######################################################################

from advection.flux_methods import DonorCellMethod as dcm
from advection.flux_methods import FrommsMethod as fm
from advection.flux_methods import BeamWarmingMethod as bwm
from advection.flux_methods import LaxWendroffMethod as lwm

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
    ###################################################################
    #   d: Donor Cell Method   (no slope)
    #   f: Fromm Method        (central slope)
    #   b: Beam Warming Method (upwind slope)
    #   d: Lax Wendroff Method (downwind slope)
    ###################################################################
    possible_modes = ['d', 'f', 'b', 'fl']
    arg_names = sys.argv[1::2]

    if len(sys.argv) > 5:
        warnings.warn('Wrong number of arguments given.')
        with open('./man/flux_manual.txt') as helpfile:
            print(helpfile.read())
        exit(1)
    elif len(sys.argv) == 1:
        warnings.warn('No arguments given, defaulting to Lax-Wendroff.')
        return 'fl', 5/99

    if '-m' not in arg_names:
        print('ERROR: Too many missing arguments.')
        with open('./man/flux_manual.txt') as helpfile:
            print(helpfile.read())
        exit(1)
    elif ('-m' in arg_names) and ('-t' not in arg_names):
        idx_mode = sys.argv.index('-m')
        warnings.warn('Missing "-t" argument, using delta_t = 5/99')
        try:
            if sys.argv[idx_mode+1] in possible_modes:
                return sys.argv[idx_mode+1], 5/99
            else:
                warnings.warn('Wrong method chosen, using Lax-Wendroff.')
                return 'fl', 5/99
        except IndexError:
            warnings.warn('No method chosen, using Lax-Wendroff.')
            return 'fl', 5/99
    elif ('-m' in arg_names) and ('-t' in arg_names):
        idx_mode = sys.argv.index('-m')
        idx_dt = sys.argv.index('-t')
        try:
            if sys.argv[idx_mode+1] in possible_modes:
                return sys.argv[idx_mode+1], sys.argv[idx_dt+1]
            else:
                warnings.warn('Wrong method chosen, using Lax Wendroff.')
                return 'fl', sys.argv[idx_dt+1]
        except IndexError:
            warnings.warn('At least one argument missing, using Lax Wendroff'+\
                          ' with delta_t = 5/99')
            return 'fl', 5/99
    else:
        print('ERROR: Something went wrong.')
        with open('./man/flux_manual.txt') as helpfile:
            print(helpfile.read())


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""

    arguments = parse_args()

    ###################################################################
    #   Set the SPATIAL_MODEL variable that determines which method is
    #   is going to be used in the simulation.
    ###################################################################
    FLUX_MODEL = arguments[0]
    dt = Fraction(arguments[1])
    delta_t = dt.numerator / dt.denominator

    ###################################################################
    #   Create a method instance which is a SimulationDomain subclass.
    #   There are three different models available:
    #       c: Center Difference Scheme
    #       u: Upstream Difference Scheme
    #       l: Lax-Wendroff Scheme
    #   If for some unknown reason none of these options are chosen,
    #   the program exits with a RuntimeError.
    ###################################################################
    if FLUX_MODEL == 'd':
        print('Using Donor Cell...')
        simulation = dcm(sim_time_steps=100)
    elif FLUX_MODEL == 'f':
        simulation = fm(sim_time_steps=100)
    elif FLUX_MODEL == 'b':
        simulation = bwm(sim_time_steps=100)
    elif FLUX_MODEL == 'fl':
        simulation = lwm(sim_time_steps=100)
    else:
        raise RuntimeError('Unknown simulation model. Exiting...')
        exit(1)

    ###################################################################
    #   Here the asdvection is simulated. The .propagate() method takes
    #   only two arguments, the time step length delta_t and the 
    #   velocity.
    ###################################################################
    simulation.propagate(delta_t=delta_t, velocity=-0.5)

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
        ax1.legend(loc='upper left')

        props = dict(boxstyle='round',
                     facecolor='white',
                     edgecolor='gray',
                     linewidth=0.5,
                     alpha=1)

        ax1.text(0.025, 0.05, f'\(\Delta t =\) {delta_t:.3f}',
                 transform=ax1.transAxes,
                 bbox=props)

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

