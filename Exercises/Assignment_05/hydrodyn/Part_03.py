#! /usr/bin/env python3
"""Hydrodynamic 1D simulator.

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        07.01.2021
    Last edited:    07.01.2021

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
#   Import user Package.
#######################################################################

from advection.flux_methods import BeamWarmingMethod as bwm
from advection.flux_methods import TwoStepLaxWendroff as lwm

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
            image_save_path = os.getcwd() + '\plots\Part_03'
            os.mkdir(image_save_path)
        except FileExistsError:
            print('Plot directory already exists.')
    else:
        ###############################################################
        #   Try to create the folder. If the folder already exists,
        #   notify the user and do nothing.
        ###############################################################
        try:
            image_save_path = os.getcwd() + '/plots/Part_03'
            os.mkdir(image_save_path)
        except FileExistsError:
            print('Plot directory already exists.')


def parse_args():
    """A simple argument parser to read commandline arguments."""
    possible_modes = ['u', 'l']
    arg_names = sys.argv[1::2]

    man_path = './man/hydro_manual.txt'

    if len(sys.argv) > 5:
        warnings.warn('Wrong number of arguments given.')
        with open(man_path) as helpfile:
            print(helpfile.read())
        exit(1)
    elif len(sys.argv) == 1:
        warnings.warn('No arguments given, defaulting to Lax-Wendroff.')
        return 'l', None

    if '-m' not in arg_names:
        print('ERROR: Too many missing arguments.')
        with open(man_path) as helpfile:
            print(helpfile.read())
        exit(1)
    elif ('-m' in arg_names) and ('-t' not in arg_names):
        idx_mode = sys.argv.index('-m')
        warnings.warn('Missing "-t" argument, using default')
        try:
            if sys.argv[idx_mode+1] in possible_modes:
                return sys.argv[idx_mode+1], None
            else:
                warnings.warn('Wrong method chosen, using Lax-Wendroff.')
                return 'l', None
        except IndexError:
            warnings.warn('No method chosen, using Lax-Wendroff.')
            return 'l', None
    elif ('-m' in arg_names) and ('-t' in arg_names):
        idx_mode = sys.argv.index('-m')
        idx_dt = sys.argv.index('-t')
        try:
            if sys.argv[idx_mode+1] in possible_modes:
                return sys.argv[idx_mode+1], sys.argv[idx_dt+1]
            else:
                warnings.warn('Wrong method chosen, using Lax Wendroff.')
                return 'l', sys.argv[idx_dt+1]
        except IndexError:
            warnings.warn('At least one argument missing, using Lax Wendroff'+\
                          ' with default time step')
            return 'l', None
    else:
        print('ERROR: Something went wrong.')
        with open(man_path) as helpfile:
            print(helpfile.read())
        exit(1)


def density_function(x):
    buff = np.zeros(np.shape(x)) + 0.125
    buff[0.35 <= x] = 1.0
    buff[0.65 <= x] = 0.125
    return buff


def velocity_function(x):
    buff = np.zeros(np.shape(x))
    return buff


def preassure_function(x):
    buff = np.zeros(np.shape(x)) + 0.1
    buff[0.35 <= x] = 1.0
    buff[0.65 <= x] = 0.1
    return buff


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
    MODEL = arguments[0]
    if arguments[1] is not None:
        dt = Fraction(arguments[1])
        delta_t = dt.numerator / dt.denominator
        del dt
    else:
        delta_t = 5/99


    ###################################################################
    #   Create a method instance which is a SimulationDomain subclass.
    #   There are three different models available:
    #       c: Center Difference Scheme
    #       u: Upstream Difference Scheme
    #       l: Lax-Wendroff Scheme
    #   If for some unknown reason none of these options are chosen,
    #   the program exits with a RuntimeError.
    ###################################################################
    if MODEL == 'l':
        simulation = lwm(function=density_function,
                         domain_range=(0, 1),
                         bins=1000)
        simulation.add_quantity(function=velocity_function)
        simulation.add_quantity(function=preassure_function)
    elif MODEL == 'b':
        pass
        # density = bwm(function=density_function,
        #               domain_range=(0, 1),
        #               bins=1000)
        # momentum_density = bwm(function=velocity_function,
        #                        domain_range=(0, 1),
        #                        bins=1000)
        # energy = bwm(function=preassure_function,
        #              domain_range=(0, 1),
        #              bins=1000)
    else:
        raise RuntimeError('Unknown simulation model. Exiting...')
        exit(1)

    ###################################################################
    #   Propagation...?
    ###################################################################

if __name__ == '__main__':
    main()
    exit(0)

