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


def lax_wendroff(time_steps):
    r = np.zeros((time_steps, 1004))
    v = np.zeros((time_steps, 1004))
    p = np.zeros((time_steps, 1004))
    rv = np.zeros((time_steps, 1004))
    e = np.zeros((time_steps, 1004))

    dx = 1 / 999

    x = np.arange(-2 * dx, 1 + 3 * dx, dx)

    for i, xx in enumerate(x):
        r[0][i] = 1 if 0.35 <= xx <= 0.65 else 0.125
        v[0][i] = 0
        p[0][i] = 1 if 0.35 <= xx <= 0.65 else 0.1

    rv[0] = r[0] * v[0]
    e[0] = 1.5 * p[0] + 0.5 * r[0] * v[0]**2

    for n in range(time_steps-1):
        dt = 0.95 * np.min(dx / (np.abs(v[n]) + np.sqrt(5/3 * p[n] / r[n])))
        for i in range(2, 1002, 1):

            ###########################################################
            #   Compute fluxes for half step.
            ###########################################################

            dp_i = (p[n][i] - p[n][i-1]) / dx
            dp_o = (p[n][i+1] - p[n][i]) / dx

            dpv_i = (p[n][i] * v[n][i] - p[n][i-1] * v[n][i-1]) / dx
            dpv_o = (p[n][i+1] * v[n][i+1] - p[n][i] * v[n][i]) / dx

            r_f_i = r[n][i-1] * v[n][i-1]
            r_f_c = r[n][i] * v[n][i]
            r_f_o = r[n][i+1] * v[n][i+1]

            rv_f_i = rv[n][i-1] * v[n][i-1]
            rv_f_c = rv[n][i] * v[n][i]
            rv_f_o = rv[n][i+1] * v[n][i+1]

            e_f_i = e[n][i-1] * v[n][i-1]
            e_f_c = e[n][i] * v[n][i]
            e_f_o = e[n][i+1] * v[n][i+1]

            ###########################################################
            #   Compute half step.
            ###########################################################

            r_i_h = 0.5 * (r[n][i] + r[n][i-1])\
                  - 0.5 * dt / dx * (r_f_c - r_f_i)
            r_o_h = 0.5 * (r[n][i+1] + r[n][i])\
                  - 0.5 * dt / dx * (r_f_o - r_f_c)

            rv_i_h = 0.5 * (rv[n][i] + rv[n][i-1])\
                   - 0.5 * dt / dx * (rv_f_c - rv_f_i)\
                   - 0.5 * dt * dp_i
            rv_o_h = 0.5 * (rv[n][i+1] + rv[n][i])\
                   - 0.5 * dt / dx * (rv_f_o - rv_f_c)\
                   - 0.5 * dt * dp_o

            e_i_h = 0.5 * (e[n][i] + e[n][i-1])\
                  - 0.5 * dt / dx * (e_f_c - e_f_i)\
                  - 0.5 * dt * dpv_i
            e_o_h = 0.5 * (e[n][i+1] + e[n][i])\
                  - 0.5 * dt / dx * (e_f_o - e_f_c)\
                  - 0.5 * dt * dpv_o

            v_i_h = rv_i_h / r_i_h
            v_o_h = rv_o_h / r_o_h

            ###########################################################
            #   Compute fluxes for full step
            ###########################################################

            r_f_i_h = r_i_h * v_i_h
            r_f_o_h = r_o_h * v_o_h

            rv_f_i_h = rv_i_h * v_i_h
            rv_f_o_h = rv_o_h * v_o_h

            e_f_i_h = e_i_h * v_i_h
            e_f_o_h = e_o_h * v_o_h

            ###########################################################
            #   Compute full step
            ###########################################################

            r[n+1][i] = r[n][i] + dt / dx * (r_f_i_h - r_f_o_h)
            rv[n+1][i] = rv[n][i] + dt / dx * (rv_f_i_h - rv_f_o_h)
            e[n+1][i] = e[n][i] + dt / dx * (e_f_i_h - e_f_o_h)

            ###########################################################
            #   Prepare for source terms.
            ###########################################################

            p_i_h = (e_i_h - 0.5 * r_i_h * v_i_h**2) * (2/3)
            p_o_h = (e_o_h - 0.5 * r_o_h * v_o_h**2) * (2/3)

            ###########################################################
            #   Compute source terms.
            ###########################################################

            rv_s = dt / dx * (p_o_h - p_i_h)
            e_s = dt / dx * (p_o_h * v_o_h - p_i_h * v_i_h)

        ###############################################################
        #   Equalize ghost cells.
        ###############################################################

        r[n+1][:2] = r[n+1][2]
        r[n+1][-2:] = r[n+1][-3]
        v[n+1][:2] = v[n+1][2]
        v[n+1][-2:] = v[n+1][-3]
        p[n+1][:2] = p[n+1][2]
        p[n+1][-2:] = p[n+1][-3]
        rv[n+1][:2] = rv[n+1][2]
        rv[n+1][-2:] = rv[n+1][-3]
        e[n+1][:2] = e[n+1][2]
        e[n+1][-2:] = e[n+1][-3]

    return(x, r, v, p, rv, e)


def upwind_method(time_steps):
    r = np.zeros((time_steps, 1004))
    v = np.zeros((time_steps, 1004))
    p = np.zeros((time_steps, 1004))
    rv = np.zeros((time_steps, 1004))
    e = np.zeros((time_steps, 1004))

    dx = 1 / 999

    x = np.arange(-2 * dx, 1 + 3 * dx, dx)

    for i, xx in enumerate(x):
        r[0][i] = 1 if 0.35 <= xx <= 0.65 else 0.125
        v[0][i] = 0
        p[0][i] = 1 if 0.35 <= xx <= 0.65 else 0.1

    rv[0] = r[0] * v[0]
    e[0] = 1.5 * p[0] + 0.5 * r[0] * v[0]**2

    for n in range(time_steps-1):
        dt = 0.2 * np.min(dx / (np.abs(v[n]) + np.sqrt(5/3 * p[n] / r[n])))
        for i in range(2, 1002, 1):
            ###########################################################
            #   Compute in and out advection speeds.
            ###########################################################
            v_i_h = 0.5 * (v[n][i-1] + v[n][i])
            v_o_h = 0.5 * (v[n][i] + v[n][i+1])

            ###########################################################
            #   Make flux half step
            ###########################################################
            if v_i_h >= 0:
                r_f_i_h = r[n][i-1] * v_i_h
                rv_f_i_h = rv[n][i-1] * v_i_h
                e_f_i_h = e[n][i-1] * v_i_h
            else:
                r_f_i_h = r[n][i] * v_i_h
                rv_f_i_h = rv[n][i] *  v_i_h
                e_f_i_h = e[n][i] * v_i_h

            if v_o_h >= 0:
                r_f_o_h = r[n][i] * v_o_h
                rv_f_o_h = rv[n][i] * v_o_h
                e_f_o_h = e[n][i] * v_o_h
            else:
                r_f_o_h = r[n][i+1] * v_o_h
                rv_f_o_h = rv[n][i+1] * v_o_h
                e_f_o_h = e[n][i+1] * v_o_h

            ###########################################################
            #   Compute full step.
            ###########################################################

            r[n+1][i] = r[n][i] + dt / dx * (r_f_i_h - r_f_o_h)
            rv[n+1][i] = rv[n][i] + dt / dx * (rv_f_i_h - rv_f_o_h)
            e[n+1][i] = e[n][i] + dt / dx * (e_f_i_h - e_f_o_h)
            ###########################################################
            #   Compute source terms.
            ###########################################################
            rv_s = 0.5 * (dt / dx) * (p[n][i+1] - p[n][i-1])
            e_s = 0.5 * (dt / dx) * ((p[n][i+1] * v[n][i+1])
                                    -(p[n][i-1] * v[n][i-1]))

            ###########################################################
            #   Make source term step and update next time step.
            ###########################################################

            rv[n+1][i] -= rv_s
            e[n+1][i] -= e_s
            v[n+1][i] = rv[n+1][i] / r[n+1][i]
            p[n+1][i] = (e[n+1][i] - 0.5 * r[n+1][i] * v[n+1][i]**2) * (2/3)

        ###############################################################
        #   Equalize ghost cells.
        ###############################################################

        r[n+1][:2] = r[n+1][2]
        r[n+1][-2:] = r[n+1][-3]
        v[n+1][:2] = v[n+1][2]
        v[n+1][-2:] = v[n+1][-3]
        p[n+1][:2] = p[n+1][2]
        p[n+1][-2:] = p[n+1][-3]
        rv[n+1][:2] = rv[n+1][2]
        rv[n+1][-2:] = rv[n+1][-3]
        e[n+1][:2] = e[n+1][2]
        e[n+1][-2:] = e[n+1][-3]

    return(x, r, v, p, rv, e)


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
        time_steps = int(arguments[1])
    else:
        time_steps = 1000

    sysOS = platform.system()
    set_image_save_path(sysOS)

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
        x, r, v, p, rv, e = lax_wendroff(time_steps=time_steps)
        t = r / (p * 1.6735575e-27 * 1.380649e-23)
    elif MODEL == 'u':
        x, r, v, p, rv, e = upwind_method(time_steps=time_steps)
        t = r / (p * 1.6735575e-27 * 1.380649e-23)
    else:
        raise RuntimeError('Unknown simulation model. Exiting...')
        exit(1)

    for n in range(time_steps):
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
        fig = plt.figure(figsize=(7.27126, 7.87402))       #   figsize in inches

        ###################################################################
        #   Plot the data.
        ###################################################################
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)

        ax1.grid(True, which='major', linewidth=0.5)
        ax2.grid(True, which='major', linewidth=0.5)
        ax3.grid(True, which='major', linewidth=0.5)
        ax4.grid(True, which='major', linewidth=0.5)
        ax5.grid(True, which='major', linewidth=0.5)
        ax6.grid(True, which='major', linewidth=0.5)

        ax1.plot(x, r[0], label='initial')
        ax1.plot(x, r[n], label='propagated')

        ax2.plot(x, v[0], label='initial')
        ax2.plot(x, v[n], label='propagated')

        ax3.plot(x, p[0], label='initial')
        ax3.plot(x, p[n], label='propagated')

        ax4.plot(x, rv[0], label='initial')
        ax4.plot(x, rv[n], label='propagated')

        ax5.plot(x, e[0], label='initial')
        ax5.plot(x, e[n], label='propagated')

        ax6.plot(x, t[0], label='initial')
        ax6.plot(x, t[n], label='propagated')

        ###############################################################
        #   Format the subplot.
        ###############################################################
        ax1.set_title('Density')
        ax1.set_xlabel('x')
        ax1.set_ylabel('\(r(x)\)')
        ax1.set_ylim(0, 1.4)
        ax1.legend(loc='upper right')

        ax2.set_title('Velocity')
        ax2.set_xlabel('x')
        ax2.set_ylabel('\(v(x)\)')
        ax2.set_ylim(-1, 1)
        ax2.legend(loc='upper left')

        ax3.set_title('Preassure')
        ax3.set_xlabel('x')
        ax3.set_ylabel('\(P(x)\)')
        ax3.set_ylim(0, 1.4)
        ax3.legend(loc='upper right')

        ax4.set_title('Momentum density')
        ax4.set_xlabel('x')
        ax4.set_ylabel('\(r(x) \cdot v(x)\)')
        ax4.set_ylim(-0.5, 0.5)
        ax4.legend(loc='upper left')

        ax5.set_title('Energy')
        ax5.set_xlabel('x')
        ax5.set_ylabel('\(\epsilon(x)\)')
        ax5.set_ylim(0, 2)
        ax5.legend(loc='upper right')

        ax6.set_title('Temperature')
        ax6.set_xlabel('x')
        ax6.set_ylabel('T(x)')
        ax6.set_ylim(3e49, 1.3e50)
        ax6.legend(loc='upper left')

        fig.suptitle(f'n = {n:04}')

        fig.subplots_adjust(top=0.925,
                            bottom=0.065,
                            left=0.075,
                            right=0.96,
                            hspace=0.325,
                            wspace=0.265)

        ###############################################################
        #   Save the Figure to a file in the current working
        #   directory.
        ###############################################################
        if n%2 == 0:
            plt.savefig(f'./plots/Part_03/frame_{n:04}.png', format='png')

        if n == 99 or n == 999:
            plt.savefig(f'./plots/Part_03/snapshot_{n}.pdf')

        ###############################################################
        #   Show the plot in in a popup window.
        ###############################################################
        #plt.show()
        plt.close(fig)


    if MODEL == 'l':
        animation_name = 'animation_lax_wendroff.gif'
    elif MODEL == 'u':
        animation_name = 'animation_upwind.gif'
    if sysOS == 'Linux':
        cmd = f'convert -delay 5 ./plots/Part_03/frame_*.png -loop 0 '\
            + f'./plots/Part_03/{animation_name} && rm ./plots/Part_03/*.png'
        try:
            _ = subprocess.check_output(cmd, shell=True,
                                        stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            warnings.warn('Unable to create animation with ImageMagick.')
        ###############################################################
        #   Propagation...?
        ###############################################################

if __name__ == '__main__':
    main()
    exit(0)

