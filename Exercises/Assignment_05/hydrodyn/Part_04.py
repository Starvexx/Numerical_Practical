#! /usr/bin/env python3
"""Project Name

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        17.01.2021
    Last edited:    17.01.2021

Description...
"""

#######################################################################
#   Import packages.
#######################################################################

import warnings

import numpy as np

import os

import platform

import subprocess

import sys

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
        run. ['Linux'|'Windows']
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
            image_save_path = os.getcwd() + '\plots\Part_04'
            os.mkdir(image_save_path)
        except FileExistsError:
            print('Plot directory already exists.')
    else:
        ###############################################################
        #   Try to create the folder. If the folder already exists,
        #   notify the user and do nothing.
        ###############################################################
        try:
            image_save_path = os.getcwd() + '/plots/Part_04'
            os.mkdir(image_save_path)
        except FileExistsError:
            print('Plot directory already exists.')

    return image_save_path


def parse_args():
    """A simple argument parser to read commandline arguments."""
    possible_modes = ['b', 'f']
    arg_names = sys.argv[1::2]

    man_path = './man/hydro_manual.txt'

    if len(sys.argv) > 5:
        warnings.warn('Wrong number of arguments given.')
        with open(man_path) as helpfile:
            print(helpfile.read())
        exit(1)
    elif len(sys.argv) == 1:
        warnings.warn('No arguments given, defaulting to Lax-Wendroff.')
        return 'f', None

    if '-m' not in arg_names:
        print('ERROR: Too many missing arguments.')
        with open(man_path) as helpfile:
            print(helpfile.read())
        exit(1)
    elif ('-m' in arg_names) and ('-c' not in arg_names):
        idx_mode = sys.argv.index('-m')
        warnings.warn('Missing "-c" argument, using default')
        try:
            if sys.argv[idx_mode+1] in possible_modes:
                return sys.argv[idx_mode+1], None
            else:
                warnings.warn('Wrong method chosen, using Lax-Wendroff.')
                return 'f', None
        except IndexError:
            warnings.warn('No method chosen, using Lax-Wendroff.')
            return 'f', None
    elif ('-m' in arg_names) and ('-c' in arg_names):
        idx_mode = sys.argv.index('-m')
        idx_c = sys.argv.index('-c')
        try:
            if sys.argv[idx_mode+1] in possible_modes:
                return sys.argv[idx_mode+1], sys.argv[idx_c+1]
            else:
                warnings.warn('Wrong method chosen, using Lax Wendroff.')
                return 'f', sys.argv[idx_c+1]
        except IndexError:
            warnings.warn('At least one argument missing, using Lax Wendroff'+\
                          ' with default time step')
            return 'f', None
    else:
        print('ERROR: Something went wrong.')
        with open(man_path) as helpfile:
            print(helpfile.read())
        exit(1)


def init(x):
    if 3.5 <= x <= 6.5:
        return 1
    else:
        return 0.1


#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""

    sysOS = platform.system()
    save_path = set_image_save_path(sysOS)

    arguments = parse_args()

    MODE = arguments[0]
    if arguments[1] is not None:
        c = float(arguments[1])
    else:
        c = 1

    bins = 100

    dx = 10 / (bins - 1)
    dt = c * 0.5 * dx**2

    x = np.arange(-2*dx, 10+3*dx, dx)

    t_start = 0
    t_end = 1

    N = 1 + int((t_end - t_start) / dt)
    dt_final = ((t_end - t_start) / dt) - (N - 1)

    q = np.zeros((N, bins+4))
    q[0] = np.array([init(xx) for xx in x])

    if MODE == 'f':
        for n in range(N-1):
            for i in range(2, len(x)-2, 1):
                if n < (N-2):
                    q[n+1][i] = q[n][i] + (dt / (dx**2))\
                              * (q[n][i+1] - 2*q[n][i] + q[n][i-1])
                elif n == (N-2):
                    q[n+1][i] = q[n][i] + (dt_final / (dx**2))\
                              * (q[n][i+1] - 2*q[n][i] + q[n][i-1])
            q[n+1][:2] = q[n+1][2]
            q[n+1][-2:] = q[n+1][-3]
    elif MODE == 'b':
        a = np.ones(bins+3) * (-dt / (dx**2))
        b = np.ones(bins+4) * (1 + (2*dt) / (dx**2))

        A = np.zeros((bins+4, bins+4))
        np.fill_diagonal(A[1:], a)
        np.fill_diagonal(A[:, 1:], a)
        np.fill_diagonal(A, b)

        for n in range(N-1):
            if n == N-2:
                a = np.ones(bins+3) * (-dt_final / (dx**2))
                b = np.ones(bins+4) * (1 + (2*dt_final) / (dx**2))

                A = np.zeros((bins+4, bins+4))
                np.fill_diagonal(A[1:], a)
                np.fill_diagonal(A[:, 1:], a)
                np.fill_diagonal(A, b)

            q[n+1] = np.dot(np.linalg.inv(A), q[n])
            q[n+1][:2] = 0.1
            q[n+1][-3:] = 0.1

    else:
        raise RuntimeError

    t = 0
    for n, q_n in enumerate(q):
        if n != N-1:
            t += dt
        elif n == N-1:
            t += dt_final

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
        fig = plt.figure(figsize=(3.55659, 2.66744))       #   figsize in inches

        ###################################################################
        #   Plot the data.
        ###################################################################
        ax1 = fig.add_subplot(111)

        ax1.grid(True, which='major', linewidth=0.5)

        ax1.plot(x, q[0], label='initial')
        ax1.plot(x, q_n, label='propagated')

        if MODE == 'f':
            ax1.set_title(f'FTCS @ t = {t:.3f}, c = {c}')
        elif MODE == 'b':
            ax1.set_title(f'BTCS @ t = {t:.3f}, c = {c}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('q(x)')
        ax1.set_ylim(0, 1.1)
        ax1.set_xlim(-0.3, 10.3)

        fig.subplots_adjust(top=0.88,
                            bottom=0.16,
                            left=0.14,
                            right=0.975,
                            hspace=0.2,
                            wspace=0.2)

        if n == int(N/2):
            plt.savefig(f'{save_path}/{MODE}tcs_snapshot_n{n:04}_c{c}.pdf')

        plt.savefig(f'{save_path}/frame_{n:04}.png', format='png')
        plt.close(fig)

    animation_name = f'{MODE.upper()}TCS_c{c}.gif'

    if sysOS == 'Linux':
        cmd = f'convert -delay 5 {save_path}/frame_*.png -loop 0 '\
            + f'{save_path}/{animation_name} && rm {save_path}/*.png'
        try:
            _ = subprocess.check_output(cmd, shell=True,
                                        stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            warnings.warn('Unable to create animation with ImageMagick.')

    print(f'Done! Check {save_path} for plots and/or animations.')

if __name__ == '__main__':
    main()
    exit(0)

