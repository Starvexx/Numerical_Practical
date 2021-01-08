#! /usr/bin/env python3
"""Auxiliary file with sub class definitions.

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        04.01.2021
    Last edited:    05.01.2021

Three sub classes are defined in this file. The three sub classes
correspond to the following:

    Central Difference Scheme
    Upstream Difference Scheme
    Lax Wendroff Scheme

They are implemented using strictly spatial methods and can only handle
a positive and constant propagation velocity in time and space in one
dimension. 
"""

#######################################################################
#   Import packages.
#######################################################################

import warnings

import numpy as np

from matplotlib import rc
from matplotlib import pyplot as plt


###################################################################
#   Import auxiliary user files.
###################################################################

from advection.BaseClasses import SimulationDomain

#######################################################################
#   Class definitions
#######################################################################


class CentralDifferenceScheme(SimulationDomain):
    """The Central Difference Scheme subclass.

    Parent:     SimulationDomain

    To compute the time evolution, the advection equation needs to be
    solved. This is a differential equation and can be solved by using
    the forward euler scheme. Therefor the following will be computed:

                                 ⎛ q_(i+1)^n - g_(i-1)^n ⎞
        q_i^(n+1) = q_i^n - u Δt ⎜———————————————————————⎟        (1)
                                 ⎝ x_(i+1)   - x_(i-1)   ⎠


    Methods:
    --------
    propagate(delta_t, velocity):
        Propagates the Package in time and space.
    """
    ###################################################################
    #   Inherit everything from the Base Class in the constructor.
    ###################################################################
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range)

    ###################################################################
    #   Method to propagate the package in time and space.
    ###################################################################
    def propagate(self, delta_t, velocity):
        """Propagation Method.

        Here the Central Difference Scheme as per Equation (1) is used.

        Parameters:
        -----------
        delta_t : scalar
            The length of the time step for each iteration.
        velocity : scalar
            The bulk motion of the surrounding medium and the package.
        """
        ###############################################################
        #   Iterate through time.
        ###############################################################
        for n in range(self.sim_time_steps-1):
            ###########################################################
            #   iterate through space.
            ###########################################################
            for i in range(2, len(self.x)-2):
                self.sim_spacetime[n+1][i] = self.sim_spacetime[n][i] \
                                           - velocity \
                                           * delta_t \
                                           / (2 * self.delta_x) \
                                           * ( self.sim_spacetime[n][i+1]
                                             - self.sim_spacetime[n][i-1] )
            ###########################################################
            #   Equalize ghost cells.
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]


class UpstreamDifferencingScheme(SimulationDomain):
    """The Upstream Difference Scheme sub class.

    Parent:     SimulationDomain

    A Problem with the Central Difference Scheme is, that is takes
    both values spatially in front and behind the current cell to
    estimate the next time step. It would be much better if only the
    value behind (Upstream; The direction the package originated) were
    used. Therefor the Upstream Difference Scheme was developed. It
    is defined as is shown below:

                               Δt
        q_i^(n+1) = q_i^n - u ———— (q_i^n - q_(i-1)^n)          (2)
                               Δx

    Methods:
    --------
    propagate(delta_t, velocity):
        Propagate the package in time and space.

    """
    ###################################################################
    #   Inherit everything from the Base Class in the constructor.
    ###################################################################
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range)


    ###################################################################
    #   Method to propagate the package in time and space.
    ###################################################################
    def propagate(self, delta_t, velocity):
        """Propagation Method.

        Here the Upstream Difference Scheme as per Equation (2) is
        used.

        Parameters:
        -----------
        delta_t : scalar
            The length of the time step for each iteration.
        velocity : scalar
            The bulk motion of the surrounding medium and the package.
        """
        ###############################################################
        #   Iterate through time.
        ###############################################################
        for n in range(self.sim_time_steps-1):
            ###########################################################
            #   iterate through space.
            ###########################################################
            for i in range(2, len(self.x)-2):
                self.sim_spacetime[n+1][i] = self.sim_spacetime[n][i] \
                                           - velocity \
                                           * (delta_t / self.delta_x) \
                                           * ( self.sim_spacetime[n][i]
                                             - self.sim_spacetime[n][i-1] )
            ###########################################################
            #   Equalize ghost cells.
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]


class LaxWendroffScheme(SimulationDomain):
    """The Upstream Difference Scheme sub class.

    Parent:     SimulationDomain

    The Lax-Wendroff Scheme is an improvement to the Center Deifference
    Scheme of sorts. When using Center Differencing, the simulation
    will soon become very inacurate due to massive oscillations at the
    edges of the package. In order to compensate, in the Lax-Wendroff
    Scheme a diffusion Term is added. The equation to solve becomes
    the following:

                                Δt
        q_i^(n+1) = q_i^n - u —————— (q_(i+1)^n - q_(i-1)^n)
                               2*Δx
                                                                (3)

                          + D (q_(i+1)^n - 2*q_i^n + q_(i-1)^n)

    Where D is the second order Diffusion constant:

            1   ⎛ u*Δt ⎞²
        D = — * ⎜——————⎟
            2   ⎝  Δx  ⎠

    Methods:
    --------
    propagate(delta_t, velocity):
        Propagate the package in time and space.

    """
    ###################################################################
    #   Inherit everything from the Base Class in the constructor.
    ###################################################################
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range)


    def propagate(self, delta_t, velocity):
        """Propagation Method.

        Here the Lax-Wendroff Scheme as per Equation (3) is used.

        Parameters:
        -----------
        delta_t : scalar
            The length of the time step for each iteration.
        velocity : scalar
            The bulk motion of the surrounding medium and the package.
        """
        D = 0.5 * (velocity * delta_t / self.delta_x)**2

        ###############################################################
        #   Iterate through time.
        ###############################################################
        for n in range(self.sim_time_steps-1):
            ###########################################################
            #   Iterate through space.
            ###########################################################
            for i in range(2, len(self.x)-2):
                self.sim_spacetime[n+1][i] = self.sim_spacetime[n][i] \
                                           - velocity \
                                           * delta_t \
                                           / (2 * self.delta_x) \
                                           * ( self.sim_spacetime[n][i+1]
                                             - self.sim_spacetime[n][i-1] ) \
                                           + D * ( self.sim_spacetime[n][i+1]
                                                 - 2 * self.sim_spacetime[n][i]
                                                 + self.sim_spacetime[n][i-1])
            ###########################################################
            #   Equalize ghost cells.
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]


