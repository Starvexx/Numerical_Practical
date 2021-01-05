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

    To compute the time evolution, the advection equation needs to be
    solved. This is a differential equation and can be solved by using
    the forward euler scheme. Therefor the following will be computed:

                                 ⎛ q_(i+1)^n - g_(i-1)^n ⎞
        q_i^(n+1) = q_i^n - u Δt ⎜———————————————————————⎟
                                 ⎝ x_(i+1)   - x_(i-1)   ⎠

    """
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
        for n in range(self.sim_time_steps-1):
            for i in range(2, len(self.x)-2):
                self.sim_spacetime[n+1][i] = self.sim_spacetime[n][i] \
                                           - velocity \
                                           * delta_t \
                                           / (2 * self.delta_x) \
                                           * ( self.sim_spacetime[n][i+1]
                                             - self.sim_spacetime[n][i-1] )
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]


class UpstreamDifferencingScheme(SimulationDomain):
    """Docstring"""
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
        for n in range(self.sim_time_steps-1):
            for i in range(2, len(self.x)-2):
                self.sim_spacetime[n+1][i] = self.sim_spacetime[n][i] \
                                           - velocity \
                                           * (delta_t / self.delta_x) \
                                           * ( self.sim_spacetime[n][i]
                                             - self.sim_spacetime[n][i-1] )
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]


class LaxWendroffScheme(SimulationDomain):
    """Docstring"""
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
        D = 0.5 * (velocity * delta_t / self.delta_x)**2
        for n in range(self.sim_time_steps-1):
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
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]


