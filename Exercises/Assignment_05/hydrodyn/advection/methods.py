#! /usr/bin/env python3
"""Auxiliary file with class definitions.

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
#   Class definitions
#######################################################################

class SimulationDomain:
    """Docstring"""
    sim_time_steps = 10
    delta_x = 10/99
    domain_range = (0, 10)


    def function(self, xx):
        """Default function if none is defined by the user."""
        self.buff = np.zeros(np.shape(xx)) + 0.1
        self.buff[4 <= xx] = 1
        self.buff[6 <= xx] = 0.1
        return self.buff


    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None):
        """Object constructor.

        First the simulation domain is set up according to the
        specifications of the user.
        """
        if delta_x is not None:
            self.delta_x = delta_x
        if sim_time_steps is not None:
            self.sim_time_steps = sim_time_steps
        if function is not None:
            self.function = function
        if isinstance(domain_range,
                      (tuple, list,
                       np.ndarray)) and len(domain_range) == 2:
            self.domain_range = domain_range
        self.x = np.arange(self.domain_range[0]-2*self.delta_x,
                           self.domain_range[1]+2*self.delta_x,
                           self.delta_x)
        self.sim_spacetime = np.zeros((len(self.x), self.sim_time_steps),
                                      dtype=np.float64).T

        self.sim_spacetime[0] = self.function(self.x)


class CentralDifferenceScheme(SimulationDomain):
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
                self.sim_spacetime[n+1][i] =   self.sim_spacetime[n][i] \
                                             - velocity \
                                             * delta_t \
                                             / (2 * self.delta_x) \
                                             * (  self.sim_spacetime[n][i+1]
                                                - self.sim_spacetime[n][i-1])
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]

