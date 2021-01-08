#! /usr/bin/env python3
"""Auxiliary file containing class definitions.

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

import numpy as np

from matplotlib import rc
from matplotlib import pyplot as plt

#######################################################################
#   Import user files.
#######################################################################

from advection.BaseClasses import SimulationDomain

#######################################################################
#   Define functions used in this file or package.
#######################################################################

class DonorCellMethod(SimulationDomain):
    """Donor Cell sub class"""
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None,
                 bins=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range,
                         bins)


    def propagate(self, delta_t, velocity):
        """Propagating the Package"""
        if velocity < 0:
            self._invert_spacetime()

        for n in range(self.sim_time_steps-1):
            for i in range(2, len(self.x)-2):
                _flux_in = self._flux(np.abs(velocity),
                                     self.sim_spacetime[n][i-1],
                                     0,
                                     delta_t)

                _flux_out = self._flux(np.abs(velocity),
                                      self.sim_spacetime[n][i],
                                      0,
                                      delta_t)

                self.sim_spacetime[n+1][i] = self.sim_spacetime[n][i] \
                                           + delta_t / self.delta_x \
                                           * (_flux_in - _flux_out)

            ###########################################################
            #   Equalize the ghost cells
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]

        if velocity < 0:
            self._invert_spacetime()


class FrommsMethod(SimulationDomain):
    """Fromm's Method sub class."""
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None,
                 bins=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range,
                         bins)


    def propagate(self, delta_t, velocity):
        """Propagating the Package"""
        print("Using Fromm's Method")
        if velocity < 0:
            self._invert_spacetime()

        for n in range(self.sim_time_steps-1):
            for i in range(2, len(self.x)-2):
                #######################################################
                #   Compute sigma
                #######################################################
                _sigma_in = ( self.sim_spacetime[n][i]
                            - self.sim_spacetime[n][i-2]) / (2 * self.delta_x)

                _sigma_out = ( self.sim_spacetime[n][i+1]
                             - self.sim_spacetime[n][i-1]) / (2 * self.delta_x)

                #######################################################
                #   Compute the fluxes
                #######################################################
                _flux_in = self._flux(np.abs(velocity),
                                     self.sim_spacetime[n][i-1],
                                     _sigma_in,
                                     delta_t)

                _flux_out = self._flux(np.abs(velocity),
                                      self.sim_spacetime[n][i],
                                      _sigma_out,
                                      delta_t)

                self.sim_spacetime[n+1][i] = ( self.sim_spacetime[n][i]
                                             + delta_t / self.delta_x
                                             * (_flux_in - _flux_out))

            ###########################################################
            #   Equalize the ghost cells
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]

        if velocity < 0:
            self._invert_spacetime()


class BeamWarmingMethod(SimulationDomain):
    """Beam Warming sub class"""
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None,
                 bins=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range,
                         bins=None)


    def propagate(self, delta_t, velocity):
        """Propagating the Package"""
        if velocity < 0:
            self._invert_spacetime()

        for n in range(self.sim_time_steps-1):
            for i in range(2, len(self.x)-2):
                #######################################################
                #   Compute sigma
                #######################################################
                _sigma_in = ( self.sim_spacetime[n][i-1]
                            - self.sim_spacetime[n][i-2]) / self.delta_x

                _sigma_out = ( self.sim_spacetime[n][i]
                             - self.sim_spacetime[n][i-1]) / self.delta_x

                #######################################################
                #   Compute the fluxes
                #######################################################
                flux_in = self._flux(np.abs(velocity),
                                     self.sim_spacetime[n][i-1],
                                     _sigma_in,
                                     delta_t)

                flux_out = self._flux(np.abs(velocity),
                                      self.sim_spacetime[n][i],
                                      _sigma_out,
                                      delta_t)

                self.sim_spacetime[n+1][i] = ( self.sim_spacetime[n][i]
                                             + delta_t / self.delta_x
                                             * (flux_in - flux_out))

            ###########################################################
            #   Equalize the ghost cells
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]

        if velocity < 0:
            self._invert_spacetime()


class LaxWendroffMethod(SimulationDomain):
    """Lax-Wendroff sub class"""
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None,
                 bins=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range,
                         bins)


    def propagate(self, delta_t, velocity):
        """Propagating the Package"""
        if velocity < 0:
            self._invert_spacetime()

        for n in range(self.sim_time_steps-1):
            for i in range(2, len(self.x)-2):
                #######################################################
                #   Compute sigma
                #######################################################
                _sigma_in = ( self.sim_spacetime[n][i]
                            - self.sim_spacetime[n][i-1]) / self.delta_x

                _sigma_out = ( self.sim_spacetime[n][i+1]
                             - self.sim_spacetime[n][i]) / self.delta_x

                #######################################################
                #   Compute the fluxes
                #######################################################
                flux_in = self._flux(np.abs(velocity),
                                     self.sim_spacetime[n][i-1],
                                     _sigma_in,
                                     delta_t)

                flux_out = self._flux(np.abs(velocity),
                                      self.sim_spacetime[n][i],
                                      _sigma_out,
                                      delta_t)

                self.sim_spacetime[n+1][i] = ( self.sim_spacetime[n][i]
                                             + delta_t / self.delta_x
                                             * (flux_in - flux_out))

            ###########################################################
            #   Equalize the ghost cells
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]

        if velocity < 0:
            self._invert_spacetime()


class TwoStepLaxWendroff(SimulationDomain):
    """Two Step Lax Wendroff Method for"""
    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None,
                 bins=None):
        super().__init__(delta_x,
                         sim_time_steps,
                         function,
                         domain_range,
                         bins)


    _half_step_fwd = np.zeros(np.shape(sim_spacetime[0]), dtype=np.float64)
    _half_step_bwd = _half_step_fwd

    def propagate(self):
        """Propagating the Package"""
        for n in range(self.sim_time_steps-1):
            for i in range(2, len(self.x)-2):
                #######################################################
                #   Compute sigma
                #######################################################
                _sigma_in = ( self.sim_spacetime[n][i]
                            - self.sim_spacetime[n][i-1]) / self.delta_x

                _sigma_out = ( self.sim_spacetime[n][i+1]
                             - self.sim_spacetime[n][i]) / self.delta_x

                #######################################################
                #   Compute the fluxes
                #######################################################
                flux_in = self._flux(np.abs(velocity),
                                     self.sim_spacetime[n][i-1],
                                     _sigma_in,
                                     delta_t)

                flux_out = self._flux(np.abs(velocity),
                                      self.sim_spacetime[n][i],
                                      _sigma_out,
                                      delta_t)

                self.sim_spacetime[n+1][i] = ( self.sim_spacetime[n][i]
                                             + delta_t / self.delta_x
                                             * (flux_in - flux_out))

            ###########################################################
            #   Equalize the ghost cells
            ###########################################################
            self.sim_spacetime[n+1][:2] = self.sim_spacetime[n+1][2]
            self.sim_spacetime[n+1][-2:] = self.sim_spacetime[n+1][-3]

