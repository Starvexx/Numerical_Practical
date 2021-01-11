#! /usr/bin/env python3
"""Auxiliary file holding the advection Base Classes.

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        05.01.2021
    Last edited:    05.01.2021

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
    """This is the Base Class for all advection class children.

    The Base Class consists only of a few very basic properties and one
    method that all children share.

    Properties:
    -----------
    sim_time_steps : scalar
        The number of simulation time steps done in the simulation.
    delta_x : scalar
        The spatial resolution of the simulation.
    domain_range : tuple
        The spatial start and end point of the simulation.
    x : 1D numpy.array
        This array holds all the values of x that lie in the center of
        a bin. For instance if we simulate in the range of [0, 10] with
        delta_x = 1, x would look like so:
            x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        However, be aware that this array now has 11 elements. This may
        not be what you want, so make sure to correctly compute delta_x
        beforehand. For ten bins in the range of [0, 10] we would get a
        delta_x of:
            delta_x = 10/9 = 1.111…
    sim_spacetime : 2D numpy.array
        This is the simulation spacetime. Each timeslot t holds an
        array containing the function values with respect to x and t.
        At the beginning only the zero-th timeslot contains data, as
        the following slots will be filled by the propagation methods
        of the sub classes.

    Methods:
    --------
    _function(xx)
        A default test function to test the algorithm. In this case it
        is a rect function which is 1 if 4 <= x <= 6 and 0.1 elsewhere.
    """
    sim_time_steps = 10
    delta_x = 10/99
    domain_range = (0, 10)
    gamma = 5/3


    def _function(self, xx):
        """Default function if none is defined by the user."""
        self.buff = np.zeros(np.shape(xx)) + 0.1
        self.buff[2 <= xx] = 1
        self.buff[4 <= xx] = 0.1
        return self.buff


    def __init__(self,
                 delta_x=None,
                 sim_time_steps=None,
                 function=None,
                 domain_range=None,
                 bins=None):
        """Object constructor.

        First the simulation domain is set up according to the
        specifications of the user.

        Parameters:
        -----------
        delta_x : scalar
            The spatial resolution of the simulation domain.
        sim_time_steps : scalar
            The number of simulation time steps that shall be
            performed.
        function : function
            A function that populates the initial time step.
        domain_range : tuple, list or numpy.array
            The lower and upper bounds of the spatial domain.
        """
        ###############################################################
        #   Check if any of the arguments are given by the user and
        #   override the default values.
        ###############################################################
        if delta_x is not None:
            self.delta_x = delta_x
        if sim_time_steps is not None:
            self.sim_time_steps = sim_time_steps
        if function is not None:
            self._function = function
        if isinstance(domain_range,
                      (tuple, list,
                       np.ndarray)) and len(domain_range) == 2:
            self.domain_range = domain_range
        if bins is not None:
            self.delta_x = (self.domain_range[1] - self.domain_range[0])\
                         / (bins - 1)

        ###############################################################
        #   initialize the x array holding the bin centers.
        ###############################################################
        self.x = np.arange(self.domain_range[0]-2*self.delta_x,
                           self.domain_range[1]+2*self.delta_x,
                           self.delta_x)

        ###############################################################
        #   initialize the simulation space time.
        ###############################################################
        self.sim_spacetime = np.zeros((len(self.x), self.sim_time_steps),
                                      dtype=np.float64).T

        ###############################################################
        #   Set the initial condition for the simulation.
        ###############################################################
        self.sim_spacetime[0] = self._function(self.x)
        print('Initialized Object.')


    def _flux(self, velocity, quantity, sigma, delta_t):
        """Compute the flux at the border."""
        return velocity * (quantity
                           + 0.5 * sigma * (self.delta_x - velocity * delta_t))


    def _invert_spacetime(self):
        """Invert the spatial component of the simulation space time"""
        self.sim_spacetime = self.sim_spacetime[:, ::-1]


    def add_quantity(self, function=None):
        """Add a new quantity to the simulation."""
        ###############################################################
        #   Distinguish between actual function and array with function
        #   evaluated values.
        ###############################################################
        if type(function).__name__ == 'ndarray'\
                and np.shape(function) == np.shape(self.x):

            self._new_quantity = np.zeros(np.shape(self.sim_spacetime[0]),
                                          dtype=np.float64)

            self._new_quantity[0] = function
        elif type(function).__name__ == 'function':
            self._function = function

            self._new_quantity = np.zeros((len(self.x), self.sim_time_steps),
                                          dtype=np.float64).T
            self._new_quantity[0] = self._function(self.x)
        else:
            raise RuntimeError

        ###############################################################
        #   Append the new quantity to the existing one(s).
        ###############################################################
        if len(np.shape(self.sim_spacetime)) == 2:
            self.sim_spacetime = np.stack((self.sim_spacetime,
                                           self._new_quantity))
        elif len(np.shape(self.sim_spacetime)) == 3:
            print(np.shape(self.sim_spacetime))
            self._new_quantity = self._new_quantity[np.newaxis, :]
            self.sim_spacetime = np.concatenate((self.sim_spacetime,
                                                 self._new_quantity))
        else:
            raise RuntimeError


