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


    def propagate(self):
        """Propagating the Package"""
        for n in range(self.sim_time_steps-1):
            r = self.sim_spacetime[0][n]
            v = self.sim_spacetime[1][n]
            p = self.sim_spacetime[2][n]
            rv = self.sim_spacetime[3][n]
            e = self.sim_spacetime[4][n]

            #print(r)

            r_next = np.zeros(np.shape(r))
            v_next = np.zeros(np.shape(r))
            p_next = np.zeros(np.shape(r))
            rv_next = np.zeros(np.shape(r))
            e_next = np.zeros(np.shape(r))

            print(f'\n\nn: {n}\n')
            print('r:  ', r)
            print('v:  ', v)
            print('p:  ', p)
            print('rv: ', rv)
            print('e:  ', e)
            for i in range(2, len(self.x)-2):
                #print(np.min(self.delta_x / (np.abs(v) + np.sqrt(self.gamma * p/r))))
                #exit()
                #print('computing delta_t')
                delta_t = 0.95 * np.min(self.delta_x
                                       /(np.abs(v)
                                            +np.sqrt(self.gamma * p / r)))
                #print('delta_t: ', delta_t)
                r_flux_in = r[i-1] * v[i-1]
                r_flux_cen = r[i] * v[i]
                r_flux_out = r[i+1] * v[i+1]

                rv_flux_in = rv[i-1] * v[i-1]
                rv_flux_cen = rv[i] * v[i]
                rv_flux_out = rv[i+1] * v[i+1]

                e_flux_in = e[i-1] * v[i-1]
                e_flux_cen = e[i] * v[i]
                e_flux_out = e[i+1] * v[i+1]

                #######################################################
                #   Compute half step.
                #######################################################

                r_in_h = 0.5 * (r[i] + r[i-1]) - delta_t / (2*self.delta_x) \
                       * (r_flux_cen - r_flux_in)
                r_out_h = 0.5 * (r[i+1] + r[i]) - delta_t / (2*self.delta_x) \
                        * (r_flux_out - r_flux_cen)

                rv_in_h = 0.5 * (rv[i] + rv[i-1])\
                        - delta_t / (2*self.delta_x)\
                        * (rv_flux_cen - rv_flux_in)\
                        - delta_t / (2*self.delta_x)\
                        * (p[i] - p[i-1])
                rv_out_h = 0.5 * (rv[i+1] + rv[i])\
                         - delta_t / (2*self.delta_x)\
                         * (rv_flux_out - rv_flux_cen)\
                         - delta_t / (2*self.delta_x)\
                         * (p[i+1] - p[i])

                e_in_h = 0.5 * (e[i] + e[i-1])\
                       - delta_t / (2*self.delta_x)\
                       * (e_flux_cen - e_flux_in)\
                       - delta_t / (2*self.delta_x)\
                       * (p[i] * v[i] - p[i-1] * v[i-1])
                e_out_h = 0.5 * (e[i] + e[i-1])\
                        - delta_t / (2*self.delta_x)\
                        * (e_flux_out - e_flux_cen)\
                        - delta_t / (2*self.delta_x)\
                        * (p[i+1] * v[i+1] - p[i] * v[i])

                #######################################################
                #   Compute half step fluxes.
                #######################################################

                v_in_h = rv_in_h / r_in_h
                v_out_h = rv_out_h / r_out_h

                r_flux_in_h = r_in_h * v_in_h
                r_flux_out_h = r_out_h * v_out_h

                rv_flux_in_h = rv_in_h * v_in_h
                rv_flux_out_h = rv_out_h * r_out_h

                e_flux_in_h = e_in_h * v_in_h
                e_flux_out_h = e_out_h * v_out_h

                #######################################################
                #   Compute full time step
                #######################################################
                r_next[i] = r[i] + delta_t / self.delta_x * (r_flux_in_h
                                                            -r_flux_out_h)
                rv_next[i] = rv[i] + delta_t / self.delta_x * (rv_flux_in_h
                                                              -rv_flux_out_h)
                e_next[i] = e[i] + delta_t / self.delta_x * (e_flux_in_h
                                                            -e_flux_out_h)

                #######################################################
                #   Compute source terms
                #######################################################
                p_in_h = (e_in_h - 0.5 * r_in_h * v_in_h**2) * (2/3)
                p_out_h = (e_out_h - 0.5 * r_out_h * v_out_h**2) * (2/3)

                #rv_source = 0.5 * delta_t / self.delta_x * (p[i+1] - p[i-1])
                #e_source = 0.5 * delta_t / self.delta_x * (p[i+1] * v[i+1]
                #                                          -p[i-1] * v[i-1])

                rv_source = delta_t / self.delta_x * (p_out_h - p_in_h)
                e_source = delta_t / self.delta_x\
                         * (p_out_h * v_out_h - p_in_h * v_in_h)

                rv_next[i] -= rv_source
                e_next[i] -= e_source
                #print(r_next[i])

            ###########################################################
            #   Equalize ghost cells
            ###########################################################
            r_next[:2] = r_next[2]
            r_next[-2:] = r_next[-3]

            rv_next[:2] = rv_next[2]
            rv_next[-2:] = rv_next[-3]

            e_next[:2] = e_next[2]
            e_next[-2:] = e_next[-3]

            ###########################################################
            #   Update API
            ###########################################################
            v_next = rv_next / r_next
            p_next = (e_next - 0.5 * (r_next * v_next**2)) * (2/3)

            #print(rv_next)
            #print(v_next)
            print(p_next)
            #print(r_next)

            self.sim_spacetime[0][n+1] = r_next
            self.sim_spacetime[1][n+1] = v_next
            self.sim_spacetime[2][n+1] = p_next
            self.sim_spacetime[3][n+1] = rv_next
            self.sim_spacetime[4][n+1] = e_next
            #for i in range(len(rv_next)):
            #    print(self.sim_spacetime[1][n+1][i])

            if n == 1:
                pass
                #exit()


            #print(self.sim_spacetime)
           #     for ii in range(5):
           #         ###################################################
           #         #   Make forward half step.
           #         ###################################################
           #         half_step_fwd[ii] = 0.5 * (self.sim_spacetime[ii][n][i+1]
           #                                  +self.sim_spacetime[ii][n][i])\
           #                          - delta_t / (2 * self.delta_x) \
           #                          * (self.sim_spacetime[ii][n][i+1]
           #                            *self.sim_spacetime[1][n][i+1]
           #                            -self.sim_spacetime[ii][n][i]
           #                            *self.sim_spacetime[1][n][i])

           #         #######################################################
           #         #   Make backward half step.
           #         #######################################################
           #         half_step_bwd[ii] = 0.5 * (self.sim_spacetime[ii][n][i]
           #                                   +self.sim_spacetime[ii][n][i-1])\
           #                           - delta_t / (2 * self.delta_x) \
           #                           * (self.sim_spacetime[ii][n][i]
           #                             *self.sim_spacetime[1][n][i]
           #                             -self.sim_spacetime[ii][n][i-1]
           #                             *self.sim_spacetime[1][n][i-1])

           #     for ii in range(5):
           #         flux_in[0] = half_step_bwd[0] * half_step_bwd[1]
           #         flux_out[0] = half_step_fwd[0] * half_step_fwd[1]

           #     for ii in range(5):
           #         self.sim_spacetime[ii][n+1][i] =\
           #                 self.sim_spacetime[ii][n][i]\
           #                 + delta_t / self.delta_x\
           #                 * (flux_in[ii] - flux_out[ii])

           #     self.sim_spacetime[3][n+1][i] = self.sim_spacetime[3][n][i]\
           #                                   - delta_t\
           #                                   * (self.sim_spacetime[2][n][i+1]
           #                                     -self.sim_spacetime[2][n][i-1])\
           #                                   / 2 * self.delta_x
           #     self.sim_spacetime[4][n+1][i] = self.sim_spacetime[4][n][i]\
           #                                   - delta_t\
           #                                   * (self.sim_spacetime[2][n][i+1]
           #                                     *self.sim_spacetime[1][n][i+1]
           #                                     -self.sim_spacetime[2][n][i-1]
           #                                     *self.sim_spacetime[1][n][i-1])\
           #                                   / 2 * self.delta_x


           # if n == 50:
           #     print(half_step_fwd)
           #     print(half_step_bwd)
           # for ii in range(5):
           #     self.sim_spacetime[ii][n+1][:2] = self.sim_spacetime[ii][n+1][2]
           #     self.sim_spacetime[ii][n+1][-2:] =\
           #                                 self.sim_spacetime[ii][n+1][-3]

