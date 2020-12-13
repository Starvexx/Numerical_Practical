#! /usr/bin/env python3
"""Project Name

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        
    Last edited:    

Description...
"""

#######################################################################
#   Import packages.
#######################################################################
import sys
import os

import scipy
from scipy import stats

import warnings

import numpy as np

from matplotlib import rc
from matplotlib import pyplot as plt

#######################################################################
#   Define functions used in this file or package.
#######################################################################

def read(filename):
    """Template function"""
    return np.genfromtxt(filename)


def likelihood(p, mu, sigma):
    """Compute likelihood."""
    return np.log(np.sum(1/np.sqrt(2 * np.pi * sigma**2) * np.exp(- (p - mu)**2
                                                    / (2 * sigma**2))))


def metropolis_hastings(p, mu, sigma):
    mu_noise = stats.norm(mu, 0.1).rvs()
    sigma_noise = stats.norm(sigma, 0.1).rvs()
    #print(mu_noise, sigma_noise)

    L_noise = likelihood(p, mu_noise, sigma_noise)
    L = likelihood(p, mu, sigma)
    print(L, L_noise)

#######################################################################
#   Main function.
#######################################################################

def main():
    """Main subroutine"""
    if len(sys.argv) != 2:
        filepath = os.getcwd() + "/Dataset25.txt"

    data = read(filepath)
    mean = np.mean(data)
    std = np.std(data)
    # starting values for mean and std: mean_0, std_0
    factors = np.random.uniform(-1, 1, 2)
    mean_0 = mean + mean * factors[0]
    std_0 = std + std * factors[1]

    metropolis_hastings(data, mean_0, std_0)

    #print(mean, mean_0, std, std_0)


if __name__ == '__main__':
    main()
    exit(0)

