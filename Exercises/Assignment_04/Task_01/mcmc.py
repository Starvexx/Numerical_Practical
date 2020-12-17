#! /usr/bin/env python3
"""Project Name

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        13.12.2020
    Last edited:    16.12.2020

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
    return np.sum(np.log(1/np.sqrt(2 * np.pi * sigma**2)\
                           * np.exp(- (p - mu)**2 / (2 * sigma**2))))


def metropolis_hastings(p, mu, sigma):
    mu_noise = stats.norm(mu, 0.1).rvs()
    sigma_noise = stats.norm(sigma, 0.1).rvs()
    #print(mu_noise, sigma_noise)

    L_noise = likelihood(p, mu_noise, sigma_noise)
    L = likelihood(p, mu, sigma)

    if L < L_noise:
        return mu_noise, sigma_noise
    else:
        return mu, sigma


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

    iters = 3000
    means = np.zeros(iters)
    stds = np.zeros(iters)

    for i in range(iters):
        means[i] = mean_0
        stds[i] = std_0
        mean_0, std_0 = metropolis_hastings(data, mean_0, std_0)

    #print(means, stds)
    #print(mean, std)

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
    fig = plt.figure(figsize=(6.3, 3.54))       #   figsize in inches

    ###################################################################
    #   Plot the data.
    ###################################################################
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.grid(True, which='major', linewidth=0.5)

    ax1.plot(np.arange(0, iters, 1), means)
    ax2.plot(np.arange(0, iters, 1), stds)
    ax3.plot(means, stds)
    bins = np.arange(int(np.min(data)), int(np.max(data)), 1)
    ax4.hist(data, bins, density=True, alpha = 0.3)
    ax4.scatter(data, stats.norm(loc=mean, scale=std).pdf(data), color='r')

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax1.set_title('MCMC')
    ax1.set_xlabel('\(\mu\)')
    ax1.set_ylabel('\(\sigma\)')
    ax1.legend()

    ###################################################################
    #   Save the Figure to a file in the current working
    #   directory.
    ###################################################################
    #plt.savefig('mcmc.pdf', format='pdf')

    ###################################################################
    #   Show the plot in in a popup window.
    ###################################################################
    plt.show()


if __name__ == '__main__':
    main()
    exit(0)

