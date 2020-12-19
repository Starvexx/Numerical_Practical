#! /usr/bin/env python3
"""Project Name

    Author:         David Hernandez
    Matr. Nr.:      01601331
    e-Mail:         david.hernandez@univie.ac.at
    Created:        13.12.2020
    Last edited:    17.12.2020

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
    """Simple function to read the data from a file.

    Parameters:
    -----------
    filename : string
        The path to the file containing the data.

    Returns:
    --------
    data : numpy.array
        The data that was extracted from the data file.
    """
    return np.genfromtxt(filename)


def likelihood(p, mu, sigma):
    """Compute likelihood.

    The likelihood can be computed as follows:

                  1           ⎛    (p - μ)^2 ⎞
        L = ———————————— * exp⎜ - ———————————⎟
             √(2*π*σ^2)       ⎝     2 * σ^2  ⎠

    Where p is our data array. This will yield an array of lieklihoods
    for each data point in the data array. Therefor the likelihood of
    the suggested values μ and σ can be calculated by taking the sum of
    the natural logarithms of the determined likelihoods for the data
    points.

    Parameters:
    -----------
    p : numpy.array
        A numpy.array holding the data.
    mu : scalar
        The mean value (location of the peak) of the distribution.
    sigma : scalar
        The standard deviation (scale or width of the curve) of the
        distribution.

    Returns:
    --------
    L : scalar
        The likelihood for the proposed values μ and σ.

    """
    return np.sum(np.log(1/np.sqrt(2 * np.pi * sigma**2)\
                           * np.exp(- (p - mu)**2 / (2 * sigma**2))))


def metropolis_hastings(p, mu, sigma):
    """Single Metropolis-Hastings iteration.

    In the Metropolis-Hastings algorithm a parameter can be
    approximated by the random walk approach. This method computes the
    likelihood for the given parameter where random noise is added to
    the parameter in every step. Should the parameter with noise be
    more likely than the previous, it will be updated and used for the
    next iteration. Otherwise the current parameter stays the same for
    the next step.

    Parameters:
    -----------
    p : numpy.array
        A numpy.array holding the data.
    mu : scalar
        One of the parameters from the previous iteration.
    sigma : scalar
        The second parameter from the previous iteration

    Returns:
    --------
    mu[_noise] : scalar
        Either the old or the new first parameter.
    sigma[_noise] : scalar
        Either the old or the new second parameter.

    """
    ###################################################################
    #   Add random noise to the initial parameter mu by generating
    #   a normal distribution around mu with σ=sigma_mh=0.1 and 
    #   fetching one random value from the distribution. The same is
    #   done for sigma.
    ###################################################################
    sigma_mh = 0.1

    mu_noise = stats.norm(mu, sigma_mh).rvs()
    sigma_noise = stats.norm(sigma, sigma_mh).rvs()

    ###################################################################
    #   Compute the likelihoods for the previous parameter and the
    #   parameter with noise.
    ###################################################################
    L_noise = likelihood(p, mu_noise, sigma_noise)
    L = likelihood(p, mu, sigma)

    ###################################################################
    #   Compare the two likelihoods and if the new likelihood (with
    #   noise) is greater than the likelihood for the old parameter,
    #   the parameters are updated, otherwise the old values are
    #   returned.
    ###################################################################
    if L < L_noise:
        return mu_noise, sigma_noise

    alpha = np.exp(L_noise - L)
    r = np.random.uniform(0, 1)
    if r < alpha:
        return mu_noise, sigma_noise

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

    ###################################################################
    #   Scaling factors for the start values for the mean and the
    #   standard deviation.
    ###################################################################
    factors = np.random.uniform(-1, 1, 2)

    ###################################################################
    #   starting values for mean and std: mean_0, std_0
    ###################################################################
    mean_0 = mean + mean * factors[0]
    std_0 = std + std * factors[1]
    if std_0 < 0:
        std_0 *= 1

    ###################################################################
    #   Number of iterations and the result arrays for the means and
    #   standard deviations.
    ###################################################################
    iters = 3000
    means = np.zeros(iters)
    stds = np.zeros(iters)

    ###################################################################
    #   Do the iterations for the random walk, and append the results
    #   of each run to the respective arrays.
    ###################################################################
    for i in range(iters):
        means[i] = mean_0
        stds[i] = std_0
        mean_0, std_0 = metropolis_hastings(data, mean_0, std_0)

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
    fig = plt.figure(figsize=(7.27126, 4))       #   figsize in inches

    ###################################################################
    #   Plot the data.
    ###################################################################
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.grid(True, which='major', linewidth=0.5)
    ax2.grid(True, which='major', linewidth=0.5)
    ax3.grid(True, which='major', linewidth=0.5)
    ax4.grid(True, which='major', linewidth=0.5)


    ax1.plot(np.arange(0, iters, 1), means)
    ax2.plot(np.arange(0, iters, 1), stds)
    ax3.plot(means, stds, alpha=0.3)
    ax3.scatter(means[0], stds[0], color='g')
    ax3.scatter(means[-1], stds[-1], color='r')
    bins = np.arange(int(np.min(data)), int(np.max(data)), 1)
    ax4.hist(data, bins, density=True, alpha = 0.3)
    sorted_data = np.sort(data, kind='quicksort')
    ax4.plot(sorted_data,
             stats.norm(loc=mean, scale=std).pdf(sorted_data),
             color='r')

    ###################################################################
    #   Format the subplot.
    ###################################################################
    ax1.set_title('Evolution of the mean')
    ax2.set_title('Evolution of the standard deviation')
    ax3.set_title('mean vs. standard deviation')
    ax4.set_title('Data and PDF')

    ax1.set_xlabel('iterations')
    ax1.set_ylabel('\(\mu\)')
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('\(\sigma\)')
    ax3.set_xlabel('\(\mu\)')
    ax3.set_ylabel('\(\sigma\)')
    ax4.set_xlabel('data')
    ax4.set_ylabel('density')


    fig.subplots_adjust(top=0.935,
                        bottom=0.11,
                        left=0.08,
                        right=0.99,
                        hspace=0.54,
                        wspace=0.295)

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

