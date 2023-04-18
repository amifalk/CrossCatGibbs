import numpy as np
from numpy.random import default_rng

import seaborn as sns
import matplotlib.pyplot as plt


# RunDEMC utils
# -----------------------------------------


def posterior_sample(model, burnin=0):
    """Return a list containing a randomly selected posterior sample from each parameter"""

    selections = [
        default_rng().choice(model.particles[burnin:, :, i].ravel())
        for i in range(model.particles.shape[-1])
    ]

    return selections


def MAP_estimates(model, burnin=0):
    """retrieve a list of MAP estimates for each parameter in the model"""
    best_ind = model.weights[burnin:].argmax()

    estimates = [
        model.particles[burnin:, :, i].ravel()[best_ind]
        for i in range(model.particles.shape[-1])
    ]

    return estimates


def plot_dist(model, burnin=0):
    """plot parameter posterior distributions"""
    samples = model.particles[burnin:]
    all_chains = samples.reshape(samples.shape[0] * samples.shape[1], -1)

    sns.kdeplot(all_chains)
    plt.title("Estimated Posteriors")
    plt.legend(model.param_display_names)
    plt.show()


def plot_loglik(model):
    """plot mean log likelihood of the chains over time"""
    log_liks = np.mean(model.weights, axis=1)
    sns.lineplot(log_liks)
    plt.title(f"Log Likelihood Averaged over {model._num_chains} Chains")
    plt.xlabel("Iterations")
    plt.ylabel("Log Likelihood")
    plt.show()


# general utils
# -----------------------------------------


def log_linspace(a, b, n):
    """
    linspace from a to b with n entries over log scale
    https://github.com/probcomp/cgpm/blob/56a481829448bddc9cdfebd42f65023287d5b7c7/src/utils/general.py
    """
    return np.exp(np.linspace(np.log(a), np.log(b), n))


def exp_normalize(x):
    """
    numerically stable method to normalize log probs into probs
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    """
    b = np.max(x)
    y = np.exp(x - b)
    return y / np.sum(y)
