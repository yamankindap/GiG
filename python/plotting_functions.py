import matplotlib.pyplot as plt
import numbers
import numpy as np
from scipy.special import kv
from scipy.special import gamma as gammafnc
from sampling_functions import W

def histogram_filter(x, lb=0, ub=1):
    """Truncates the tail of samples for better visualisation.
    
    Parameters
    ----------
    x : array-like
        One-dimensional numeric arrays.
    
    lb : float in [0, 1], optional
        Defines the lower bound quantile
    
    ub : float in [0, 1], optional
        Defines the upper bound quantile
    """
    return x[(np.quantile(x, q=lb) < x) & (x < np.quantile(x, q=ub))]

def gig_pdf(x, lam, gamma, delta):
    return np.power(gamma/delta, lam)*(1/(2*kv(lam, delta*gamma))*np.power(x, lam-1)*np.exp(-(gamma**2*x+delta**2/x)/2))

def reciprocal_gamma_pdf(x, alpha, beta):
    return (np.power(beta, alpha)/gammafnc(alpha))*np.power(1/x, alpha+1)*np.exp(-beta/x)

def histogram(sample, lam, gamma, delta, figsize=(14,8)):
    plt.ion()
    density_lim = sample.max()
    fig, ax = plt.subplots(figsize=figsize)

    bins = np.arange(1e-5,10000,100)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    support = np.logspace(-3, np.log10(bins[-1]), num=10000)

    ax.hist(np.clip(sample, bins[0], bins[-1]),
            bins=logbins,
            color='#1f77b4',
            density=True,
            label=r'$\lambda = {}$, $\gamma = {}$, $\delta = {}$'.format(lam, gamma, delta))
    if (gamma == 0) and (lam < 0):
        ax.plot(support, reciprocal_gamma_pdf(support, -lam, 0.5*delta**2), c='#ff7f0e', lw=2.5)
    else:
        ax.plot(support, gig_pdf(support, lam, gamma, delta), c='#ff7f0e', lw=2.5)

    ax.set_xscale('log')
    #ax.set_yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(fontsize=18)
    #plt.savefig(r"GIG_rv_plot with lambda={} gamma={} delta={}.png".format(lam, gamma, delta))
    plt.show(block=True)

def process_plot(process, lam, gamma, delta, figsize=(14,8)):
    time_sequence = np.linspace(start=0, stop=1, num=process[0].size)
    position = [W(t, process) for t in time_sequence]

    plt.ion()
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(time_sequence, position, where='post', lw=2.0)
    ax.grid(True)
    plt.xlabel('time', fontsize=18)
    plt.ylabel('W(t)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show(block=True)
    #plt.savefig(r'plots/gigprocess lambda = {} - gamma = {} - delta = {} - M = {}.jpg'.format(lam, gamma, delta, M), dpi=150)