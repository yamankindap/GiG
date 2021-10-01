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
    plt.show(block=False)

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

def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False, 
           rug_length=0.05, rug_kwargs=None, font_size=14, title_size=14, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.

    Source: https://stats.stackexchange.com/questions/403652/two-sample-quantile-quantile-plot-in-python 
    Author: Artem Mavrin
    """
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=title_size) 
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)