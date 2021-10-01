from sampling_functions import simulate_GIG_rv
from sampling_functions import random_GIG_size_enforced
from plotting_functions import histogram, qqplot
import matplotlib.pyplot as plt
from scipy import stats


lam = -0.8
delta = 2
gamma = 0.1
M = 1000 # Length of series
N = 10000 #Number of samples from the GIG distribution

sample1 = simulate_GIG_rv(lam, gamma, delta, M, N)
sample2 = random_GIG_size_enforced(lam, gamma, delta, N)

print('--------------------------------------------------------')
print(stats.ks_2samp(sample1, sample2))
print('--------------------------------------------------------')

histogram(sample1, lam, gamma, delta)

plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,10))
qqplot(x=sample2, y=sample1, c='g', alpha=0.8, edgecolor='k', ax=ax, quantiles=1000)
ax.grid(True)
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('RV sampler', fontsize=15)
plt.ylabel('Levy process at t=1',fontsize=15)
plt.show(block=True)

