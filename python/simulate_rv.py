from sampling_functions import simulate_GIG_rv
from plotting_functions import histogram

lam = 0.8
delta = 2
gamma = 0.1
M = 10000 # Length of series
N = 10000 # Number of samples from the GIG distribution

sample = simulate_GIG_rv(lam, gamma, delta, M, N)

histogram(sample, lam, gamma, delta)