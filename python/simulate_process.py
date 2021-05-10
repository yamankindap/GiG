from sampling_functions import simulate_GIG_process, W
from plotting_functions import process_plot

lam = 0.8
delta = 2
gamma = 0.1
M = 10000 # Length of series

process = simulate_GIG_process(lam, gamma, delta, M)

process_plot(process, lam, gamma, delta)