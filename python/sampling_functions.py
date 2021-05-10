import numpy as np
from scipy.special import kv
from scipy.special import yv as bessely
from scipy.special import jv as besselj
from scipy.special import hankel1, hankel2
from scipy.special import gamma as gammafnc
from scipy.special import gammainc, gammaincc, gammaincinv

def incgammau(s, x):
    return gammaincc(s,x)*gammafnc(s)

def incgammal(s, x):
    return gammainc(s,x)*gammafnc(s)

def h_stable(gamma, alpha, C=1):
    return np.power((alpha/C)*gamma, np.divide(-1,alpha))

def h_gamma(gamma, a, b):
    return b/(np.exp(gamma/a)-1)

def cornerpoint(nu):
    return np.power(np.power(2, 1-2*np.abs(nu))*np.pi/np.power(gammafnc(np.abs(nu)), 2), 1/(1-2*np.abs(nu)))

def H_squared(z, nu):
    return np.real(hankel1(np.abs(nu), z)*hankel2(np.abs(nu), z))

def sim_condition1(lam, gamma, delta, M=100):
    alpha = 0.5
    C = delta*gammafnc(0.5)/(np.sqrt(2)*np.pi)
    
    # Simulate Stable Process:
    gamma_sequence = np.random.exponential(1, size=M).cumsum()
    h_sequence = h_stable(gamma_sequence, alpha=alpha, C=C)

    # Temper Stable Process:
    tempering_function = np.exp(-0.5*h_sequence*gamma**2)
    u = np.random.uniform(low=0.0, high=1.0, size=h_sequence.size)
    tempered_sequence = h_sequence[u < tempering_function]
    
    # Simulate Square-Root Gamma Process:
    z_series = np.sqrt(np.random.gamma(shape=0.5, scale=np.power(tempered_sequence/(2*np.power(delta,2)), -1.0)))
    
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = 2/(hankel_squared*z_series*np.pi)
    
    u = np.random.uniform(low=0.0, high=1.0, size=tempered_sequence.size)
    jump_sequence = tempered_sequence[u < acceptance_prob]
    return jump_sequence

def sim_positive_extension(lam, gamma, M=100):
    # Set constants:
    a = lam
    b = 2/np.power(gamma, 2)
    
    # Simulate Gamma Process:
    gamma_sequence = np.random.exponential(1, size=M).cumsum()
    x_series = h_gamma(gamma_sequence, a=a, b=b)
    
    # Thin Gamma Process:
    thinning_function = (1+x_series/b)*np.exp(-x_series/b)                                        
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    thinned_series = x_series[u < thinning_function]
    return thinned_series

def sim_condition2_N1(lam, gamma, delta, M=100):
    # Set constants:
    z0 = cornerpoint(lam)
    H0 = z0*H_squared(z0, lam)
    a1 = z0/((np.pi**2)*np.abs(lam)*(1+np.abs(lam))*H0)
    b1 = 2/np.power(gamma, 2)
    a2 = z0/((np.pi**2)*(1+np.abs(lam))*H0)
    b2 = (((z0**2)/(2*delta**2))+((gamma**2)/(2)))**(-1)

    # Simulate Gamma Process:
    gamma_sequence1 = np.random.exponential(1, size=int(M/2)).cumsum()
    x_series1 = h_gamma(gamma_sequence1, a=a1, b=b1)
    gamma_sequence2 = np.random.exponential(1, size=int(M/2)).cumsum()
    x_series2 = h_gamma(gamma_sequence2, a=a2, b=b2)
    x_series = np.sort(np.append(x_series1, x_series2))[::-1]
    x_series = x_series[x_series>0]

    # Thin Gamma Process:
    thinning_function = ((np.abs(lam)*(1+np.abs(lam))*incgammal(np.abs(lam),(z0**2)*x_series/(2*delta**2)))
                         /((((z0**2)*x_series/(2*delta**2))**(np.abs(lam)))
                           *(1+np.abs(lam)*np.exp(-(((z0**2)*x_series/(2*delta**2)))))))                                          
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    thinned_series = x_series[u < thinning_function]

    # Simulate Truncated Square-Root Gamma Process:
    u = np.random.uniform(low=0.0, high=1.0, size=thinned_series.size)
    z_series = np.sqrt(((2*delta**2)/thinned_series)*gammaincinv(0.5, 
                                                          u*(gammaincc(0.5, 
                                                                      (z0**2)*thinned_series/(2*delta**2)))+
                                                                gammainc(0.5,
                                                                         (z0**2)*thinned_series/(2*delta**2))))
    
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = H0/(hankel_squared*(z_series**(2*np.abs(lam))/(z0**(2*np.abs(lam)-1))))
    acceptance_prob[np.isnan(acceptance_prob)] = 1
    acceptance_prob = np.minimum(1, acceptance_prob)

    u = np.random.uniform(low=0.0, high=1.0, size=thinned_series.size)
    jump_sequence = thinned_series[u < acceptance_prob]
    return jump_sequence

def sim_condition2_N2(lam, gamma, delta, M=100):
    # Set constants:
    z0 = cornerpoint(lam)
    H0 = z0*H_squared(z0, lam)
    C = (2*delta**2)/(z0*H0*np.pi**2)
    alpha = 1

    # Simulate Stable Process:
    gamma_sequence = np.random.exponential(1, size=M).cumsum()
    h_sequence = h_stable(gamma_sequence, alpha=alpha, C=C)

    # Temper Stable Process:
    tempering_function = (np.exp(-0.5*h_sequence*(gamma**2+(z0**2)/(delta**2)))
                          *(z0*np.sqrt(h_sequence)*incgammau(0.5, ((z0**2)*h_sequence/(2*delta**2))))
                          /(np.sqrt(2*delta**2)*np.exp(-((z0**2)*h_sequence/(2*delta**2)))))
    u = np.random.uniform(low=0.0, high=1.0, size=h_sequence.size)
    tempered_sequence = h_sequence[u < tempering_function]

    # Simulate Truncated Square-Root Gamma Process:
    u = np.random.uniform(low=0.0, high=1.0, size=tempered_sequence.size)
    z_series = np.sqrt(((2*delta**2)/tempered_sequence)*gammaincinv(0.5, 
                                                          u*(gammaincc(0.5, 
                                                                      (z0**2)*tempered_sequence/(2*delta**2)))+
                                                                gammainc(0.5,
                                                                         (z0**2)*tempered_sequence/(2*delta**2))))
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = H0/(hankel_squared*z_series)
    
    u = np.random.uniform(low=0.0, high=1.0, size=acceptance_prob.size)
    jump_sequence = tempered_sequence[u < acceptance_prob]
    return jump_sequence    

def sim_condition2(lam, gamma, delta, M):
    N1 = sim_condition2_N1(lam, gamma, delta, M=int(M/2))
    N2 = sim_condition2_N2(lam, gamma, delta, M=int(M/2))
    N = np.append(N1, N2)
    return N

def simulate_jump_magnitudes(lam, gamma, delta, M):
    if (np.abs(lam) > 0.5):
        N = sim_condition1(lam, gamma, delta, M)
    else:
        N = sim_condition2(lam, gamma, delta, M)
        
    if lam > 0:
        Np = sim_positive_extension(lam, gamma, M)
        N = np.append(N, Np)
    return N

def simulate_GIG_process(lam, gamma, delta, M=100, interval=(0,1)):
    N = simulate_jump_magnitudes(lam, gamma, delta, M)
    event_times = np.random.uniform(low=interval[0], high=interval[1], size=N.size) 
    process = (event_times, N)
    return process

def simulate_GIG_rv(lam, gamma, delta, M=100, N=1):
    sample = np.array([])
    for i in range(N):
        sample = np.append(sample, simulate_jump_magnitudes(lam, gamma, delta, M).sum())
    return sample

def W(t, process):
    """
    The process variable is a tuple data structure that contains the event times and jump magnitudes,
    respectively.
    """
    return process[1][np.nonzero(process[0] <= t)[0]].sum()