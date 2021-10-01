from scipy import stats
from scipy import special
import numpy as np
from scipy.special import kv
from scipy.special import yv as bessely
from scipy.special import jv as besselj
from scipy.special import hankel1, hankel2
from scipy.special import gamma as gammafnc
from scipy.special import gammainc, gammaincc, gammaincinv

def thinning_function(delta, x):
    return np.exp(-(1/2)*(np.power(delta, 2)*(1/x)))

def reciprocal_sample(x, i):
    return x**(i)

def random_GIG(lam, gamma, delta, size=1):
    i = 1
    if lam < 0:
        tmp = gamma
        gamma = delta
        delta = tmp
        lam = -lam
        i = -1
    
    shape = lam
    scale = 2/np.power(gamma, 2)
    
    gamma_rv = np.random.gamma(shape=shape, scale=scale, size=size)
    u = np.random.uniform(low=0.0, high=1.0, size=size)
    sample = gamma_rv[u < thinning_function(delta, gamma_rv)]
    return reciprocal_sample(sample, i)

def random_GIG_size_enforced(lam, gamma, delta, size=1):
    sample = np.array([])
    while sample.size < size:
        sample = np.append(sample, random_GIG(lam, gamma, delta, size=size))
    return sample[np.random.randint(low=0, high=sample.size, size=size)]

def incgammau(s, x):
    return gammaincc(s,x)*gammafnc(s)

def incgammal(s, x):
    return gammainc(s,x)*gammafnc(s)

def h_stable(gamma, alpha, C=1):
    return np.power((alpha/C)*gamma, np.divide(-1,alpha))

def h_gamma(gamma, C, beta):
    return 1/(beta*(np.exp(gamma/C)-1))

def cornerpoint(nu):
    return np.power(np.power(float(2), 1-2*np.abs(nu))*np.pi/np.power(gammafnc(np.abs(nu)), 2), 1/(1-2*np.abs(nu)))

def H_squared(z, nu):
    return np.real(hankel1(np.abs(nu), z)*hankel2(np.abs(nu), z))

def sim_gamma_jumps(C, beta, M):
    gamma_sequence = np.random.exponential(1, size=M).cumsum()
    x_series = h_gamma(gamma_sequence, C=C, beta=beta)
    thinning_function = (1+beta*x_series)*np.exp(-beta*x_series)
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    return x_series[u < thinning_function]

def sim_stable_jumps(alpha, C, M):
    gamma_sequence = np.random.exponential(1, size=M).cumsum()
    x_series = h_stable(gamma_sequence, alpha=alpha, C=C)
    return x_series

def sim_tempered_stable_jumps(alpha, beta, C, M):
    gamma_sequence = np.random.exponential(1, size=M).cumsum()
    x_series = h_stable(gamma_sequence, alpha=alpha, C=C)
    thinning_function = np.exp(-beta*x_series)
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    return x_series[u < thinning_function]

def sim_phase1_method1(lam, gamma, delta, M):
    alpha = 0.5
    C = delta*gammafnc(0.5)/(np.sqrt(2)*np.pi)
    beta = 0.5*gamma**2
    x_series = sim_tempered_stable_jumps(alpha, beta, C, M)
    z_series = np.sqrt(np.random.gamma(shape=0.5, scale=np.power(x_series/(2*np.power(delta,2)), -1.0)))
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = 2/(hankel_squared*z_series*np.pi)
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    jump_sequence = x_series[u < acceptance_prob]
    return jump_sequence

def sim_phase1_method2_N1(lam, gamma, delta, M): 
    z1= cornerpoint(lam)
    abs_lam = np.abs(lam)
    C = z1/(np.pi*abs_lam*2)
    beta = 0.5*gamma**2
    
    x_series = sim_gamma_jumps(C, beta, M)
    #envelope_fnc = incgammal(abs_lam, (z1**2)*x_series/(2*delta**2))*(abs_lam*(2*delta**2)**(abs_lam))/((x_series**abs_lam)*z1**(2*abs_lam))
    envelope_fnc = abs_lam*incgammal(abs_lam, (z1**2)*x_series/(2*delta**2))/(((z1**2)*x_series/(2*delta**2))**abs_lam)
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    x_series = x_series[u < envelope_fnc]
    
    # Simulate Truncated Square-Root Gamma Process:
    u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)    
    z_series = np.sqrt(((2*delta**2)/x_series)*gammaincinv(abs_lam, u_z*gammainc(abs_lam, (z1**2)*x_series/(2*delta**2))))
    
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*abs_lam))/(z1**(2*abs_lam-1))))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    jump_sequence = x_series[u < acceptance_prob]
    return jump_sequence

def sim_phase1_method2_N1_alternative(lam, gamma, delta, M): 
    z1= cornerpoint(lam)
    abs_lam = np.abs(lam)
    C1 = z1/(np.pi*abs_lam*2*(1+abs_lam))
    beta1 = 0.5*gamma**2
    C2 = z1/(np.pi*2*(1+abs_lam))
    beta2 = 0.5*gamma**2 + (z1**2)/(2*delta**2)
    
    x_series1 = sim_gamma_jumps(C1, beta1, M)
    x_series2 = sim_gamma_jumps(C2, beta2, M)
    x_series = np.append(x_series1, x_series2)
    
    envelope_fnc = ((2*delta**2)**abs_lam)* incgammal(abs_lam, (z1**2)*x_series/(2*delta**2))*abs_lam*(1+abs_lam)/((x_series**abs_lam)*(z1**(2*abs_lam))*(1+abs_lam*np.exp(-(z1**2)*x_series/(2*delta**2))))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    x_series = x_series[u < envelope_fnc]
    
    # Simulate Truncated Square-Root Gamma Process:
    u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)    
    z_series = np.sqrt(((2*delta**2)/x_series)*gammaincinv(abs_lam, u_z*gammainc(abs_lam, (z1**2)*x_series/(2*delta**2))))
    
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*abs_lam))/(z1**(2*abs_lam-1))))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    jump_sequence = x_series[u < acceptance_prob]
    return jump_sequence

    
def sim_phase1_method2_N2(lam, gamma, delta, M): #
    z1 = cornerpoint(lam)
    C = delta*gammafnc(0.5)/(np.sqrt(2)*np.pi)
    alpha = 0.5
    beta = 0.5*gamma**2
    x_series = sim_tempered_stable_jumps(alpha, beta, C, M)
    envelope_fnc = gammaincc(0.5, (z1**2)*x_series/(2*delta**2))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    x_series = x_series[u < envelope_fnc]
    
    # Simulate Truncated Square-Root Gamma Process:
    u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    z_series = np.sqrt(((2*delta**2)/x_series)*gammaincinv(0.5, u_z*(gammaincc(0.5, (z1**2)*x_series/(2*delta**2)))
                                                           + gammainc(0.5, (z1**2)*x_series/(2*delta**2))))
    
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = 2/(hankel_squared*z_series*np.pi)
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    jump_sequence = x_series[u < acceptance_prob]
    return jump_sequence

def sim_phase1_method2(lam, gamma, delta, M, method='1gamma'):
    if method == '1gamma':
        N1 = sim_phase1_method2_N1(lam, gamma, delta, M=M)
    elif method == '2gamma':
        N1 = sim_phase1_method2_N1_alternative(lam, gamma, delta, M=M)
    else:
        raise ValueError('method undefined')
    N2 = sim_phase1_method2_N2(lam, gamma, delta, M=M)
    N = np.append(N1, N2)
    return N

def sim_phase2_N1_method1(lam, gamma, delta, M):
    z0 = cornerpoint(lam)
    H0 = z0*H_squared(z0, lam)
    abs_lam = np.abs(lam)
    C = z0/((np.pi**2)*abs_lam*H0)
    beta = 0.5*gamma**2
    
    # Alternative (faster?) simulation algorithm:
    #gamma_sequence = np.random.exponential(1, size=M).cumsum()
    #x_series = h_gamma(gamma_sequence, C=C, beta=beta)
    #envelope_fnc = ((1+beta*x_series)*np.exp(-beta*x_series)*abs_lam*incgammal(abs_lam, (z0**2)*x_series/(2*delta**2))
    #                /(((z0**2)*x_series/(2*delta**2))**abs_lam))
    #u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    #x_series = x_series[u < envelope_fnc]
    
    x_series = sim_gamma_jumps(C, beta, M)
    envelope_fnc = abs_lam*incgammal(abs_lam, (z0**2)*x_series/(2*delta**2))/(((z0**2)*x_series/(2*delta**2))**abs_lam)
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    x_series = x_series[u < envelope_fnc]
    
    # Simulate Truncated Square-Root Gamma Process:
    u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    z_series = np.sqrt(((2*delta**2)/x_series)*gammaincinv(abs_lam, u_z*(gammainc(abs_lam, (z0**2)*x_series/(2*delta**2)))))
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = H0/(hankel_squared*((z_series**(2*abs_lam))/(z0**(2*abs_lam-1))))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    jump_sequence = x_series[u < acceptance_prob]
    return jump_sequence

def sim_phase2_N1_method2(lam, gamma, delta, M):
    z0= cornerpoint(lam)
    H0 = z0*H_squared(z0, lam)
    abs_lam = np.abs(lam)
    C1 = z0/((np.pi**2)*H0*abs_lam*(1+abs_lam))
    beta1 = 0.5*gamma**2
    C2 = z0/((np.pi**2)*(1+abs_lam)*H0)
    beta2 = 0.5*gamma**2 + (z0**2)/(2*delta**2)
    
    x_series1 = sim_gamma_jumps(C1, beta1, M)
    x_series2 = sim_gamma_jumps(C2, beta2, M)
    x_series = np.append(x_series1, x_series2)
    
    envelope_fnc = ((2*delta**2)**abs_lam)* incgammal(abs_lam, (z0**2)*x_series/(2*delta**2))*abs_lam*(1+abs_lam)/((x_series**abs_lam)*(z0**(2*abs_lam))*(1+abs_lam*np.exp(-(z0**2)*x_series/(2*delta**2))))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    x_series = x_series[u < envelope_fnc]
    
    # Simulate Truncated Square-Root Gamma Process:
    u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    z_series = np.sqrt(((2*delta**2)/x_series)*gammaincinv(abs_lam, u_z*(gammainc(abs_lam, (z0**2)*x_series/(2*delta**2)))))
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = H0/(hankel_squared*((z_series**(2*abs_lam))/(z0**(2*abs_lam-1))))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    jump_sequence = x_series[u < acceptance_prob]
    return jump_sequence
    

def sim_phase2_N2(lam, gamma, delta, M):
    z0 = cornerpoint(lam)
    H0 = z0*H_squared(z0, lam)
    C = np.sqrt(2*delta**2)*gammafnc(0.5)/(H0*np.pi**2)
    alpha = 0.5
    beta = 0.5*gamma**2
    x_series = sim_tempered_stable_jumps(alpha, beta, C, M)
    envelope_fnc = gammaincc(0.5, (z0**2)*x_series/(2*delta**2))
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    x_series = x_series[u < envelope_fnc]
    
    # Simulate Truncated Square-Root Gamma Process:
    u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    z_series = np.sqrt(((2*delta**2)/x_series)*gammaincinv(0.5, u_z*(gammaincc(0.5, (z0**2)*x_series/(2*delta**2)))
                                                           +gammainc(0.5, (z0**2)*x_series/(2*delta**2))))
    hankel_squared = H_squared(z_series, lam)
    acceptance_prob = H0/(hankel_squared*z_series)
    u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
    jump_sequence = x_series[u < acceptance_prob]
    return jump_sequence

def sim_phase2(lam, gamma, delta, M, method='1gamma'):
    if method == '1gamma':
        N1 = sim_phase2_N1_method1(lam, gamma, delta, M=M)
    elif method == '2gamma':
        N1 = sim_phase2_N1_method2(lam, gamma, delta, M=M)
    else:
        raise ValueError('method undefined')
    N2 = sim_phase2_N2(lam, gamma, delta, M=M)
    N = np.append(N1, N2)
    return N
    
def sim_positive_extension(lam, gamma, M=100):
    C = lam
    beta = 0.5*gamma**2
    x_series = sim_gamma_jumps(C, beta, M)
    return x_series

def simulate_jump_magnitudes(lam, gamma, delta, M):
    if (np.abs(lam) > 0.5):
        N = sim_phase1_method1(lam, gamma, delta, M)
    else:
        N = sim_phase2(lam, gamma, delta, M)
        
    if lam > 0:
        Np = sim_positive_extension(lam, gamma, M)
        N = np.append(N, Np)
    return N

def simulate_GIG_process(lam, gamma, delta, M=1000, interval=(0,1)):
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