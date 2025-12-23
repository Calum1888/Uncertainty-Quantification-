# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:20:51 2025

@author: c_reg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

#--------QUESTION 1a--------

#function to calculate forward Euler approximation
#input variables are A,B,X_0,Y_0,dt,T for greater flexibility

def forward_euler(A, B, X_0, Y_0, dt, T):
    
    X_n = [X_0] #vector of X values, with X_0 
    
    Y_n = [Y_0] #vector of Y values with Y_0
    
    steps = int(T/dt) #how many steps we want to compute for 
    
    #creates an array of numbers with spacing "steps"
    t = np.linspace(0, T, steps + 1)
    
    for i in range(0,steps): #for each step do the following
        
        #calculate X_(n+1) from our forward Euler approximation
        #add the value of X_(n+1) to the vector of X_ns 
        X_n.append(X_n[i]+(A-(B+1.0)*X_n[i]+((X_n[i])**2)*Y_n[i])*dt) 
        #AI used for ** instead of ^
        
        #calculate Y_(n+1) from our forward Euler approximation
        #add the value of Y_(n+1) to the vector of Y_ns
        Y_n.append(Y_n[i]+(B*X_n[i]-((X_n[i])**2)*Y_n[i])*dt) 
        
        #AI used here to debug power as I used ^ instead of **
        
    # Create a DataFrame for X_n, Y_n values for easy malnipulation
    X_Y_df = pd.DataFrame({
    'Time': t,
    'X_n': X_n,
    'Y_n': Y_n
    })
        
    return X_Y_df #return the dataframe of values of X_n,Y_n

#compute for different time steps 0.1,0.01,0.001,0.0001
approx_1 = forward_euler(1, 1.8, 1.2, 2.0, 0.1, 50) #timestep = 0.1
approx_2 = forward_euler(1, 1.8,1.2, 2.0, 0.01, 50) #timestep = 0.01
approx_3 = forward_euler(1, 1.8,1.2, 2.0, 0.001, 50) #timestep = 0.001
approx_4 = forward_euler(1, 1.8,1.2, 2.0, 0.0001, 50) #timestep = 0.0001

#plot approximations for X on same graph
plt.figure(figsize=(12,8))
plt.plot(approx_1['Time'], approx_1['X_n'], label='dt=0.1')
plt.plot(approx_2['Time'], approx_2['X_n'], label='dt=0.01')
plt.plot(approx_3['Time'], approx_3['X_n'], label='dt=0.001')
plt.plot(approx_4['Time'], approx_4['X_n'], label='dt=0.0001')  
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.title('Forward Euler Approximation of X(t) for Different Timesteps')
plt.legend()
plt.show()

#plot approximations for Y on same graph
plt.figure(figsize=(12,8))
plt.plot(approx_1['Time'], approx_1['Y_n'], label='dt=0.1')
plt.plot(approx_2['Time'], approx_2['Y_n'], label='dt=0.01')
plt.plot(approx_3['Time'], approx_3['Y_n'], label='dt=0.001')
plt.plot(approx_4['Time'], approx_4['Y_n'], label='dt=0.0001')  
plt.xlabel('Time')
plt.ylabel('Y(t)')
plt.title('Forward Euler Approximation of Y(t) for Different Timesteps')
plt.legend()
plt.show()

#error of the forward Euler method is |"actual solution"-"computed solution"|
#dont have actual solution so we will use values at timestep 0.0001
#as it is a good approximation of the true solution.

#function to compute error using timestep 0.0001 as the true value
def approx_error(approximation, dt, T):
    
    #define fine and coarse, makes sure we compare true and approximate values 
    #at the right timepoints instead of by the index of the dataframe
    
    #fine solution (dt = 0.0001)
    fine = approx_4.set_index("Time")

    #coarse solution
    coarse = approximation.set_index("Time")

    #map fine solution onto coarse time grid
    fine_interp = fine.reindex(coarse.index).interpolate()
    
    #error calculation
    error_X = np.abs(fine_interp['X_n'] - coarse['X_n'])
    error_Y = np.abs(fine_interp['Y_n'] - coarse['Y_n'])
    
    #add values to a dataframe for easier malnipulation 
    error = pd.DataFrame({
        'Time': coarse.index,
        'X_error': error_X,
        'Y_error': error_Y
    })

    return error

#compute the error at each timepoint
error_1 = approx_error(approx_1, 0.1, 50)
error_2 = approx_error(approx_2, 0.01, 50)
error_3 = approx_error(approx_3, 0.001, 50)

#plotting the error of X for the three time steps
plt.figure(figsize=(12,8))
plt.plot(error_1['Time'], error_1['X_error'], label='dt=0.1')
plt.plot(error_2['Time'], error_2['X_error'], label='dt=0.01')
plt.plot(error_3['Time'], error_3['X_error'], label='dt=0.001')
plt.xlabel('Time')
plt.ylabel('Error in X(t)')
plt.title('Forward Euler Error of X(t) for Different Timesteps')
plt.legend()
plt.show()

#plotting the error of X for the three time steps
plt.figure(figsize=(12,8))
plt.plot(error_1['Time'], error_1['Y_error'], label='dt=0.1')
plt.plot(error_2['Time'], error_2['Y_error'], label='dt=0.01')
plt.plot(error_3['Time'], error_3['Y_error'], label='dt=0.001')
plt.xlabel('Time')
plt.ylabel('Error in Y(t)')
plt.title('Forward Euler Error of Y(t) for Different Timesteps')
plt.legend()
plt.show()

#create one large dataframe of all error values
#we will take values at T=50 and show order of convergence
error_df = pd.DataFrame({
    'Time': error_1['Time'],
    'Error1_X': error_1['X_error'],
    'Error1_Y': error_1['Y_error'],
    'Error2_X': error_2['X_error'],
    'Error2_Y': error_2['Y_error'],
    'Error3_X': error_3['X_error'],
    'Error3_Y': error_3['Y_error'],
})

#select row T=50
error_time50 = error_df[error_df['Time'] == 50]

#time steps as points on x axis
dt_values = [0.1, 0.01, 0.001]

#get scalar errors at T=50 for X
error_X = [
    error_time50['Error1_X'].values[0],
    error_time50['Error2_X'].values[0],
    error_time50['Error3_X'].values[0]
]

#get scalar errors at T=50 for Y
error_Y = [
    error_time50['Error1_Y'].values[0],
    error_time50['Error2_Y'].values[0],
    error_time50['Error3_Y'].values[0]
]

#plot error at three time steps at T=50 for X for order of convergence
plt.figure(figsize=(12,8))
plt.plot(dt_values, error_X, 'o-', label='X Error')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Step size dt')
plt.ylabel('Error in X(t)')
plt.title('Order of Convergence of Forward Euler Method for X(t)')
plt.legend()
plt.show()

#plot error at three time steps at T=50 for Y for order of convergence
plt.figure(figsize=(12,8))
plt.plot(dt_values, error_Y, 'o-', label='Y Error')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Step size dt')
plt.ylabel('Error in Y(t)')
plt.title('Order of Convergence of Forward Euler Method for Y(t)')
plt.legend()
plt.show()


#--------QUESTION 1b--------

#function to compute time-averaged variance for one simulation
#T0, T1 denote start and end times of the period of interest
#other parameters are same as foward_euler
def compute_time_avg_variance(A, B, X_0, Y_0, dt, T, T0, T1):
    
    #values from forward Euler approximation
    df = forward_euler(A, B, X_0, Y_0, dt, T)
    
    #look at time period we are interested in 
    X_interest = df[(df['Time'] >= T0) & (df['Time'] <= T1)]['X_n'].values
    
    #compute variance
    var_X = np.var(X_interest, ddof=1)
    
    return var_X

#function to perform Monte Carlo estimation
#number of samples is set at 5000
#range of possible values of B is [1.5,2.5]
def monte_carlo_variance(A, X_0, Y_0, dt, T, T0, T1, num_MC=5000, 
                         B_range=(1.5, 2.5)):
    
    #stores each estimate of time-averaged variance
    var_list = []
    
    #for each Monte Carlo simulation the function will
    #1)sample from the uniform distribution B~Uniform([1.5,2.5])
    #2)compute the time average variance for that value of B
    #3)add that value for the variance to the list 
    for _ in range(num_MC):
        B = np.random.uniform(*B_range)
        var_X = compute_time_avg_variance(A, B, X_0, Y_0, dt, T, T0, T1)
        var_list.append(var_X)
    
    #turn into an array for easier handling
    var_array = np.array(var_list)
    
    #take the expected value of the time-averaged variance 
    expected_var = np.mean(var_array)
    
    return var_array, expected_var

#run Monte Carlo with following parameters
var_array, expected_var = monte_carlo_variance(A=1, 
                                               X_0=1.2,
                                               Y_0=2.0,
                                               dt=0.01,
                                               T=50,
                                               T0=30,
                                               T1=50,
                                               num_MC=5000)

#want to compute number of samples for certain error
#need variance of the quantity we are estimating
variance_var = np.var(var_array)

#desired monte carlo error
monte_carlo_error = 0.001

#formula for number of samples
M = variance_var/((monte_carlo_error)**2)

#displays the expected value of the variance, 
#variance of time-averaged variance, number of samples for certain error 
print(f"Estimated E[Var[X(t)]] over [30,50]: {expected_var:.4f}")
print(f"Variance of time-averaged variance : {variance_var:.4f}")
print(f"Minimum number of samples to have Monte Carlo error of 0.01: {M:.4f}")

#function to plot density
def plot_density(var_array):
    plt.figure(figsize=(10,6))
    plt.hist(var_array, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.xlabel('Time-averaged variance of X(t)')
    plt.ylabel('Density')
    plt.title('Monte Carlo Density of Time-averaged Variance')
    plt.show()

#plot density
plot_density(var_array) 

#--------QUESTION 2a--------

#function for the Euler-Maruyama method.
def euler_mar(X_V, Y_V, A, B, V, dt, T):
    
    #divide by volume to get X_0,Y_0
    X_0 = X_V / V
    Y_0 = Y_V / V
    
    #number of steps to find X,Y at
    steps = int(T / dt)
    
    #create two vectors of length steps and fill with zeros to store data
    X = np.zeros(steps + 1)
    Y = np.zeros(steps + 1)
    
    #creates an array of numbers with spacing "steps"
    t = np.linspace(0, T, steps + 1)

    #set our vectors of X and Y to contain X_0,Y_0
    X[0] = X_0
    Y[0] = Y_0
    
    for n in range(steps):

        #for each time step, generate 4 Weier terms with mean 0, sigma 1
        #multiplied by square root of time step
        dW1, dW2, dW3, dW4 = np.sqrt(dt) * np.random.normal(0, 1, 4)

        #drift/deterministic terms for X and Y
        X_drift = A*V - B*X[n] - X[n] + (X[n] * (X[n] - 1) * Y[n]) / V**2
        Y_drift = B*X[n] - (X[n] * (X[n] - 1) * Y[n]) / V**2

        #diffusion/stochastic terms for X,Y
        X_diff = (np.sqrt(A*V) * dW1
              - np.sqrt(B*X[n]) * dW2
              - np.sqrt(X[n]) * dW3
              + np.sqrt((X[n] * (X[n] - 1) * Y[n]) / V**2) * dW4)

        Y_diff = (np.sqrt(B*X[n]) * dW2
              - np.sqrt((X[n] * (X[n] - 1) * Y[n]) / V**2) * dW4)

        #Euler–Maruyama method by combining both drift and diffusion terms
        X[n+1] = X[n] + X_drift * dt + X_diff
        Y[n+1] = Y[n] + Y_drift * dt + Y_diff
        
        #X,Y cannot be negative, 
        #if either is negative then it is set to 0 for that timepoint
        if X[n+1] < 0:
            X[n+1] = 0.0
        if Y[n+1] < 0:
            Y[n+1] = 0.0
        
    #create a datframe that contains all information on X,Y    
    df = pd.DataFrame({
        "Time": t,
        "X_n": X/V,
        "Y_n": Y/V
    })

    return df

#function to plot Euler-Maruyama approximation
def plot_euler_mar(df,dt):
    plt.figure(figsize=(12,8))
    plt.plot(df['Time'], df['X_n'], label='X')
    plt.plot(df['Time'], df['Y_n'], label='Y')
    plt.xlabel('Time')
    plt.ylabel('Quantity')
    plt.title(f' Euler-Maruyama Approximation of X and Y with Timestep {dt}')
    plt.legend()
    plt.show()

df = euler_mar(X_V = 1.2*10000,Y_V = 2*10000,A = 1,B = 1.8,V = 10000,
               dt = 0.01,T = 100.0)

plot_euler_mar(df, 0.01)

#function to plot the ODE next to SDE solution
def plot_ODE_SDE(ode_approx, sde_approx, dt, T):
    plt.figure(figsize=(12,8))
    plt.plot(ode_approx['Time'], ode_approx['X_n'], label='Forward Euler')
    plt.plot(sde_approx['Time'], sde_approx['X_n'], label='Euler-Maruyama')
    plt.xlabel('Time')
    plt.ylabel('Quantity')
    plt.title(f' Forward Euler Approximation against Euler-Maruyama Approximation for X(t) with Timestep={dt}')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12,8))
    plt.plot(ode_approx['Time'], ode_approx['Y_n'], label='Forward Euler')
    plt.plot(sde_approx['Time'], sde_approx['Y_n'], label='Euler-Maruyama')
    plt.xlabel('Time')
    plt.ylabel('Quantity')
    plt.title(f' Forward Euler Approximation against Euler-Maruyama Approximation for Y(t) with Timestep={dt}')
    plt.legend()
    plt.show()

approx = forward_euler(1, 1.8,1.2, 2.0, 0.01, 100) 

plot_ODE_SDE(approx, df, 0.01, 100)

#--------QUESTION 2b--------

#function to compute quantities of interest
#T0,T1 for start and stop time of QoIs
def quantities(A, B, X_V, Y_V, dt, V, T, T0, T1):
    
    #values from forward Euler approximation
    df = euler_mar(X_V, Y_V, A, B, V, dt, T)
    
    #look at time period we are interested in 
    X_interest = df[(df['Time'] >= T0) & (df['Time'] <= T1)]['X_n'].values
    df_interest = df[(df['Time'] >= T0) & (df['Time'] <= T1)]
    
    #compute max of x^2y
    Q_1 = np.max((df_interest['X_n']**2)*df_interest['Y_n'])
    #AI used as didnt know max was in numpy package
    
    #compute variance
    Q_2 = np.var(X_interest, ddof=1)
    
    return Q_1, Q_2

#function to run monte carlo simulation 
def mc_quantities(A, B, X_V, Y_V, dt, V, T, T0, T1, num_MC):
    
    #stores Q_1,Q_2 values
    Q_1_list = []
    Q_2_list = []
    
    for _ in range(num_MC):
        
        #get quantities 
        Q_1, Q_2 = quantities(A, B, X_V, Y_V, dt, V, T, T0, T1)
        
        #add them to corresponding list
        Q_1_list.append(Q_1)
        Q_2_list.append(Q_2)
    
    #convert to arrays for easy malnipulation
    Q_1_array = np.array(Q_1_list)
    Q_2_array = np.array(Q_2_list)
    
    #compute expectation of both quantities
    expected_Q_1 = np.mean(Q_1_array)
    expected_Q_2 = np.mean(Q_2_array)
    
    return Q_1_array, Q_2_array, expected_Q_1, expected_Q_2

#function to plot density for Q_1, Q_2 and joint density
def plot_sde_denisty(Q_1_array, Q_2_array):
    
    plt.figure(figsize=(10,6))
    plt.hist(Q_1_array, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.xlabel('Maximum of x^2y')
    plt.ylabel('Density')
    plt.title('Monte Carlo Density of Maximum of x^2y')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.hist(Q_2_array, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.xlabel('Time-averaged variance of X(t)')
    plt.ylabel('Density')
    plt.title('Monte Carlo Density of Time-averaged Variance') 
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.hist2d(Q_1_array, Q_2_array, bins=50, density=True, cmap="Blues")
    plt.colorbar(label="Density")
    plt.title("2D Density Histogram of Q_1 vs Q_2")
    plt.xlabel("Q_1")
    plt.ylabel("Q_2")
    plt.show()

Q_1_array, Q_2_array, expected_Q_1, expected_Q_2 = mc_quantities(A=1,
                                                                 B=2.2,
                                                                 X_V = 1.2*10000, 
                                                                 Y_V = 2*10000, 
                                                                 dt=0.01,
                                                                 V=10000, 
                                                                 T=50, 
                                                                 T0=30, 
                                                                 T1=50,
                                                                 num_MC=1000)

#expectations of Q_1, Q_2
print(f"Estimated E[max(x^2y) over [30,50]: {expected_Q_1:.4f}")
print(f"Estimated E[Var[X(t)]] over [30,50]: {expected_Q_2:.4f}")

plot_sde_denisty(Q_1_array, Q_2_array)

#want to find number of samples to get a ceratin error
#find variances of quantities
Q_1_var = np.var(Q_1_array)
Q_2_var = np.var(Q_2_array)

#specfified error
EM_monte_carlo_error = 0.0001

#compute number of samples for quantities
M_Q_1 = Q_1_var/((EM_monte_carlo_error)**2)
M_Q_2 = Q_2_var/((EM_monte_carlo_error)**2)

#show number of samples needed to get a certain error
print(f"Samples for Q_1: {M_Q_1:.4f}")
print(f"Samples for Q_2: {M_Q_2:.4f}")


#--------QUESTION 3a---

#data from Table 1
data = np.array([
    [6.87094*10**(-1), 8.86372*10**(-1), 1.16966*10**(0)],
    [2.57202*10**(0), 2.10889*10**(0), 1.48356*10**(0)]
    ])

#times of interest
T_array = [10,15,20]

#parameters of Gaussian noise
mean_noise = 1.0
sigma_noise = 0.1

#parameters of a Gaussian prior
mean_prior = 2.0
sigma_prior = 0.2

#function to estimate the observation operator G(B)
def observation_operator(B, T_array):
    
    #store approximations from forward Euler
    x_vals = []
    y_vals = []

    for T in T_array:
        
        #for each time 
        #1)get the values for the approximation 
        #2)locate the values at time T
        #3)add the X,Y values to corresponding lists
        df = forward_euler(A=1, B=B, X_0=1.2, Y_0=2.0, dt=0.01, T=T)
        
        idx = (df['Time'] - T).abs().idxmin()
        row = df.loc[idx]
        
        x_vals.append(row['X_n'])
        y_vals.append(row['Y_n'])

    #convert into a 2×3 matrix --> same form as in question 
    G = np.vstack([x_vals, y_vals])

    return G

#function to estimate likelihood 
def likelihood(B, data, T_array):
    
    #get observation operator
    G = observation_operator(B, T_array)
    
    #need the difference for ||G(B)-Y|| part of density
    difference = G-data
    
    #calculate likelihood based on Gaussian noise density ~N(0,sigma_noise)
    density = np.exp((-1/(2*sigma_noise**2))*np.sum(difference**2))
    
    return density

#function for density of the prior for B
def prior(B):
    
    #compute density base on Gaussian density ~N(2.0,0.2^2)
    density = (1/(sigma_prior*np.sqrt(2*np.pi)))*np.exp((-1/(2*sigma_prior**2))*(B-mean_prior)**2)
    
    return density #AI caught i was using sigma noise instead of sigma prior 
 
#function to compute posterior density   
def posterior(B, data, T_array):
    
    #get prior
    prior_density = prior(B)
    
    #get likelihood
    like = likelihood(B, data, T_array)
    
    #formula for posterior density
    posterior_density = prior_density*like
    
    return posterior_density
    
#function for evaluating posterior denisty on a grid for B in [1.4,2.6]
#start,stop,spacing denote interavl and space between values on uniform grid
def evaluate_and_plot(start, stop, spacing, data, T_array):
    
    #define our grid of B values
    B_grid = np.linspace(start, stop, int((stop-start)/spacing) + 1)
    
    #compute prior, posterior values for each B
    prior_values = np.array([prior(B) for B in B_grid])
    posterior_values = np.array([posterior(B, data, T_array) for B in B_grid])
    
    #normalise prior, posterior values with trapeziod method
    #this involves numerically intergrating of the grid
    #np.trapezoid is a function in numpy that follows this method
    normalised_prior_values = prior_values/np.trapezoid(prior_values, B_grid)
    normalised_posterior_values = posterior_values/np.trapezoid(posterior_values, B_grid)
    
    #plot the density of the prior and posterior
    plt.figure(figsize=(12,8))
    plt.plot(B_grid, normalised_prior_values, label='Prior')
    plt.plot(B_grid, normalised_posterior_values, label='Posterior')
    plt.xlabel('B')
    plt.ylabel('Density')
    plt.title('Prior and Posterior Densities of B')
    plt.legend()
    plt.show()
    
    return normalised_posterior_values, B_grid
    
normalised_posterior_values, B_grid = evaluate_and_plot(1.4, 2.6, 0.001, 
                                             data=data, T_array=T_array)

#calculate posterior mean and density
posterior_mean = np.trapezoid(B_grid * normalised_posterior_values, B_grid)
posterior_variance = np.trapezoid((B_grid - posterior_mean)**2 * normalised_posterior_values, B_grid)

print(f'Posterior mean: {posterior_mean:.4f}')
print(f'Posterior variance: {posterior_variance:.4f}')

#--------QUESTION 3b--------

#function for random walk Metropolis Hastings 
def RWMH(beta, B_0, RWMH_samples, data, T_array):
    
    #stores values of B starting with initial value
    B = [B_0]
    
    #dictionary to cache posterior evaluations
    posterior_cache = {}
    
    #precompute posterior for initial state
    posterior_cache[B_0] = posterior(B_0, data=data, T_array=T_array)
    
    for i in range(RWMH_samples):
        
        #for each sample number
        #1)sample Y from N(B_(i-1),beta^2)
        #2)calculate the acceptance probability 
        #with Metropolis-Hastings formula
        #3)accept or reject Y
        Y = np.random.normal(loc=B[-1], scale=beta)
        
        #posterior cache speeds up the calculation of 
        #posteriors to speed up overall runtime
        #by checking if the posterior for that value of 
        #B has already been calculated
        #if not it is calculated anyway and added to the cache
        if Y in posterior_cache:
            post_Y = posterior_cache[Y]
        else:
            post_Y = posterior(Y, data=data, T_array=T_array)
            posterior_cache[Y] = post_Y
        
        post_current = posterior_cache[B[-1]]
        
        #acceptance probability from Metropolis-Hastings method
        prob_ratio = post_Y / post_current
        acceptance_prob = min(1, prob_ratio)
        
        U = np.random.uniform()  
        
        #acceptance => B_i=Y_i
        if U < acceptance_prob:
            B.append(Y) 
        #rejection => B_i=B_(i-1)
        else:
            B.append(B[-1])
        
        #I added this as a sanity check because it took so long to run
        #so I did not know if it was working correctly
        #Sorry if it is a bit annoying when you run my code :)
        if (i+1) % 10000 == 0:  # print every 1000 iterations
            print(f"Iter ({(i+1)/RWMH_samples*100:.1f}%)")
    
    #create dataframe for easier malnipulation 
    df = pd.DataFrame({
        "iteration": np.arange(0, RWMH_samples + 1),
        "B": B
    })
    
    return df

B_df = RWMH(beta = 0.1, B_0 = 1.5, RWMH_samples = 100000, 
            data=data, T_array=T_array)

#function to plot RWMH
def plot_RWMH(B_df):
    
    plt.figure(figsize=(20,8))
    plt.scatter(B_df['iteration'], B_df['B'], s=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('B')
    plt.title('Random Walk Metropolis-Hastings Method')
    plt.show()
    
plot_RWMH(B_df)

#function to remove certain number of burn-in samples
def remove_burn_in(B_df, N_remove):
    
    B_df_new = B_df.iloc[N_remove:].reset_index(drop=True)
    
    return B_df_new

#function to plot histogram against posterior
def hist_vs_posterior(B_df, B_grid, posterior):
    
    plt.figure(figsize=(12,8))
    plt.hist(B_df['B'], bins=50, alpha=0.7, density=True, 
             edgecolor='black', linewidth=0.2, label ='RWMH Method')
    plt.plot(B_grid, posterior, label = 'Posterior ')
    plt.xlabel('B')
    plt.ylabel('Density')
    plt.title('RWMH Method Density Compared to Posterior of B')
    plt.show()

B_df_new = remove_burn_in(B_df, N_remove = 2000)
hist_vs_posterior(B_df=B_df_new, B_grid= B_grid, posterior=normalised_posterior_values)  

#calculate mean, variance, standard deviation and confidence interval    
sample_mean = np.mean(B_df_new['B'])
sample_variance = np.var(B_df_new['B'])
sample_std = np.std(B_df_new['B'])
lower_CI = sample_mean-(np.sqrt(20)*sample_std)
upper_CI = sample_mean+(np.sqrt(20)*sample_std)

print(f'Sample mean of samples of B: {sample_mean:.4f}')
print(f'Sample variance of samples of B: {sample_variance:.4f}')
print(f'Sample standard deviation of samples of B: {sample_std:.4f}')
print(f'95% confidence interval for B: [{lower_CI:.4f},{upper_CI:.4f}]')



























