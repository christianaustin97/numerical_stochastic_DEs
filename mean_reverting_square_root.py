import numpy as np
import matplotlib.pyplot as plt

# Approximates the solution to the SDE 
# dX(t) = r(mu-X(t)) dt + sigma*sqrt(X(t)) dW_{t}, 
# X(0) = X_{0}
# using Euler-Marauyama

# Solving the mean reverting squrae root process
# The problem has mean at mu, and should always ``come back home'
# For X0>0, the process will always stay positive since the 
#   variance terms become small when X approaches 0, provided
#   2*r*mu > sigma^2

# play around with the initial condition to see different behavior
X0 = 0.9

r = 1.0
mu = 3.0
sigma = 0.25


# solving the stochastic DE
# dX(t) = f(X(t)) dt + g(X(t)) dW_{t}
# X(0) = X0
def f(x):
    return r*(mu-x)

def g(x):
    #return float(sigma*np.pow(abs(x), 0.5))
    return float(sigma*np.pow(x, 0.5))

T = 10.0
N = 10000
delta = T/N

num_simulations = 6
X = np.zeros((num_simulations, N))
X[:,0] = X[:,0]+X0

# generate Brownian increments
dW = np.random.normal(0, np.sqrt(delta), (num_simulations, N))

# Euler-Maruyama
for k in range(num_simulations):
    for i in range(N-1):
        X[k, i+1] = X[k, i] + f(X[k, i]) * delta + g(X[k,i]) * dW[k,i]

# generate Brownian motion (optional, if you have a closed form for exact solution) 
"""
W = np.zeros((num_simulations, N))
for k in range(num_simulations):
    W[k,] = np.append([0], np.cumsum(dW[k,:-1]))
"""

# plot
t = np.linspace(0,T, N)
mean_line = np.zeros(N)+mu
expected_X = X0*np.exp(-r*t) + mu*(1-np.exp(-r*t))

fig, axs = plt.subplots()

for k in range(num_simulations):
    axs.plot(t, X[k,], label='Sample %d'%k)

axs.plot(t, expected_X, label='Expected Value', color='black')

#axs.plot(t, mean_line, label="_Mean Value", color="black")

axs.set_title("Solution with $X_{0}$ = %1.1f and $\\sigma = $%1.1f Using Euler-Marauyama T=%d, N=%d"%(X0, sigma, T, N))
axs.legend()
plt.show()




