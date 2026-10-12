import numpy as np
import matplotlib.pyplot as plt


# Approximates the solution to the SDE 
# dX(t) = -V'(X(t)) dt + sigma dW_{t}, 
# X(0) = X_{0}
# using Euler-Marauyama

# This is the stochastic double-well potential problem, and is described by Higham and Kloeden in 5.3.10

# The problem has wells at x=0 and x=2, and an unstable maximum point at x=1
# play around with the initial condition and sigma to see different behavior
X0 = 0.9

# diffusion coefficient. At sigma=0, this is just the double well potential problem
sigma = 0.3

# potential function
def V(x):
    # V(x) = x^2*(2-x)^2 = 4x^2 -4x^3 + x^4
    v = np.pow(x,2)*np.pow(2-x, 2)
    return float(v)

# and its derivative
def Vprime(x):
    # V'(x) =8x - 12x^2 + 4x^3
    vp = 8*x - 12*np.pow(x, 2) + 4*np.pow(x, 3)
    return float(vp)


# solving the stochastic DE
# dX(t) = f(X(t)) dt + g(X(t)) dW_{t}
# X(0) = X0
def f(x):
    return -Vprime(x)

def g(x):
    #return sigma*max((abs(x), abs(x-2)))
    return sigma

T = 5.0
N = 5000
delta = T/N

num_simulations = 6
X = np.zeros((num_simulations, N))
X[:,0] = X[:,0]+X0

# generate Brownian increments
dW = np.random.normal(0, np.sqrt(delta), (num_simulations, N))

t = np.linspace(0,T, N)

# Euler-Maruyama
for k in range(num_simulations):
    for i in range(N-1):
        X[k, i+1] = X[k, i] + f(X[k, i]) * delta + g(X[k,i]) * dW[k,i]
        #X[k, i+1] = X[k, i] + f(X[k, i]) * delta + g(X[k,i])*abs(1-V(X[k,i])) * dW[k,i]

# generate Brownian motion (optional) 
"""
W = np.zeros((num_simulations, N))
for k in range(num_simulations):
    W[k,] = np.append([0], np.cumsum(dW[k,:-1]))
"""

# plot
well1 = np.zeros(N)
center = np.zeros(N)+1.0
well2 = np.zeros(N)+2.0

fig, axs = plt.subplots()

for k in range(num_simulations):
    axs.plot(t, X[k,], label='Sample %d'%k)

axs.plot(t, well1, label="_Well 1", color="black")
axs.plot(t, center, label="_Unstable Center", linestyle = "--", color="black")
axs.plot(t, well2, label="_Well 2", color="black")
axs.set_ylim(-0.5, 2.5)

axs.set_title("Solution with $X_{0}$ = %1.1f and $\\sigma = $%1.1f Using Euler-Marauyama T=%d, N=%d"%(X0, sigma, T, N))
axs.legend()
plt.show()