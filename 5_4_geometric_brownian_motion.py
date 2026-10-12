import numpy as np
import matplotlib.pyplot as plt


# Approximates the solution to the SDE 
# dX_{t} = f(X_{t})dt + g(X_{t})dW_{t}
# using Euler-Marauyama

mu = 33./32
sigma = 1./4
def f(x):
    return mu*x

def g(x):
    return sigma*x

# this particular choice of f(x)=mu*x and g(x)=sigma*x is called a linear, multiplicative-noise SDE. 
# It has a strong solution for any mu and sigma and Brownian motion sample W(t), given by 
# X(t) = X(0) * exp( (mu-sigma^2/2)*t + sigma*W(t) )
# sometimes called geometric brownian motion

T = 1.0
N = 1000
delta = T/N

num_simulations = 6

X = np.zeros((num_simulations, N))
X[:,0] = X[:,0]+1

dW = np.random.normal(0, np.sqrt(delta), (num_simulations, N))
for k in range(num_simulations):
    for i in range(N-1):
        X[k, i+1] = X[k, i] + f(X[k, i]) * delta + g(X[k,i]) * dW[k,i]

W = np.zeros((num_simulations, N))
for k in range(num_simulations):
    W[k,] = np.append([0], np.cumsum(dW[k,:-1]))

t = np.linspace(0,T, N)
X_exact = np.zeros((num_simulations, N))
for k in range(num_simulations):
    X_exact[k,] = np.exp(t+W[k,]/4)

ev_X = np.exp(33*t/32)



fig, axs = plt.subplots(1,2)

for k in range(num_simulations):
    axs[0].plot(t, X[k,], label='Sample %d'%k)
    axs[1].plot(t, X_exact[k,], label='Sample %d'%k)
    #axs[0].plot(t, W[k,], label='Sample %d'%k)

#axs[0].set_title("Discrete Brownian Paths")
#axs[0].legend()

axs[0].plot(t, ev_X, color='black', linestyle='dashed', label='Expected Value')
axs[0].set_title("Solution Using Euler-Marauyama T=1, N=%d"%N)
axs[0].legend()

axs[1].plot(t, ev_X, color='black', linestyle='dashed', label='Expected Value')
axs[1].set_title("Exact Solution for 4 Brownian Paths T=1, N=%d"%N)
axs[1].legend()
plt.show()
