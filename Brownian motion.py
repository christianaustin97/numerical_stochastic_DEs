import numpy as np
import matplotlib.pyplot as plt

T = 1.0
N = 1000
delta = T/N

W = np.zeros((4,N))
for k in range(4):
    for i in range(N-1):
        increment = np.random.normal(0, np.sqrt(delta))
        W[k, i+1] = W[k, i] + increment

t = np.linspace(0,T, N)

fig, axs = plt.subplots(2,2)
axs[0,0].plot(t, W[0,])
axs[0,0].set_title("Sample 1")
axs[0,1].plot(t, W[1,], color='orange')
axs[0,1].set_title("Sample 2")
axs[1,0].plot(t, W[2,], color='green')
axs[1,0].set_title("Sample 3")
axs[1,1].plot(t, W[3,], color='red')
axs[1,1].set_title("Sample 4")

fig = plt.gcf()
fig.suptitle("Samples of Discretized Brownian Motion, T=1, N=1000")
plt.show()

E = np.zeros((4,N))
E[0,] = np.exp(t+W[0,]/4)
E[1,] = np.exp(t+W[1,]/4)
E[2,] = np.exp(t+W[2,]/4)
E[3,] = np.exp(t+W[3,]/4)
ev_E = np.exp(33*t/32)

plt.plot(t, E[0,], label='Sample 1')
plt.plot(t, E[1,], color='orange', label='Sample 2')
plt.plot(t, E[2,], color='green', label='Sample 3')
plt.plot(t, E[3,], color='red', label='Sample 4')
plt.plot(t, ev_E, color='black', linestyle='dashed', label='Expected Value')
fig = plt.gcf()
fig.suptitle("Samples of exp(t+W(t)/4), T=1, N=1000")
fig.legend()
plt.show()

