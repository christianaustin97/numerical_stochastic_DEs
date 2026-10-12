import numpy as np
import matplotlib.pyplot as plt


# Approximates the solution to the SDE 
# dX_{t} = f(X_{t})dt + g(X_{t})dW_{t}
# using Euler-Marauyama

# This SDE is known as the Vasicek Interest Rate Model, described in 5.3 of Higham and Kloeden