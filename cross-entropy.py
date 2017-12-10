import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    notY = np.array(Y) - 1
    prob = np.absolute(notY + P)
    logs = -1 * np.log(prob)
    return np.sum(logs)
