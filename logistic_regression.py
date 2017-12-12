import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
def prediction(X, W, b):
    return sigmoid(np.matmul(X,W)+b)
def error_vector(y, y_hat):
    return [-y[i]*np.log(y_hat[i]) - (1-y[i])*np.log(1-y_hat[i]) for i in range(len(y))]
def error(y, y_hat):
    ev = error_vector(y, y_hat)
    return sum(ev)/len(ev)

def dErrors(X, y, y_hat):
    dErrs = []
    yDiff = [(y_hat[i] - y[i]) for i in range(len(y))]
    # With respect to all wi
    for j in range(len(X[0])):
        dErrs.append([X[i][j] * yDiff[i]  for i in range(len(y))])
    # with respect to b
    dErrs.append(yDiff)
    return dErrs

def gradientDescentStep(X, y, W, b, learn_rate = 0.01):
    y_hat = prediction(X,W,b)
    e = sum(error_vector(y, y_hat))
    gradient = dErrors(X, y, y_hat)
    W = [W[i]-sum(gradient[i]) * learn_rate for i in range(len(W))]
    b -= sum(gradient[2])*learn_rate
    return W, b, e

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainLR(X, y, learn_rate = 0.01, num_epochs = 100):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    # Initialize the weights randomly
    W = np.array(np.random.rand(2,1))*2 -1
    b = np.random.rand(1)[0]*2 - 1
    # These are the solution lines that get plotted below.
    boundary_lines = []
    errors = []
    for i in range(num_epochs):
        # In each epoch, we apply the gradient descent step.
        W, b, error = gradientDescentStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        errors.append(error)
    return boundary_lines, errors
