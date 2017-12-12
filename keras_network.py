import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from keras.layers import Dense, Dropout
# Using TensorFlow 1.0.0; use tf.python_io in later versions
tf.python.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Building the model
xor = Sequential()

# Add required layers
xor.add(Dense(32, activation='tanh', input_dim=X.shape[1]))
xor.add(Dropout(0.2))
xor.add(Dense(64, activation='relu'))
xor.add(Dropout(0.2))
xor.add(Dense(1, activation='sigmoid'))

xor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Uncomment this line to print the model architecture
xor.summary()

# Fitting the model
history = xor.fit(X, y, nb_epoch=50, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))
