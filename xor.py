from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

inputs = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
outputs = np.array([0,1,1,0]).astype('float32')

xor = Sequential()
xor.add(Dense(4, input_dim=2))
xor.add(Activation("relu"))
xor.add(Dense(1))
xor.add(Activation("sigmoid"))

xor.compile(optimizer='rmsprop', loss='mse')

xor.fit(inputs, outputs, nb_epoch=10000)
print "Training done"
print xor.predict(inputs)
