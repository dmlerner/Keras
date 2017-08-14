import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K
from mnist import MNIST
import numpy
from numpy import array
import tensorflow

mndata = MNIST('python-mnist/data/')
mndata.load_training()
mndata.load_testing()
#test_images = array(mndata.test_images).reshape((10000,28*28))
#test_labels = keras.utils.to_categorical(array(mndata.test_labels), 10)
train_images = array(mndata.train_images).reshape((60000,28*28)).astype(numpy.float32)
train_labels = keras.utils.to_categorical(array(mndata.train_labels), 10).astype(numpy.float32)

inputs = Input(shape=(28*28,))
#outputs = Dense(10, input_shape=(28*28,))(inputs)
outputs = Dense(10)(inputs)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(train_images, train_labels)

function = K.function([model.input]+ [K.learning_phase()], [model.layers[-1].output])
x = train_images[0]
y = function([x])[0]
print(y)
print(y.shape)
print(numpy.argmax(y))
