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
_train_labels = array(mndata.train_labels).astype(numpy.float32)
train_labels = keras.utils.to_categorical(_train_labels, 10)

inputs = Input(shape=(28*28,))
hidden = Dense(20, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(hidden)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(train_images, train_labels, epochs=20, batch_size=20)

function = K.function([model.input], [model.layers[-1].output])
y = function([train_images])[0]
predicted_labels = array(list(map(numpy.argmax, y)))
percent_correct = (predicted_labels == _train_labels).sum() / _train_labels.size
print(percent_correct)
