import keras
from keras.models import Model
from keras.layers import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import backend as K
from mnist import MNIST
import numpy
from numpy import array
import tensorflow

mndata = MNIST('python-mnist/data/')
mndata.load_training()
mndata.load_testing()

test_images = array(mndata.test_images).reshape((10000,1,28,28)).astype(numpy.float32) / 255 - .5
_test_labels = array(mndata.test_labels).astype(numpy.float32)
test_labels = keras.utils.to_categorical(_test_labels, 10)

train_images = array(mndata.train_images).reshape((60000,1,28,28)).astype(numpy.float32) / 255 - .5
_train_labels = array(mndata.train_labels).astype(numpy.float32)
train_labels = keras.utils.to_categorical(_train_labels, 10)

inputs = Input(shape=(1,28,28))
convolution = Conv2D(20, (7,7), data_format='channels_first')(inputs)
maxpool = MaxPooling2D(2, data_format='channels_first')(convolution)
flat = Flatten()(maxpool)
x = Dense(100, activation='sigmoid', use_bias=True)(flat)
outputs = Dense(10, activation='sigmoid', use_bias=True)(x)

model = Model(inputs=inputs, outputs=outputs)
#K.set_learning_phase(1)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(train_images, train_labels, epochs=100, batch_size=300)

function = K.function([model.input]+[K.learning_phase()], [model.layers[-1].output])
train_predictions = function([train_images])[0]
test_predictions = function([test_images])[0]

predicted_labels_train = array(list(map(numpy.argmax, train_predictions))).astype(numpy.float32)
predicted_labels_test = array(list(map(numpy.argmax, test_predictions))).astype(numpy.float32)
percent_correct_train = (predicted_labels_train == _train_labels).sum() / _train_labels.size
percent_correct_test = (predicted_labels_test == _test_labels).sum() / _test_labels.size

print(percent_correct_train, percent_correct_test)
