import numpy
import sys
import tensorflow as tf
from make_parallel import *
import keras
from math import log10
from keras.preprocessing.image import ImageDataGenerator
from numpy import array, argmax, apply_along_axis
from keras.regularizers import *
import keras.utils
import numpy
from keras.models import Model
from keras.layers import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import *
from keras import backend as K
from pylab import * 
from keras.layers.normalization import BatchNormalization
import itertools
from utils import *

gpu_number =  sys.argv[1] if len(sys.argv)>1 else '0'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
K.set_learning_phase(1)
root = '/media/david/1600E62300E60997/ILSVRC/cropped/'
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    #zca_whitening=True,
    horizontal_flip=True,
    rotation_range=10,
    shear_range=3.14*.05,
    vertical_flip=True,
)
def sample(flow, n):
    images, labels = flow.next()
    while images.shape[0] < n:
        image, label = flow.next()
        images = numpy.vstack((images, image))
        labels = numpy.vstack((labels, label))

    images, labels = array(images[:n]), array(labels[:n])
    labels = apply_along_axis(argmax, 1, labels)
    return images, labels

#def sample(flow, n):
    #return list(itertools.islice(flow, n))
    #does flow basically have this built in?

batch_size = 50
flow_train = generator.flow_from_directory(root+'train', class_mode='categorical', batch_size=batch_size)
flow_test = generator.flow_from_directory(root+'test', class_mode='categorical', batch_size=batch_size)

#fit_images = sample(flow_train, 1000)[0]
#generator.fit(fit_images)
#generator.zca_whitening = True



def conv_suite(inputs, n, size):
  conv = Conv2D(n, (size,)*2)(inputs)
  batch = BatchNormalization(axis=1)(conv)
  pool = MaxPooling2D((4,4))(batch)
  activation = Activation('relu')(pool)
  return activation
  
def dense_suite(inputs, n, activation='relu'):
  dense = Dense(n, activation=activation, use_bias=True)(inputs)
  return BatchNormalization()(dense)

inputs = Input(shape=(256,256,3))
c1 = conv_suite(inputs, 20, 3)
c2 = conv_suite(c1, 40, 3)
c3 = conv_suite(c2, 40, 3)
flat = Flatten()(c3)
d1 = dense_suite(flat, 2000)
d2 = dense_suite(d1, 1000)
d3 = dense_suite(d2, 500)
d4 = dense_suite(d3, 250)
outputs = Dense(569, activation='sigmoid', use_bias=True)(d4)

import sys
model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
model = make_parallel(model, int(sys.argv[1]) if len(sys.argv)>1 else 1)
model.compile(optimizer=optimizers.nadam(), loss='categorical_crossentropy')
function = K.function([model.input]+[K.learning_phase()], [model.layers[-1].output])
def percent_correct(flow, n):
  batch_size = 100
  samples = []
  for batch in range(n//batch_size):
    images, labels = sample(flow, batch_size)
    predictions = function([images])[0]
    predicted_labels = apply_along_axis(argmax, 1, predictions)
    percent_correct = (predicted_labels == labels).sum() / labels.size
    samples.append(percent_correct)
  return sum(samples) / len(samples)

p_train = []
p_test = []
train_loss, test_loss = [], []
skip = 1
ion()
for epoch in range(10000):
  t = numpy.arange(epoch)
  if epoch and epoch % skip == 0:
    clf()
    p_train.append(percent_correct(flow_train, 100))
    p_test.append(percent_correct(flow_test, 100))
    subplot(121, axisbg='black')
    plot(t[::skip], p_test, 'w.')
    plot(t[::skip], p_train, 'r.')
    subplot(122, axisbg='black')
    plot(t, array(test_loss), 'w.', markersize=10)
    plot(t, array(train_loss), 'r.', markersize=10)
    plot(t, ewma(test_loss), 'w-')  
    plot(t, ewma(train_loss), 'r-')  
    print('.'*15, epoch, p_train[-1], p_test[-1])
    pause(.01)
    model.save('keras.sav')
  history = model.fit_generator(flow_train, validation_data=flow_test, steps_per_epoch=1, validation_steps=1)
  train_loss.append(history.history['loss'][0])
  test_loss.append(history.history['val_loss'][0])

