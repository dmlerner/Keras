from keras.preprocessing.image import ImageDataGenerator
import keras
from numpy import array, argmax, apply_along_axis
import keras.utils
import numpy
from keras.models import Model
from keras.layers import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import backend as K

K.set_learning_phase(1)
root = '/media/david/1600E62300E60997/ILSVRC/cropped/'
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
#    zca_whitening=True,
    horizontal_flip=True,
)
flow = generator.flow_from_directory(root, class_mode='categorical', batch_size=100)

inputs = Input(shape=(256,256,3))

conv1 = Conv2D(20, (7,7))(inputs)
mp1 = MaxPooling2D(2)(conv1)
act1 = Activation('relu')(mp1)
dropout = Dropout(.2)(act1)
conv2 = Conv2D(40, (5,5))(dropout)
mp2 = MaxPooling2D(2)(conv2)
act2 = Activation('relu')(mp2)

flat = Flatten()(act2)

d1 = Dense(500, activation='relu', use_bias=True)(flat)
d2 = Dense(200, activation='sigmoid', use_bias=True)(d1)
outputs = Dense(569, activation='sigmoid', use_bias=True)(d2)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
function = K.function([model.input]+[K.learning_phase()], [model.layers[-1].output])

def percent_correct(n):
    batch_size = 100
    samples = []
    for batch in range(n//batch_size):
        images, labels = flow.next()
        while images.shape[0] < batch_size:
            image, label = flow.next()
            images = numpy.vstack((images, image))
            labels = numpy.vstack((labels, label))
        test_images, test_labels = array(images[:n]), array(labels[:n])
        test_labels = apply_along_axis(argmax, 1, test_labels)
        predictions = function([test_images])[0]
        predicted_labels = apply_along_axis(argmax, 1, predictions)
        percent_correct = (predicted_labels == test_labels).sum() / test_labels.size
        samples.append(percent_correct)
    return sum(samples) / len(samples)

for epoch in range(10000):
    if epoch % 5 == 0:
        print('    ', epoch, percent_correct(500))
    model.fit_generator(flow, steps_per_epoch=10)
