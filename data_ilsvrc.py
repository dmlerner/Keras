import keras
from keras.preprocessing.image import ImageDataGenerator
from numpy import array, argmax, apply_along_axis
import keras.utils
import numpy
from keras.models import Model
from keras.layers import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import *
from keras import backend as K
from pylab import plot, ion, draw, pause, subplot

K.set_learning_phase(1)
root = '/media/david/1600E62300E60997/ILSVRC/cropped/'
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
#    zca_whitening=True,
    horizontal_flip=True,
)
flow_train = generator.flow_from_directory(root+'train', class_mode='categorical', batch_size=100)
flow_test = generator.flow_from_directory(root+'test', class_mode='categorical', batch_size=100)

inputs = Input(shape=(256,256,3))

conv1 = Conv2D(40, (7,7))(inputs)
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
sgd = optimizers.SGD(lr=.01, decay=1e-6, momentum=.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
function = K.function([model.input]+[K.learning_phase()], [model.layers[-1].output])

def percent_correct(flow, n):
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

p_train = []
p_test = []
train_loss, test_loss = [], []
skip = 1
ion()
for epoch in range(100):
    if epoch and epoch % skip == 0:
        p_train.append(percent_correct(flow_train, 1000))
        p_test.append(percent_correct(flow_test, 1000))
        t = numpy.arange(epoch)
        subplot(121, axisbg='black')
        plot(t[::skip], p_train, 'r.')
        plot(t[::skip], p_test, 'b.')
        subplot(122, axisbg='black')
        plot(t, train_loss, 'r.')
        plot(t, test_loss, 'b.')
        print(epoch, p_train[-1], p_test[-1])
        pause(.01)

    history = model.fit_generator(flow_train, validation_data=flow_test, steps_per_epoch=10, validation_steps=10)
    train_loss.append(history.history['loss'])
    test_loss.append(history.history['val_loss'])
