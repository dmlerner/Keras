from keras.preprocessing.image import ImageDataGenerator
from numpy import array
from keras.models import Model
from keras.layers import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import backend as K

root = '/media/david/1600E62300E60997/ILSVRC/cropped/'
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    zca_whitening=True,
    horizontal_flip=True,
).flow_from_directory(root, class_mode='categorical', batch_size=10)

inputs = Input(shape=(256,256,3))
convolution = Conv2D(20, (5,5))(inputs)
maxpool = MaxPooling2D(2)(convolution)
flat = Flatten()(maxpool)
d1 = Dense(500, activation='relu', use_bias=True)(flat)
d2 = Dense(500, activation='sigmoid', use_bias=True)(d1)
outputs = Dense(569, use_bias=True)(d1)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit_generator(generator, epochs=100, steps_per_epoch=10)
#todo: implement % correct measurement
