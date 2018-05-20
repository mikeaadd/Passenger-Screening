import keras.applications.resnet as rn
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, \
                         Flatten, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

def MVCNN():
    resnet = rn.ResNet50()

    model = Sequential()
    model.add(Convolution2D(512, kernel_size=(1,1), stride=1, padding='valid', dilation_rate=1, use_bias=True, activation='relu', input_shape=(1,28,28)))
    mo
    model.add(Convolution2D(32, 3, 3, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='softmax'))


# Basic Example
input_img = Input(shape=(256, 256, 3))

# Define 3 Conv Layers
result = []
for i in rnage(n_views):
    # must get single view and input
    view =
    input

    conv_1 = Convolution2D(512, kernel_size=(1,1), stride=1, padding='valid', dilation_rate=1, use_bias=True, activation='relu')(input_img)
    pool_1 = _maxpool('pool1', conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')(conv_1)
    conv_2 = Convolution2D(256, kernel_size=(3,3), stride=2, padding='same', dilation_rate=1, use_bias=True, activation='relu')(pool_1)
    pool_2 = _maxpool('pool1', conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')(conv_2
    conv_3 = Convolution2D(128, kernel_size=(5,5), stride=3, padding='same', dilation_rate=1, use_bias=True, activation='relu')(pool_2)

# Merge them using for example mode='sum'
merged = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)
merged = Flatten()(merged)

# Add some Dense Layers
out = Dense(1, activation='sigmoid')(merged)

# Create the Model
some_model = Model(input_img, out)
some_model.summary()
