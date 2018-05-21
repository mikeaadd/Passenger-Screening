from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Dropout, Flatten, Input, merge, \
                         Conv2D, Concatenate, LSTM, Conv1D, Reshape, Permute
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D, AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import exploratory as tsa
import keras
import pdb
from skimage.transform import resize
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import cv2


def cnn_model():
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(17, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)
    return model

def get_subject_labels(path):
    df = pd.read_csv(path)
    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    #TODO: convert zone to correct int here
    df = df[['Subject', 'Zone', 'Probability']]
    return df

def AlexNet(weights_path=None):
    inputs = Input(shape=(3, 227, 227))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Dense(17, activation='sigmoid', kernel_initializer='glorot_normal')(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)

    return model

def generator3channel(src, label_path, input_size=(155,128), batch_size=1):
    # intialize tracking and saving items
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    threat_zone_examples = []
    labels = get_subject_labels(label_path)
    print_shape = True
    while True:
        for i in range(0, len(files), batch_size):
            y_batch = []
            features = {}
            for j in range(0, 16):
                features[str(j)] = []
            for file in files[i:i+batch_size]:
                images = tsa.read_data(os.path.join(src, file))
                # transpose so that the slice is the first dimension shape(16, 620, 512)
                images = images.transpose()
                for j in range(0, 16):
                    image = resize(images[j], input_size)
                    image3 = np.array([images[j], images[j], images[j]])
                    image3 = image3.transpose()
                    # image3 = image3[:,:,:,np.newaxis]
                    """if print_shape:
                        print ("Shape of re-transposed image:")
                        print (image.shape)
                        print_shape = False
                    """
                    #resized_image = images[j] #scipy.ndimage.zoom(images[j], (0.5, 0.5))
                    #features[str(j)].append(np.reshape(resized_image, (660, 512, 1)))# (330, 256, 1)))
                    features[str(j)].append(image3)# (330, 256, 1)))

                # get label
                y = np.zeros((17))
                threat_list = labels.loc[labels['Subject'] == file.split(".")[0]]
                threat_iter = threat_list.iterrows()
                while True:
                    threat = next(threat_iter, None)
                    if threat is None:
                        break
                    threat = threat[1]
                    if threat['Probability'] is 1:
                        zone = threat['Zone']
                        zone = int(zone[4:])
                        y[zone-1] = 1
                """
                y = np.array(tsa.get_subject_zone_label(THREAT_ZONE,
                                 tsa.get_subject_labels(STAGE1_LABELS, subject)))
                np.reshape(y, (2, 1))
                """
                y_batch.append(y)

            for j in range(0, 16):
                features[str(j)] = np.array(features[str(j)])
                #features2.append(np.array(features[j]))

            yield features, np.array(y_batch)


def generator(src, label_path, input_size=(155,128), batch_size=1):
    # intialize tracking and saving items
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    threat_zone_examples = []
    labels = get_subject_labels(label_path)
    print_shape = True
    while True:
        for i in range(0, len(files), batch_size):
            y_batch = []
            features = {}
            for j in range(0, 16):
                features[str(j)] = []
            for file in files[i:i+batch_size]:
                images = tsa.read_data(os.path.join(src, file))
                # transpose so that the slice is the first dimension shape(16, 620, 512)
                images = images.transpose()
                for j in range(0, 16):
                    image = resize(images[j], input_size)
                    image = image[:,:,np.newaxis]
                    """if print_shape:
                        print ("Shape of re-transposed image:")
                        print (image.shape)
                        print_shape = False
                    """
                    #resized_image = images[j] #scipy.ndimage.zoom(images[j], (0.5, 0.5))
                    #features[str(j)].append(np.reshape(resized_image, (660, 512, 1)))# (330, 256, 1)))
                    features[str(j)].append(image)# (330, 256, 1)))

                # get label
                y = np.zeros((17))
                threat_list = labels.loc[labels['Subject'] == file.split(".")[0]]
                threat_iter = threat_list.iterrows()
                while True:
                    threat = next(threat_iter, None)
                    if threat is None:
                        break
                    threat = threat[1]
                    if threat['Probability'] is 1:
                        zone = threat['Zone']
                        zone = int(zone[4:])
                        y[zone-1] = 1
                """
                y = np.array(tsa.get_subject_zone_label(THREAT_ZONE,
                                 tsa.get_subject_labels(STAGE1_LABELS, subject)))
                np.reshape(y, (2, 1))
                """
                y_batch.append(y)

            for j in range(0, 16):
                features[str(j)] = np.array(features[str(j)])
                #features2.append(np.array(features[j]))

            yield features, np.array(y_batch)


def MVCNN_small(input_size=(155,128),weights_path=None):

    inputs = []
    view_pool = []

    """
    new_input = Input((660, 512, 1)) #make new input
    #new_input = Input((330, 256, 1)) #make new input
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(new_input)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Flatten()(x)
    cnn1 = Model(inputs = new_input, outputs = x)
    """
    for i in range(0, 16):
        new_input = Input(input_size+(1,), name=str(i)) #make new input
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(new_input)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Flatten()(x)
        inputs.append(new_input)
        view_pool.append(x)
        """
        cnn1 = Model(inputs = new_input, outputs = x)
        new_input = Input((660, 512, 1), name=str(i)) #make new input
        #new_input = Input((330, 256, 1), name=str(i)) #make new input
        new_model = cnn1(new_input)
        inputs.append(new_input)
        view_pool.append(new_model)
        """
    vp = Concatenate(axis=0)(view_pool) #tf.concat([vp, v], 0)
    model = Dense(34, activation='relu', kernel_initializer='glorot_normal')(vp)
    #model = Dropout(0.2)(model)
    #model = Dropout(0.2)(model)
    #model = Flatten()(model)
    """model = Dense(2048, activation='relu')(vp)
    model = Dense(1024, activation='relu')(model)
    model = Dense(512, activation='relu')(model)
    model = Dense(256, activation='relu')(model) """
    model = Dense(17, activation='sigmoid', kernel_initializer='glorot_normal')(model)

    full_model = Model(inputs=inputs, outputs=model)

    full_model.compile(loss=keras.losses.binary_crossentropy,
            optimizer= keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    full_model.summary()

    return full_model

def MVCNN(input_size=(155,128), weights_path=None):

    inputs = []
    view_pool = []

    """
    new_input = Input((660, 512, 1)) #make new input
    #new_input = Input((330, 256, 1)) #make new input
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(new_input)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Flatten()(x)
    cnn1 = Model(inputs = new_input, outputs = x)
    """
    for i in range(0, 16):
        new_input = Input(input_size + (1,), name=str(i)) #make new input
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(new_input)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        x = Flatten()(x)
        inputs.append(new_input)
        view_pool.append(x)
        """
        cnn1 = Model(inputs = new_input, outputs = x)
        new_input = Input((660, 512, 1), name=str(i)) #make new input
        #new_input = Input((330, 256, 1), name=str(i)) #make new input
        new_model = cnn1(new_input)
        inputs.append(new_input)
        view_pool.append(new_model)
        """
    vp = Concatenate(axis=0)(view_pool) #tf.concat([vp, v], 0)
    model = Dense(512, activation='relu', kernel_initializer='glorot_normal')(vp)
    #model = Dropout(0.2)(model)
    model = Dense(512, activation='relu', kernel_initializer='glorot_normal')(model)
    #model = Dropout(0.2)(model)
    #model = Flatten()(model)
    """model = Dense(2048, activation='relu')(vp)
    model = Dense(1024, activation='relu')(model)
    model = Dense(512, activation='relu')(model)
    model = Dense(256, activation='relu')(model) """
    model = Dense(17, activation='sigmoid', kernel_initializer='glorot_normal')(model)

    full_model = Model(inputs=inputs, outputs=model)

    full_model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer= keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    full_model.summary()

    return full_model

def MVCNN_lstm(input_size=(155,128), weights_path=None):
        inputs = []
        view_pool = []

        for i in range(0, 16):
            new_input = Input(input_size + (1,), name=str(i)) #make new input
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(new_input)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Flatten()(x)
            inputs.append(new_input)
            view_pool.append(x)
        vp = Concatenate(axis=1)(view_pool) #tf.concat([vp, v], 0)
        vp = Reshape((1,-1))(vp)
        model = LSTM(4, input_shape=(None,1))(vp)
        model = Dense(17, activation='sigmoid', kernel_initializer='glorot_normal')(model)

        full_model = Model(inputs=inputs, outputs=model)

        full_model.compile(loss=keras.losses.binary_crossentropy,
                optimizer= keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

        full_model.summary()

        return full_model

def MVCNN_resnet(input_size=(155,128)):
        inputs = []
        view_pool = []
        resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
        last_layer = resnet.get_layer('avg_pool').output
        last_layer = Flatten()(last_layer)
        cnn1 = Model(inputs = resnet.input, outputs = last_layer)# = pool3)


        for i in range(0, 16):
            new_input = Input(input_size + (3,), name=str(i)) #make new input
            # resnet = ResNet50(input_tensor = new_input, weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
            # last_layer = resnet.get_layer('avg_pool').output
            # print(last_layer._keras_shape)
            # x = Flatten(name='flatten')(last_layer)
            # print(x._keras_shape)
            x = cnn1(new_input)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal')(x)
            x = MaxPooling2D((2,2), strides=(2,2))(x)
            x = Flatten()(x)
            inputs.append(new_input)
            view_pool.append(x)

        vp = Concatenate(axis=1)(view_pool) #tf.concat([vp, v], 0)
        model = Dense(17, activation='sigmoid', kernel_initializer='glorot_normal')(model)
        full_model = Model(inputs=inputs, outputs=model)

        full_model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer= keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

        full_model.summary()

        return full_model

def VGG_16(weights_path=None):

    inputs = []
    view_pool = []

    #new_input = Input((512, 660, 3)) #make new input

    #Instantiate shared CNN1 Layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.layers[10].output
    #conv1 = Conv2D(32, (3, 3), activation='relu')(x)
    #pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)
    #conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    #pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)
    #conv3 = Conv2D(32, (3, 3), activation='relu')(conv2)
    #conv4 = Conv2D(32, (3, 3), activation='relu')(conv3)
    #conv5 = Conv2D(32, (3, 3), activation='relu')(conv4)
    #pool3 = MaxPooling2D((2,2), strides=(2,2))(conv5)
    x = Flatten()(x)
    cnn1 = Model(inputs = base_model.input, outputs = x)# = pool3)

    for i in range(0, 16):
        new_input = Input((512, 660, 3), name=str(i)) #make new input
        x = cnn1(new_input)
        new_model = MaxPooling2D((2,2), strides=(2,2))(x)
        inputs.append(new_input)
        view_pool.append(new_model)

    #vp = make_view_pool(view_pool, "vp")

    for layer in base_model.layers:
        layer.trainable = False

    vp = Concatenate(axis=0)(view_pool) #tf.concat([vp, v], 0)
    model = Dense(128, activation='relu')(vp)
    model = Dropout(0.2)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.2)(model)
    print(model._keras_shape)
    # model = Flatten()(model)
    model = Dense(17, activation='sigmoid')(model)

    full_model = Model(inputs=inputs, outputs=model)

    full_model.compile(loss=keras.losses.binary_crossentropy,
            optimizer= keras.optimizers.Adam(lr=0.001), metrics=['acc'])

    base_model.summary()
    print("----")

    full_model.summary()

    return full_model

def resnet50(input_size):
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
    x = resnet_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='linear', name='fc1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid', name='prediction')(x)
    model = Model(inputs=resnet_model.inputs, outputs=x)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def test_generator(test_path):
    # intialize tracking and saving items
    files = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    while True:
        for i in range(0, len(files), 1):
            features = {}
            for j in range(0, 16):
                features[str(j)] = []
            for file in files[i:i+1]:
                images = tsa.read_data(os.path.join(test_path, file))
                # transpose so that the slice is the first dimension shape(16, 620, 512)
                images = images.transpose()
                for j in range(0, 16):
                    image = resize(images[j], (82, 64))
                    image = image[:,:,np.newaxis]
                    features[str(j)].append(image)# (330, 256, 1)))
            for j in range(0, 16):
                features[str(j)] = np.array(features[str(j)])
                #features2.append(np.array(features[j]))

            yield features


def predict_model(model, test_path):
    test_gen = test_generator(test_path)
    predictions = []
    predictions.append(model.predict_generator(test_gen,steps=1))
    return predictions

def model_eval(model, train_path, val_path, label_path, input_size, batch_size, epochs=1, name=None):
    train_files = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
    val_files = [f for f in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, f))]

    train_steps = np.ceil(float(len(train_files)) / float(batch_size))

    train_gen = generator(train_path, label_path, input_size, batch_size)
    val_gen = generator(val_path, label_path, input_size, batch_size)
    val_steps = np.ceil(float(len(val_files)) / float(batch_size))
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
    history = model.fit_generator(generator=train_gen, validation_data=val_gen, steps_per_epoch = train_steps, validation_steps = val_steps,epochs = epochs, verbose=1, callbacks=[checkpointer])
    if not None:
        model.save('models/' + name + '.h5')
    return model, history

def main():
    input_size= (224, 224)
    # input_size= (310, 256)
    # input_size= (224, 224)
    # input_size= (100,100)
    batch_size = 1
    epochs = 1
    model = MVCNN_lstm(input_size)
    current_path = os.path.dirname(os.path.realpath(__file__))
    label_path = os.path.join(current_path, 'data/stage1_labels.csv')
    train_path = os.path.join(current_path, 'data/train')
    val_path = os.path.join(current_path, 'data/val')
    return model_eval(model, train_path, val_path, label_path, input_size=input_size, batch_size=batch_size, epochs=epochs, name='test')

if __name__ == '__main__':
    model, history = main()
