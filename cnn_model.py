from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Dropout, Flatten, Input, merge, \
                         Conv2D, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import exploratory as tsa
import keras
import pdb
from skimage.transform import resize

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

def generator2(subjects, label_path, batch_size):
    # intialize tracking and saving items
    threat_zone_examples = []
    labels = get_subject_labels(label_path)
    print_shape = True
    subjects = [subject.split('.')[0] for subject in subjects]
    while True:
        for i in range(0, len(subjects), batch_size):
            y_batch = []
            features = {}
            for j in range(0, 16):
                features[str(j)] = []
            for subject in subjects[i:i+batch_size]:
                images = tsa.read_data('/Users/michaeladdonisio/Documents/Galvanize/case-studies/tsa_screen/data/train' + '/' + subject + '.aps')
                # transpose so that the slice is the first dimension shape(16, 620, 512)
                images = images.transpose()
                for j in range(0, 16):
                    image = resize(images[j], (82, 64))
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
                threat_list = labels.loc[labels['Subject'] == subject.split(".")[0]]
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


def generator(src, label_path, batch_size):
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
                    image = resize(images[j], (155, 128))
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

def plot_loss_acc():
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args["plot"])

def plot_loss_acc2(hist):
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def train_binary_net():

    batch_size = 1

    get_train_test_file_list()
    test_subjects =TEST_SUBJECT_LIST# get_relevant_subjects(TEST_SUBJECT_LIST, THREAT_ZONE)
    train_subjects =TRAIN_SUBJECT_LIST# get_relevant_subjects(TRAIN_SUBJECT_LIST, THREAT_ZONE)

    test_gen = generator(test_subjects, batch_size)
    train_gen = generator(train_subjects, batch_size)

    #test_gen = generator(TEST_SUBJECT_LIST, batch_size)
    #train_gen = generator(TRAIN_SUBJECT_LIST, batch_size)
    print("train_gen info:")
    print(np.array(next(train_gen)[1]).shape)
    print(next(train_gen)[1])
    print ("data lengths:")
    print (len(train_subjects))
    print (len(test_subjects))


    train_steps = np.ceil(float(len(train_subjects)) / float(batch_size))
    test_steps = np.ceil(float(len(test_subjects)) / float(batch_size))

    LR = 0.01

    model = VGG_16()#MVCNN()
    #model.load_weights(MVCNN_PATH)

    print ("LR is " + str(LR))
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer= keras.optimizers.AdamAccum(lr=LR, accumulator=32.0), metrics=['acc'])
    mvcnn_checkpoint = ModelCheckpoint(MVCNN_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit_generator(generator=train_gen, validation_data=test_gen, steps_per_epoch = train_steps, validation_steps = test_steps,
        epochs = 1000, verbose=2, callbacks=[mvcnn_checkpoint])

    """for i in range(0, 15):

        im, label = next(train_gen)

        print("imshape, label:")
        print (im.shape)
        print (label)

    print ("lengths:")
    print(len(test_subjects))
    print(len(train_subjects))
    # get train and test batches
    get_train_test_file_list()
    features, labels = get_dataset(TRAIN_SET_FILE_LIST, PREPROCESSED_DATA_FOLDER)
    val_features, val_labels = get_dataset(TEST_SET_FILE_LIST, PREPROCESSED_DATA_FOLDER)
    features = features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 3)
    labels = labels.reshape(-1, 2)
    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 3)
    val_labels = val_labels.reshape(-1, 2)
    print (features[0].shape)
    # instantiate model
    #model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    log = model.fit(x=features, y=labels, batch_size=1, epochs=10000, verbose=2, validation_data=(val_features, val_labels))
    """
    return

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def MVCNN_small(weights_path=None):

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
        new_input = Input((155, 128, 1), name=str(i)) #make new input
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
    model = Dense(32, activation='relu', kernel_initializer='glorot_normal')(vp)
    #model = Dropout(0.2)(model)
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

def MVCNN(weights_path=None):

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
        new_input = Input((155, 128, 1), name=str(i)) #make new input
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
            pdb.set_trace()
            for j in range(0, 16):
                features[str(j)] = np.array(features[str(j)])
                #features2.append(np.array(features[j]))

            yield features


def predict_model(model, test_path):
    test_gen = test_generator(test_path)
    predictions = []
    predictions.append(model.predict_generator(test_gen,steps=1))
    return predictions

def model_eval(model, train_path, val_path, label_path, batch_size, name=None):
    train_files = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
    val_files = [f for f in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, f))]

    train_steps = np.ceil(float(len(train_files)) / float(batch_size))

    train_gen = generator(train_path, label_path, batch_size)
    val_gen = generator(val_path, label_path, batch_size)
    val_steps = np.ceil(float(len(val_files)) / float(batch_size))
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
    history = model.fit_generator(generator=train_gen, validation_data=val_gen, steps_per_epoch = train_steps, validation_steps = val_steps,epochs = 50, verbose=2, callbacks=[checkpointer])
    if not None:
        model.save('models/' + name + '.h5')
    return model, history



def main():
    model = MVCNN()
    current_path = os.path.dirname(os.path.realpath(__file__))
    label_path = os.path.join(current_path, 'data/stage1_labels.csv')
    train_path = os.path.join(current_path, 'data/train')
    val_path = os.path.join(current_path, 'data/val')
    return model_eval(model, train_path, val_path, label_path, batch_size=4, name='test')

if __name__ == '__main__':
    model, history = main()
