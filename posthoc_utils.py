from keras import backend as K

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
