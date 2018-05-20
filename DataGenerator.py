import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def generator(subjects, batch_size):
    # intialize tracking and saving items
    threat_zone_examples = []
    start_time = timer()
    labels = get_subject_labels()
    print_shape = True
    while True:
        for i in range(0, len(subjects), batch_size):
            y_batch = []
            features = {}
            for j in range(0, 16):
                features[str(j)] = []
            for subject in subjects[i:i+batch_size]:
                images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')
                # transpose so that the slice is the first dimension shape(16, 620, 512)
                images = images.transpose()
                for j in range(0, 16):
                    fake_rgb = np.array([images[j], images[j], images[j]])
                    image = fake_rgb.transpose()
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
