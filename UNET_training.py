import numpy as np
import glob
import keras
from keras import layers
from keras.models import Sequential
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_list, target_list, batch_size=4, dim=(176, 368), n_input_channels=133,
                 n_output_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.target_list = target_list
        self.input_list = input_list
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.input_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        input_list_temp = [self.input_list[k] for k in indexes]

        target_list_temp = [self.target_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(input_list_temp, target_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.input_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, input_list_temp, target_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_input_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_output_channels))

        # Generate data
        for i, ID in enumerate(input_list_temp):
            # Store sample
            X[i,] = resize(np.load(ID), (176, 368, 133))

        for i, ID in enumerate(target_list_temp):
            # Store sample
            y[i,] = resize(np.load(ID), (176, 368, 1))
            # Store class

        return X, y


# Parameters
params = {'dim': (176, 368),
          'batch_size': 4,
          'n_output_channels': 1,
          'n_input_channels': 133,
          'shuffle': False}

# Datasets
input_list = glob.glob('/mnt/nucaps-s3/philip/UNET_Model_Data/Input/180x360_np/'+'*.npy')
target_list = glob.glob('/mnt/nucaps-s3/philip/UNET_Model_Data/Target/180x360_np/'+'*.npy')

training_input_list = input_list[0:88]
training_target_list = target_list[0:88]
val_input_list = input_list[88:91]
val_target_list = target_list[88:91]

def val_generation(val_batch_size,val_input_list, val_target_list):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    x_val = np.empty((val_batch_size, *(176, 368), 133))
    y_val = np.empty((val_batch_size, *(176, 368), 1))

    # Generate data
    for i, ID in enumerate(val_input_list):
        # Store sample
        x_val[i,] = resize(np.load(ID), (176, 368, 133))

    for i, ID in enumerate(val_target_list):
        # Store sample
        y_val[i,] = resize(np.load(ID), (176, 368, 1))
        # Store class

    return x_val, y_val

x_val, y_val = val_generation(3, val_input_list,val_target_list)

# Generators
training_generator = DataGenerator(training_input_list, training_target_list, **params)

img_size = (176, 368)

def get_model(img_size):
    inputs = keras.Input(shape=img_size + (133,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(266, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [532, 1064, 2128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [2128, 1064, 532, 266]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(1, 1, activation='linear')(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    model.compile()
    return model


# Build model
model = get_model(img_size)

model.compile(optimizer=Adam(), loss='mean_squared_error')
# model.summary()

checkpoint_filepath = '/mnt/nucaps-s3/philip/MERRA-2/checkpoint2.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(x=training_generator,
            validation_data=(x_val, y_val),
            use_multiprocessing=False,
            epochs=200,
            callbacks=[model_checkpoint_callback],
            verbose=1)

model.save('/mnt/nucaps-s3/philip/MERRA-2/UNETMODELNEW1.keras')
