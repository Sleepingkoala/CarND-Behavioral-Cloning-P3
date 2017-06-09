import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

import helper

tf.python.control_flow_ops = tf

number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation = 'relu'

# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
# data normalization layer 1. output = 64x64x3.
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(64, 64, 3)))

# starts with five convolutional and maxpooling layers
# convolutional layer 1. input = 64x64x3, output = 31x31x24.
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
#layer 2. input = 31x31x24, output = 15x15x36.
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# layer 3. input = 15x15x36, output = 7x7x48.
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# layer 4. input = 7x7x48, output = 6x6x64.
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# layer 5. input = 6x6x64, output = 5x5x64.
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# layer 6. input = 5x5x64, output = 1600.
model.add(Flatten())

# Next, five fully connected layers
# layer 7. output = 1164.
model.add(Dense(1164))
model.add(Activation(activation))
model.add(Dropout(0.5))
# layer 8. output = 100.
model.add(Dense(100))
model.add(Activation(activation))
model.add(Dropout(0.5))
# layer 9. output = 50.
model.add(Dense(50))
model.add(Activation(activation))
# layer 10. output = 10.
model.add(Dense(10))
model.add(Activation(activation))
# layer 11. output = 1.
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

# create two generators for training and validation
train_gen = helper.generate_next_batch()
validation_gen = helper.generate_next_batch()

history = model.fit_generator(train_gen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

# finally save our model and weights
helper.save_model(model)
