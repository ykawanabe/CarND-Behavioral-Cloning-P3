import csv
import cv2
import shutil
import scipy.misc

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.extend([line])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import numpy as np
import sklearn
BATCH_SIZE = 1024
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                    source_path = line[0]
                    filename = source_path.split('/')[-1]
                    current_path = '../data/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = image[60:140, 0:320]
                    image = scipy.misc.imresize(image, (64, 64))
                    images.extend([image])
                    measurement = float(line[3])
                    measurements.extend([measurement])
                    images.extend([cv2.flip(image,1)])
                    measurements.extend([measurement *-1.0])
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Dropout
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
# from keras.models import Model
# from matplotlib.pyplot as plt

col, row, ch = 64, 64, 3

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5,
        input_shape=(col, row, ch),
        output_shape=(col, row, ch)))
model.add(Conv2D(24, (5, 5), padding='valid', activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), padding='valid', activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), padding='valid', activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), padding='valid', activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Dropout(.2))
model.add(Dense(50))
model.add(Dropout(.1))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, epochs=4)
model.fit_generator(train_generator, steps_per_epoch=6*len(train_samples), validation_data=validation_generator, validation_steps=6*len(validation_samples), epochs=3)

model.save('model.h5')
