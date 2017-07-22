import csv
import cv2
import shutil
import scipy.misc
import random
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
TURN_CORRECTION = 0.20

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.extend([line])

data = []
for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = cv2.imread(current_path)
        measurement = float(line[3])

        # Recovery data right turn
        if measurement > 0.15:
            filename = line[1].split('/')[-1]
            current_path = '../data/IMG/' + filename
            image = cv2.imread(current_path)
            measurement = float(line[3]) + TURN_CORRECTION

        # Recovery data left turn
        if measurement < 0.15:
            filename = line[2].split('/')[-1]
            current_path = '../data/IMG/' + filename
            image = cv2.imread(current_path)
            measurement = float(line[3]) - TURN_CORRECTION

        data.extend([(image, measurement)])

def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for sample in batch_samples:
                    image = crop_resize(random_brightness(sample[0])
                    measurement = sample[1]
                    images.extend([image])
                    measurements.extend([measurement])

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img

def crop_resize(image):
  cropped = cv2.resize(image[60:140,:], (64,64))
  return cropped

train_samples, validation_samples = train_test_split(data, test_size=0.2)
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
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3)

model.save('model.h5')
