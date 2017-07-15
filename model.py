import csv
import cv2
import shutil

files = []
for i in range(3):
    files.append([])
    with open('../data%s/driving_log.csv' % str(i)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            files[i].append(line)

samples = []
for idx, lines in enumerate(files):
    for line in lines:
        samples.append(line)
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = '../data' + str(idx) + '/IMG/' + filename
            image = cv2.imread(current_path)
            crop_image = image[50:140, 0:320]
            small_image = cv2.resize(crop_image, (0,0), fx=0.5, fy=0.5)
            new_path = '../IMG/' + filename
            cv2.imwrite(new_path,small_image)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import numpy as np
import sklearn
batch_size = 32
def generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '../IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3])
                    correction = 0.2
                    if i == 1:
                        measurement + correction
                    elif i == 2:
                        measurement - correction
                    measurements.append(measurement)
                    images.append(cv2.flip(image,1))
                    measurements.append(measurement *-1.0)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
# from keras.models import Model
# from matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(45,160,3)))
model.add(Conv2D(24, (5, 5), border_mode='valid', activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), border_mode='valid', activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), border_mode='valid', activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), border_mode='same', activation="relu"))
model.add(Conv2D(64, (3, 3), border_mode='valid', activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, epochs=4)
model.fit_generator(train_generator, steps_per_epoch=6*len(train_samples), validation_data=validation_generator, validation_steps=6*(validation_samples), epochs=1)

model.save('model.h5')
