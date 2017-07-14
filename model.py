import csv
import cv2
import shutil

files = []
for i in range(1):
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
            new_path = '../IMG/' + filename
            #shutil.copy2(current_path, new_path)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import numpy as np
import sklearn

def generator(samples, batch_size=32):
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
            print(images)
            print(measurements)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
# from keras.models import Model
# from matplotlib.pyplot as plt

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, epochs=4)
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=(validation_samples), epochs=3)
model.save('model.h5')
