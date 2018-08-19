import csv
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

# Reading the file paths into a list
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# split into train and validation
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

# Define a generator function to  extract chunks of data in real time - for processing/augmentation.
def generator(lines, batch_size = 8):
	num_samples = len(lines)
	while 1:
		shuffle(lines)
		for offset in range(0, num_samples, batch_size):
			batch_samples = lines[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
				#read in left and right images along with center images.
				for i in range(3):
					source_path = batch_sample[i]
					tokens = source_path.split('/')
					filename = tokens[-1]
					local_path = './data/IMG/' + filename
					image = cv2.imread(local_path)
					#convert to RGB
					images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

				# correspoding measurement for each image.

				correction = 0.2
				measurement = float(batch_sample[3])
				measurements.append(measurement)
				measurements.append((measurement + correction))
				measurements.append((measurement - correction))
				# Data augmentation - flip the images.

				augmented_images = []
				augmented_measurements = []
 				# Data augmentaion - random brightness change 
					
				for image, measurement in zip(images, measurements):
					augmented_images.append(image)
					augmented_measurements.append(measurement)
					flipped_image = cv2.flip(image, 1)
					flipped_measurement = measurement * -1.0
					augmented_images.append(flipped_image)
					augmented_measurements.append(flipped_measurement)

				for i in range(2):
					index = random.randint(0, len(augmented_images) - 1)
					temp_img = cv2.cvtColor(augmented_images[index], cv2.COLOR_RGB2HSV)
					random_wgt = 0.25*np.random.uniform()
					temp_img[:,:,2] = temp_img[:,:,2]*random_wgt
					temp_img = cv2.cvtColor(temp_img, cv2.COLOR_HSV2RGB)
					augmented_images.append(temp_img)
					augmented_measurements.append(measurement)	

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield shuffle(X_train,y_train)

train_generator      =  generator(train_samples, batch_size=8)
validation_generator  =  generator(validation_samples, batch_size =8)


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25),(0,0))))  #Output image shape is 65x320x3

model.add(Convolution2D(24,(5,5), strides=(2, 2), activation='relu'))

model.add(Dropout(0.5))

model.add(Convolution2D(36,(5,5), strides=(2,2),activation='relu'))

model.add(Dropout(0.5))

model.add(Convolution2D(48,(5,5), strides=(2,2),activation='relu'))

model.add(Dropout(0.5))

model.add(Convolution2D(64,(3,3),activation='relu'))

model.add(Dropout(0.5))

model.add(Convolution2D(64,(3,3), activation='relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100))

model.add(Dense(50))

model.add(Dense(10))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//8, validation_data = validation_generator, validation_steps = len(validation_samples)//8, epochs = 5, use_multiprocessing=True )
model.save('model.h5')
