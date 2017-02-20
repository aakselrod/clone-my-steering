from train_config import *
import csv, random
import numpy as np
import tensorflow as tf
from math import sqrt
from PIL import Image
from scipy.misc import fromimage, imresize
from scipy.ndimage.interpolation import rotate
from keras.engine.topology import Merge
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from sklearn.model_selection import train_test_split

# Load CSV data
steering = {}
speed = {}
throttle = {}
for base_path in base_paths:
	with open(base_path + '/driving_log.csv') as csvfile:
		fieldnames = ['ctr','lt','rt','steer','throttle','brake','speed']
		reader = csv.DictReader(csvfile, fieldnames)
		for row in reader:
			st = float(row['steer'])
			th = float(row['throttle'])
			br = float(row['brake'])
			sp = float(row['speed'])
			# Use only rows that pass filter
			if row_filter(st, th, br, sp):
				if use_ctr:
					ctr = base_path + '/IMG' + row['ctr'][row['ctr'].rfind('/'):].strip()
					steering[ctr] = st
					throttle[ctr] = th
					speed[ctr] = sp
				if use_lr:
					lt = base_path + '/IMG' + row['lt'][row['lt'].rfind('/'):].strip()
					steering[lt] = st + steering_adjustment
					throttle[lt] = th * throttle_adjustment
					speed[lt] = sp
					rt = base_path + '/IMG' + row['rt'][row['rt'].rfind('/'):].strip()
					steering[rt] = st - steering_adjustment
					throttle[rt] = th * throttle_adjustment
					speed[rt] = sp

# A list for random selection
images = list(steering.keys())
images, test_images = train_test_split(images, test_size=validation_ratio)

# This generator creates a train/validation batch at random, including augmentation.
def my_generator(batch_size, imgdata):
	while 1:
		# Create empty arrays for image and speed inputs and labels
		X_img = np.empty((batch_size,img_size,img_size,3), dtype=np.float32)
		X_speed = np.empty((batch_size,1), dtype=np.float32)
		if predict_throttle:
			shape = (batch_size, 2)
		else:
			shape = (batch_size, 1)
		y = np.empty(shape, dtype=np.float32)
		# For each image in a batch, load and augment the image
		for i in range(batch_size):
			index = random.randint(0, len(imgdata) - 1)
			# Load image
			img = fromimage(Image.open(imgdata[index])).astype(np.float32)
			# Shift left or right by up to max_lr_shift pixels
			lr_shift = int(random.uniform(-max_lr_shift-0.1,max_lr_shift+0.1))
			# Shift up or down by up to max_ud_shift pixels
			ud_shift = int(random.uniform(-max_ud_shift-0.1,max_ud_shift+0.1))
			# Crop based on shift and top/bottom crop to take out car, horizon
			img = img[(max_ud_shift-ud_shift):(160-max_ud_shift-ud_shift),(max_lr_shift-lr_shift):(320-max_lr_shift-lr_shift),:]
			X_speed[i] = speed[images[index]]
			# Adjust steering by shifted pixels multiplied by steering adjustment per pixel
			y[i,0] = steering[images[index]] - (lr_shift * steering_per_pixel)
			# Random rotation
			if max_rotate > 0:
				angle = random.uniform(-max_rotate,max_rotate)
				img = rotate(img, angle, reshape=False, mode='nearest')
				y[i,0] -= (angle * steering_per_degree)
			# Resize image by resize ratio
			X_img[i] = imresize(img, (64, 64, 3))
			if predict_throttle:
				y[i,1] = throttle[images[index]]
			# Random brightness adjustment
			if max_bright > 0:
				brightness = random.uniform(1-max_bright,1+max_bright)
				X_img[i] = X_img[i] * brightness
			# 50% chance of flipping image/steering
			if random.random() > 0.5:
				X_img[i] = np.fliplr(X_img[i])
				y[i,0] = -y[i,0]
		if include_speed:
			yield [X_img, X_speed], y
		else:
			yield X_img, y

# Create the convolutional part of the model for the image
model_conv = Sequential()
# Normalize the image colors
model_conv.add(Lambda(lambda x: ((x/255)-0.5), input_shape=(img_size, img_size, 3)))
# Option to put multiple convolutional layers between MaxPool layers
# 'same' border mode because it can give us deeper networks
for (d, s) in convs:
	model_conv.add(Convolution2D(d, s, s, border_mode='same', activation='relu'))
	model_conv.add(MaxPooling2D())
model_conv.add(Flatten())

# Merge convolutional model into final model:
if include_speed:
	# Create the flat part of the model for the speed
	model_flat = Sequential()
	model_flat.add(Dense(1, input_shape=(1,)))
	model = Sequential()
	model.add(Merge([model_conv, model_flat], mode='concat', concat_axis=1))
else:
	model = model_conv

# Add dense layers to predict based on features found by convolutional
# layers and maybe speed
for width in dense_widths:
	model.add(Dropout(dropout))
	model.add(Dense(width, activation='relu'))

# Add prediction layer
if predict_throttle:
	num_predictions = 2
else:
	num_predictions = 1
model.add(Dense(num_predictions, activation='linear'))

# Train the model
model.compile('adam', 'mse')
# Manual loop, save after every epoch so the training can be stopped with Ctrl-C and all the training up
# through the last epoch is saved
for i in range(nb_epoch):
	model.fit_generator(my_generator(batch_size, images), samples_per_epoch = samples_per_epoch,
		nb_epoch = 1, validation_data = my_generator(batch_size, test_images),
		nb_val_samples = samples_per_validation)
	open('model.json', 'w').write(model.to_json())
	model.save_weights('model.h5')

