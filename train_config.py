###################################################################
# The included model.json and model.py were trained with the
# hyperparameters defined below
# There are ways to tweak it and still make it work, but these
# settings seemed to minimize swerving
###################################################################

# Where to find training data
base_paths = ['training-data/alex2-track1']

# Image resize value - this fits well on my GTX 750
# Image is resized AFTER augmentation
img_size = 64

###################################################################
# Training parameters

# Batch size - this fits well on my GTX 50
batch_size = 256

# Number of samples per epoch - should be multiple of batch size
samples_per_epoch = 10240

# Number of samples per validation run - should be multiple of batch size
samples_per_validation = 2560

# Number of epochs
# NOTE: model checkpoint is saved after every epoch, so training
# can be interrupted at any time
nb_epoch = 15

# Validation portion of data
validation_ratio = 0.25

###################################################################
# Model Architecture

# Include speed in input to model
include_speed = False

# Try to predict throttle
predict_throttle = False

# Constant speed with which to drive if throttle prediction is off
target_speed = 15

# Constant throttle to use when throttle prediction is off and
# speed is lower than the target speed
constant_throttle = 0.4

# Convolutional layers - array of tuples
# Each tuple contains depth and filter size (square)
convs = [(32, 3), (64, 3)]

# Dense layers - each one has a dropout before it
dense_widths = [64, 16]

# Dropout value
dropout = 0.5

###################################################################
# Use center image
use_ctr = True

###################################################################
# Use left/right images
use_lr = True

# Steering adjustment (additive) for side images
steering_adjustment = 0.3

# Throttle adjustment (multiplicative) for side images
# Only used when throttle prediction is on
throttle_adjustment = 0.75

###################################################################
# Data augmentation - brightness
# Set max adjustment to 0 to disable

# Max brightness adjustment - between 0 and 1
max_bright = 0.5

###################################################################
# Data augmentation - rotate - VERY CPU INTENSIVE
# Training takes about 10x as much time with this on
# scipy.ndimage.interpolation.rotate does NOT use GPU
# keras.preprocessing.image.random_rotation also does NOT use GPU,
# and does NOT return rotation angle, so we don't use it
# Set max angle to 0 to disable

# Max angle to rotate
max_rotate = 0

# Steering adjustment per shifted degree
steering_per_degree = 0.003

###################################################################
# Data augmentation - shift left/right
# Set max shift to 0 to disable

# Max number of pixels to shift left to right
max_lr_shift = 10

# Steering adjustment per shifted pixel
steering_per_pixel = 0.003

###################################################################
# Data augmentation - shift up/down
# Set max shift to 0 to disable

# Max number of pixels to shift up and down
max_ud_shift = 35

###################################################################
# Row filter
# Return True if the row should be used, False if not
def row_filter(steer, throttle, brake, speed):
	return speed > 10
