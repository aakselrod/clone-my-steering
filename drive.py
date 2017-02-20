from train_config import *
import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from scipy.misc import imresize

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    # Decode b64-encoded image into array
    image = np.asarray(Image.open(BytesIO(base64.b64decode(imgString))))
    # Crop image
    image = image[(max_ud_shift):(160-max_ud_shift),(max_lr_shift):(320-max_lr_shift),:]
    # Resize image
    image = imresize(image, (img_size, img_size, 3))
    # Add None batch size to image
    image = image[None, :, :, :]
    # Use image and maybe speed to predict steering angle and maybe throttle
    if include_speed:
        speed_array = np.empty((1))
        speed_array[0] = speed
        speed_array = speed_array[None, :]
        X = [image, speed_array]
    else:
        X = image

    prediction = model.predict(X, batch_size=1)

    steering_angle = float(prediction[0,0])
    if predict_throttle:
        # Use the throttle prediction
        throttle = float(prediction[0,1])
    else:
        # Keep a constant speed by accelerating only when speed is too low
        if speed < target_speed:
            throttle = constant_throttle
        else:
            throttle = 0

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
