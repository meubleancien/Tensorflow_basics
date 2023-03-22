#Setting up the environment

import os
PROJECT_ID = "cloud-training-demos"  # Replace with your PROJECT
BUCKET = PROJECT_ID 
REGION = 'us-central1'
# Store the value of `BUCKET` and `PROJECT_ID` in environment variables.
os.environ["PROJECT_ID"] = PROJECT_ID
os.environ["BUCKET"] = BUCKET

'''
# Using `mkdir` we can create an empty directory
!mkdir train
# Using `touch` we can create an empty file
!touch train/__init__.py
'''

#create modeel 

import tensorflow as tf
import numpy as np

# Get data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# add empty color dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

def create_model():
# The `tf.keras.Sequential` method will sequential groups a linear stack of layers into a tf.keras.Model.
    model = tf.keras.models.Sequential()
# The `Flatten()` method will flattens the input and it does not affect the batch size.
    model.add(tf.keras.layers.Flatten(input_shape=x_train.shape[1:]))
# The `Dense()` method is just your regular densely-connected NN layer.
    model.add(tf.keras.layers.Dense(1028))
# The `Activation()` method applies an activation function to an output.
    model.add(tf.keras.layers.Activation('relu'))
# The `Dropout()` method applies dropout to the input.
    model.add(tf.keras.layers.Dropout(0.5))
# The `Dense()` method is just your regular densely-connected NN layer.
    model.add(tf.keras.layers.Dense(512))
# The `Activation()` method applies an activation function to an output.
    model.add(tf.keras.layers.Activation('relu'))
# The `Dropout()` method applies dropout to the input.
    model.add(tf.keras.layers.Dropout(0.5))
# The `Dense()` method is just your regular densely-connected NN layer.
    model.add(tf.keras.layers.Dense(256))
# The `Activation()` method applies an activation function to an output.
    model.add(tf.keras.layers.Activation('relu'))
# The `Dropout()` method applies dropout to the input.
    model.add(tf.keras.layers.Dropout(0.5))
# The `Dense()` method is just your regular densely-connected NN layer.
    model.add(tf.keras.layers.Dense(10))
# The `Activation()` method applies an activation function to an output.
    model.add(tf.keras.layers.Activation('softmax'))
    return model


#Create a model to train locally

import os
# The Python time module provides many ways of representing time in code, such as objects, numbers, and strings.
# It also provides functionality other than representing time, like waiting during code execution and measuring the efficiency of your code.
import time
# Here we'll import data processing libraries like Numpy and Tensorflow
import tensorflow as tf
import numpy as np
from train import model_definition

#Get data
# TODO 2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# add empty color dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

def create_dataset(X, Y, epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset

ds_train = create_dataset(x_train, y_train, 20, 5000)
ds_test = create_dataset(x_test, y_test, 1, 1000)

model = model_definition.create_model()

model.compile(
# Using `tf.keras.optimizers.Adam` the optimizer will implements the Adam algorithm.
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),
  loss='sparse_categorical_crossentropy',
  metrics=['sparse_categorical_accuracy'])
    
start = time.time()

model.fit(
    ds_train,
    validation_data=ds_test, 
    verbose=1
)
print("Training time without GPUs locally: {}".format(time.time() - start))


#Train on multiple GPUs/CPUs with MultiWorkerMirrored Strategy

import os
# The Python time module provides many ways of representing time in code, such as objects, numbers, and strings.
# It also provides functionality other than representing time, like waiting during code execution and measuring the efficiency of your code.
import time
# Here we'll import data processing libraries like Numpy and Tensorflow
import tensorflow as tf
import numpy as np
from . import model_definition

# The `MultiWorkerMirroredStrategy()` method will work as a distribution strategy for synchronous training on multiple workers.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

#Get data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# add empty color dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

def create_dataset(X, Y, epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset

ds_train = create_dataset(x_train, y_train, 20, 5000)
ds_test = create_dataset(x_test, y_test, 1, 1000)

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = model_definition.create_model()
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])
    
start = time.time()

model.fit(
    ds_train,
    validation_data=ds_test, 
    verbose=2
)
print("Training time with multiple GPUs: {}".format(time.time() - start))


#Training with multiple GPUs/CPUs on created model using MultiWorkerMirrored Strategy
'''
BASH
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-highcpu-16


now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="cpu_only_fashion_minst_$now"

gcloud ai-platform jobs submit training $JOB_NAME \
  --staging-bucket=gs://$BUCKET \
  --package-path=train \
  --module-name=train.train_mult_worker_mirrored \
  --runtime-version=2.3 \
  --python-version=3.7 \
  --region=us-west1 \
  --config config.yaml


# Configure a master worker
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-highcpu-16
  masterConfig:
    acceleratorConfig:
      count: 2
      type: NVIDIA_TESLA_K80


now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="multi_gpu_fashion_minst_2gpu_$now"

gcloud ai-platform jobs submit training $JOB_NAME \
  --staging-bucket=gs://$BUCKET \
  --package-path=train \
  --module-name=train.train_mult_worker_mirrored \
  --runtime-version=2.3 \
  --python-version=3.7 \
  --region=us-west1 \
  --config config.yaml

'''