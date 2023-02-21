# The json module is mainly used to convert the python dictionary above into a JSON string that can be written into a file
import json
import math
import os
from pprint import pprint


import numpy as np
import tensorflow as tf
# Here we'll show the currently installed version of TensorFlow
print(tf.version.VERSION)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



N_POINTS = 10
# The .constant() method will creates a constant tensor from a tensor-like object.
X = tf.constant(range(N_POINTS), dtype=tf.float32)
Y = 2 * X + 10



# Let's define create_dataset() procedure
# TODO 1
def create_dataset(X, Y, epochs, batch_size):
# Using the tf.data.Dataset.from_tensor_slices() method we are able to get the slices of list or array
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset




BATCH_SIZE = 3
EPOCH = 2

dataset = create_dataset(X, Y, epochs=EPOCH, batch_size=BATCH_SIZE)

for i, (x, y) in enumerate(dataset):
# You can convert a native TF tensor to a NumPy array using .numpy() method
# Let's output the value of `x` and `y`
    print("x:", x.numpy(), "y:", y.numpy())
    assert len(x) == BATCH_SIZE
    assert len(y) == BATCH_SIZE



# Let's define loss_mse() procedure which will return computed mean of elements across dimensions of a tensor.
def loss_mse(X, Y, w0, w1):
    Y_hat = w0 * X + w1
    errors = (Y_hat - Y)**2
    return tf.reduce_mean(errors)


# Let's define compute_gradients() procedure which will return value of recorded operations for automatic differentiation
def compute_gradients(X, Y, w0, w1):
    with tf.GradientTape() as tape:
        loss = loss_mse(X, Y, w0, w1)
    return tape.gradient(loss, [w0, w1])




# TODO 2
EPOCHS = 250
BATCH_SIZE = 2
LEARNING_RATE = .02

MSG = "STEP {step} - loss: {loss}, w0: {w0}, w1: {w1}\n"

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dataset = create_dataset(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

for step, (X_batch, Y_batch) in enumerate(dataset):

    dw0, dw1 = compute_gradients(X_batch, Y_batch, w0, w1)
    w0.assign_sub(dw0 * LEARNING_RATE)
    w1.assign_sub(dw1 * LEARNING_RATE)

    if step % 100 == 0:
        loss = loss_mse(X_batch, Y_batch, w0, w1)
        print(MSG.format(step=step, loss=loss, w0=w0.numpy(), w1=w1.numpy()))
        
assert loss < 0.0001
assert abs(w0 - 2) < 0.001
assert abs(w1 - 10) < 0.001



# Defining the feature names into a list `CSV_COLUMNS`
CSV_COLUMNS = [
    'fare_amount',
    'pickup_datetime',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
    'key'
]
LABEL_COLUMN = 'fare_amount'
# Defining the default values into a list `DEFAULTS`
DEFAULTS = [[0.0], ['na'], [0.0], [0.0], [0.0], [0.0], [0.0], ['na']]



# TODO 3
def create_dataset(pattern):
# The tf.data.experimental.make_csv_dataset() method reads CSV files into a dataset
    return tf.data.experimental.make_csv_dataset(
        pattern, 1, CSV_COLUMNS, DEFAULTS)


tempds = create_dataset('../toy_data/taxi-train*')
# Let's output the value of `tempds`
print(tempds)


# Let's iterate over the first two element of this dataset using `dataset.take(2)`.
# Then convert them ordinary Python dictionary with numpy array as values for more readability:
for data in tempds.take(2):
    pprint({k: v.numpy() for k, v in data.items()})
    print("\n")



UNWANTED_COLS = ['pickup_datetime', 'key']

# Let's define the features_and_labels() method
# TODO 4a
def features_and_labels(row_data):
# The .pop() method will return item and drop from frame. 
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label


for row_data in tempds.take(2):
    features, label = features_and_labels(row_data)
    pprint(features)
    print(label, "\n")
    assert UNWANTED_COLS[0] not in features.keys()
    assert UNWANTED_COLS[1] not in features.keys()
    assert label.shape == [1]



# Let's define the create_dataset() method
# TODO 4b
def create_dataset(pattern, batch_size):
# The tf.data.experimental.make_csv_dataset() method reads CSV files into a dataset
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)
    return dataset.map(features_and_labels)$




BATCH_SIZE = 2

tempds = create_dataset('../toy_data/taxi-train*', batch_size=2)

for X_batch, Y_batch in tempds.take(2):
    pprint({k: v.numpy() for k, v in X_batch.items()})
    print(Y_batch.numpy(), "\n")
    assert len(Y_batch) == BATCH_SIZE



# TODO 4c
def create_dataset(pattern, batch_size=1, mode="eval"):
# The tf.data.experimental.make_csv_dataset() method reads CSV files into a dataset
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)

# The map() function executes a specified function for each item in an iterable.
# The item is sent to the function as a parameter.
    dataset = dataset.map(features_and_labels).cache()

    if mode == "train":
        dataset = dataset.shuffle(1000).repeat()

    # take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(1)
    return dataset



tempds = create_dataset('../toy_data/taxi-train*', 2, "train")
print(list(tempds.take(1)))
 

print(list(tempds.take(1)))

#next steps: use the dataset from killer game
