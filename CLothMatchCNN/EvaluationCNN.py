#used; https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#used: https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718

# Keras and TensorFlow must be (pip) installed.
from keras.applications import InceptionV3
from keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
import keras
import os
import PIL.Image
from keras.layers import Dense, Dropout, Activation

import argparse
import keras
import os

import tensorflow as tf
import numpy      as np

from keras.models       import Model
from keras.applications import InceptionV3
from keras.models       import Sequential
from keras.layers       import Dense, Dropout, Activation
from keras.optimizers   import SGD

INPUT_HEIGHT = 299
INPUT_WIDTH  = 299
INPUT_MEAN   = 127.5
INPUT_STD    = 127.5

def read_file(file_name):
    """
    Convert string of .jpg file path to normalized np array for image processing.
    """
    file_reader = tf.io.read_file(file_name, "file_reader")
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
    #resized = tf.image.resize(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
    normalized = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD])

    sess = tf.Session()
    result = sess.run(normalized)

    return result[0]

def LoadDataset(path, imgLim = -1):
    images = []
    labels = []
    for indx, (root, dirs, files) in enumerate(os.walk(path, topdown=False)):
        for file in files[:imgLim]:
            if indx >= imgLim and imgLim != -1: break
            print(file, indx)
            labels.append(indx)
            #images.append(PIL.Image.open(root + "\\" + file))
            images.append(read_file(root + "\\" + file))

    return np.asarray(images), np.asarray(labels)

processed_imgs_array, labels = LoadDataset("C:\\Users\\kaborg15\\PycharmProjects\\MasterThesisClothesFittingwGANs\\EvaluationDataset",2000)
processed_val_array, labelsRed = LoadDataset("C:\\Users\\kaborg15\\PycharmProjects\\MasterThesisClothesFittingwGANs\\DiverseImagesSimplified")

original_model = InceptionV3()
bottleneck_input = original_model.get_layer(index=0).input
bottleneck_output = original_model.get_layer(index=-2).output
bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)

for layer in bottleneck_model.layers:
    layer.trainable = False


new_model = keras.Sequential()
new_model.add(bottleneck_model)
new_model.add(Dense(2, activation="softmax", input_dim=2048)) #input_dim=2048, input_shape=(2, )

new_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

bottleneck_model.summary()
new_model.summary()
exit()

one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
new_model.fit(processed_imgs_array,
              one_hot_labels,
              epochs=2000,
              batch_size=32)

processed_test_array, labels_test = LoadDataset("C:\\Users\\kaborg15\\PycharmProjects\\MasterThesisClothesFittingwGANs\\EvaluationTestDataset", 100)
one_hot_labels_test = keras.utils.to_categorical(labels_test, num_classes=2)
eval = new_model.evaluate(processed_test_array, one_hot_labels_test)
print(eval)

preds = new_model.predict(processed_val_array)
print(preds)
print(preds.max())
print(preds.argmax(-1))


new_model.save("fullModel_3.h5")
print("Saved model to disk")
# # serialize model to JSON
# model_json = new_model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# new_model.save_weights("model.h5")
# print("Saved model to disk")