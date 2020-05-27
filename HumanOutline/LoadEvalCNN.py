from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import tensorflow as tf
import numpy as np
import keras
import cv2

#from EvaluationCNN import LoadDataset
INPUT_HEIGHT = 299
INPUT_WIDTH  = 299
INPUT_MEAN   = 127.5
INPUT_STD    = 127.5

def read_file(file_name, sess):
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

    result = sess.run(normalized)

    return result[0]
    # numpyVar = cv2.imread(file_name)
    # return cv2.resize(numpyVar, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)


def LoadDataset(path, imgLim = -1):
    sess = tf.Session()

    images = []
    labels = []
    fileNames = []
    for indx, (root, dirs, files) in enumerate(os.walk(path, topdown=False)):
        for file in files[:imgLim]:
            print(file, indx)
            labels.append(indx)
            #images.append(PIL.Image.open(root + "\\" + file))
            fileNames.append(file)
            images.append(read_file(root + "//" + file, sess))

    return np.asarray(images), np.asarray(labels), fileNames


# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")


# load model
loaded_model = keras.models.load_model('ImageEvaluation\\TrainedCNNModels\\fullmModel.h5')
# summarize model.
loaded_model.summary()
# load dataset
# split into input (X) and output (Y) variables
# evaluate the model
# score = model.evaluate(X, Y, verbose=0)
exit()
#processed_test_array, labels_test = LoadDataset("ImageEvaluation\\EvaluationDataset", 100)

processed_pred_array, labels_pred, file_names = LoadDataset("ImageEvaluation\gaTest")#"MainExperimentResults")
#one_hot_labels_pred = keras.utils.to_categorical(labels_pred, num_classes=2)

# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#one_hot_labels_test = keras.utils.to_categorical(labels_test, num_classes=2)
#score = loaded_model.evaluate(processed_test_array, one_hot_labels_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

preds = loaded_model.predict(processed_pred_array)
for i, pred in enumerate(preds):
    print(pred, file_names[i])
#print(preds)
print(preds.max())
print(preds.argmax(-1))