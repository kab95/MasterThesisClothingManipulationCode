import random

from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.layers. normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np




#------------------------ Functions -------------------------------


# ----- preprosesing------

# Loads the data sett in should contain labels and 2 images per data point
# [0] is datasett person [1] datasetcloath
def loadDataSet():
    datagen = ImageDataGenerator()
    train_it = datagen.flow_from_directory('damagesDataset/ClothMatchDataset256/train',classes=["0", "1"], target_size=(256, 512) ,class_mode='binary', batch_size=64)
    return train_it


# Make the datasett graded on difficulity preprossesing
def dataMakerGradedOnDifficulity(dataSet,difficulity):
    dataSetGraded = []#"dataSetGraded"
    threshold = (50 - difficulity)/100
    
    for index,person in enumerate(dataSet[0]):
        rand = random.random()
        if rand >= threshold:
            clothingIndex = random.randint(0,len(dataset[1]))
            while clothingIndex == index:
                clothingIndex = random.randint(0,len(dataset[1]))
            dataSetGraded = [person,dataset[clothingIndex].choice,0]
        else:
            dataSetGraded = [i,dataset[1],1]           
    return datasettGraded

# load the dataset from the preprossed dataset with the right difficulity
def DataSetLoaderBasedOnDifficulity(difficulity):
    
    print("returns a dataSet with right difficulity")
    return dataSet

# ------- working with models-----

# Should make the convolutional moddel
def makeModel():
    model = "model"
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE_Y, IMG_SIZE_X, 3)))
#     model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(2, activation = 'softmax'))
    #print("todo Make it return",model)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    model.summary()
    return model

# saves The model
def saveModel(model,modelName):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("ConvolutionDatasett/"+modelName+"2"+".h5")
    print("Saved model to disk")
    
# loads the model    
def loadModel():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model





# the main traing loop
def trainingLoop(model,dataSet):
    #datagen.flow_from_directory('damagesDataset/train', classes=["True", " False"], target_size=(512, 256) ,class_mode='binary', batch_size=64)
    datagen = ImageDataGenerator()
    model.fit_generator( dataSet,  steps_per_epoch=64,
        epochs=500)
   # model.fit_generator(dataSet,steps_per_epoch=16 )
    print("train model")
    

# Returns the prediction from the network
def prediction(model):
    prediction = "prediction"
    print("make it return prediction")
    return prediction
    

# ------------------------variabels---------------------------------------



OriginalDataSet = loadDataSet()

# Modes will be training witch train from scratch and reTrain witch trains a alredy trained network and prediction
# prediction is when the network is in use to predict outcomes
# preprossesing to make the data sett
mode = "training"
#whenether the training will try to train with diffrent datasets depending on difficulity
difficulityOffOn = "off"
#number of epoch training for
epoch = 5

IMG_SIZE_X = 2048
IMG_SIZE_Y = 1024
IMG_SIZE_X = 512
IMG_SIZE_Y = 256
# The name the model will be saved with
modelName = "nameOfModel"

# The path to the model that will be retrained
loadModelPath = ""

    
    
# ------------------------------ Main -------------------------------------------

if mode == "training":
    if difficulityOffOn == "on":
        for i in range(50):
            dataSet = DataSetLoaderBasedOnDifficulity()
            trainingLoop(model,dataSet)
    elif difficulityOffOn == "off":
        dataSet = loadDataSet()
        model = makeModel()
        trainingLoop(model,dataSet)
    saveModel(model,modelName)
elif mode == "preprossesing":
    dataSet = dataMakerGradedOnDifficulity(OriginalDataSet,i)
     # retrains the network
elif mode == "reTrain":
    model = loadModel()
    if difficulityOffOn == "on":
        for i in range(50):
            dataSet = DataSetLoaderBasedOnDifficulity()
            trainingLoop(model,dataSet)
    elif difficulityOffOn == "off":
        dataSet = loadDataSet()
        trainingLoop(model,dataSet)
    saveModel(model,modelName)

                        # the mode that will return predictions
elif mode == "prediction":
    model = loadModel()
    prediction = prediction(model)
    print("prediction")
