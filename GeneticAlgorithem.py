from keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import PIL.ImageOps
import random
import keras
import predicting
import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import PIL.Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


class indevid():
    
    def __init__(self):
        self.fitness = 0
        self.number = []
        self.img = None


#
def makeStartingIndeviduels(POPULATION_NUMBER):
    population = []
    for i in range (POPULATION_NUMBER):
        newIndevid = indevid()
        for i in range(4):
            newIndevid.number.append( random.uniform(-0.3,0.3))
            #print("numbers",newIndevid.number)
        population.append(newIndevid)
    return population

#juge the fitnees of a indevidual 
def fittnes(population,Clothing_array,randomStartingImage,path,Gs,sess,graph_resize): 
    with graph_Generate.as_default():
        with session_Generate.as_default():
            img_array_list, image_list, = predicting.imageGeneration(population,randomStartingImage,path,Gs,sess,graph_resize)
#     model = predicting.loadModel("ConvolutionDatasett/nameOfModel.h5")
    count = 0
    printlineList =[]
    for indevidual in population:
        if indevidual.fitness == 0:
            predictionCloth = []
            predictionPerson = []
            sizeTouple = (256,256)
            resized_image = PIL.Image.fromarray(img_array_list[count]).resize((256,256),resample=0,box=None)
#             resized_image =  cv2.resize(img_array_list[count],sizeTouple, interpolation=cv2. INTER_AREA)
#             resized_image = predicting.reSizeTo299(img_array_list[count],256,256,sess,graph_resize)
            
#             clothInput = np.concatenate((Clothing_array,resized_image), axis=None).reshape([-1,512,256,3])
            amlgImg = PIL.Image.new('RGB', (512, 256))
          
#             amlgImg.paste(PIL.Image.fromarray((resized_image*255).astype(np.uint8)), (0,256))
            amlgImg.paste((resized_image), (256,0))
            amlgImg.paste(Clothing_array ,(0, 0))
#             amlgImg = PIL.ImageOps.invert(amlgImg)
#             amlgImg.save("comparGaTest"+"/" + "test" + ".png")
            
            input_arr = keras.preprocessing.image.img_to_array(amlgImg)
            clothInput  = np.array([input_arr])  # Convert single image to a batch.
            
#             clothInput = np.asarray(compImg).reshape([-1,512,256,3])
           
          
          
            personInput = predicting.reSizeTo299(img_array_list[count],299,299,sess,graph_resize).reshape([-1,299,299,3])
           
            with graph_ClothCNN.as_default():
                 with session_ClothCNN.as_default():
                        predictionCloth = modelClothCNN.predict( clothInput , verbose=0)
            with graph_PersonCNN.as_default():
                 with session_PersonCNN.as_default():
                        predictionPerson = modelPersonCNN.predict( personInput , verbose=0)
            score = 0
           
            for i in predictionPerson:
                score += i[1]
           
            for i in predictionCloth:
                score += i[0]
#             score = predictionCloth[0][0]+predictionPerson[0][1]*1000
                
            
            indevidual.fitness = score
            indevidual.img = image_list[count]
            #print("KLARTE DET KOM IGJENOM")
            printlineList.append(["feedback"+str(predictionPerson)+" "+str(predictionCloth)+str(score),score])
            count+=1
    printlineList.sort(key=lambda x: x[1], reverse=True)
    for i in printlineList:
        print(i[0])
   

            
            
#make new variations
def bred(population,CHANCE_OF_MUTATION):
    newPopulation = population
    random.shuffle(population)
    for i in range(int(len(population)/2)):
        parent1 = population.pop()
        parent2 = population.pop()
        if parent1.number != parent2.number:
            child1 = indevid()
            child1.number =parent1.number.copy()
            gen = random.randint(0,3)
            child1.number[gen] = parent2.number[gen] 
            child2 = indevid()
            child2.number =parent2.number.copy()
            gen = random.randint(0,3)
            child2.number[gen] = parent1.number[gen]
            mutation(child1,CHANCE_OF_MUTATION)
            mutation(child2,CHANCE_OF_MUTATION)
            newPopulation.append(parent1)
            newPopulation.append(parent2)
            newPopulation.append(child1)
            newPopulation.append(child2)
    return newPopulation

#random chance of mutation
def mutation(indevid,CHANCE_OF_MUTATION):
    if random.uniform(0, 1) > 1.0-CHANCE_OF_MUTATION:
        effected_gene = random.randint(0,3)
        #print("mutated")
        indevid.number[effected_gene] = indevid.number[effected_gene] + random.uniform(-0.5, 0.5)

def OnlyMutate(population):
    newPopulation = []
    for i in population:
        newIndevid = indevid()
        newIndevid.number = i.number.copy()
        mutation(newIndevid,1)
        newPopulation.append(i)
        newPopulation.append(newIndevid)
    return newPopulation
        
        
        
    

#kill half population
def killHalf(population):
    #sort after fitness remove half with lower fittness
    #print("population before",len(population))
    population.sort(key=lambda x: x.fitness, reverse=True)
    population = population[0:int(len(population)/2)]
    #print("population after",len(population))
    printFitness(population)
    return population
    
    
def printFitness(population):
    count = 1
    for i in population:
        #print(count, i.fitness)
        count+=1
        
# removes the images currently in the folder
def deletefiles(folder):
    files = glob.glob(folder+'/*')
    count=0
    for f in files:
         os.remove(f)
        

    
# Adds png to end of all files in folder
def renamefiles(folder):
    files = glob.glob(folder+'/*')
    count=0
    for f in files:
    #      os.remove(f)
       os.rename(f,f+".png")
       count+=1


def  startsavingImages(clothingImgpath,clothingVector,randomStartingImagePath,index,path,Gs):
     
    clothingPathes = os.listdir("results/00047-project-real-images/")
    pathToImage = ""
    for i in  clothingPathes:
#         print(i,"i and path :",path)
        if clothingImgpath[:9] in i[:9] :
#             print("her er jeg 1",i)
            if   "target" in i :
#                 print("her er jeg 2",i)
                pathToImage="results/00047-project-real-images/"+i
            
    
    
    totalImageWidth = 0

#     print("Imagelist length ",str(len(ImageList)))
    vectorOfStartingImage = np.load(randomStartingImagePath)
    tempImageList = []
    with graph_Generate.as_default():
        with session_Generate.as_default():
            startingImg = predicting.makeStartingImageFromVector(vectorOfStartingImage,Gs)
            clothing_img = predicting.makeStartingImageFromVector(clothingVector,Gs)
    tempImageList.append( PIL.Image.open(pathToImage))
    tempImageList.append(clothing_img)
    tempImageList.append(startingImg)
    imageList = []
    for tmpIm in tempImageList:
        imageList.append(tmpIm)
        totalImageWidth += tmpIm.width
    amlgImg = PIL.Image.new('RGB', (totalImageWidth, imageList[0].height))
    print("imagelist length ",str(len(imageList)))
    imgWidthTracker = 0
    for imgT in imageList:
        amlgImg.paste(imgT, (imgWidthTracker, 0))
        imgWidthTracker += imgT.width
    amlgImg.save(folder+"/" + "start-"+str(index)+"-"+path+ ".png")
        

        
def savingImage(image,folder):
    amlgImg = PIL.Image.new('RGB', (image[0].width, image[0].height))
   
    amlgImg.paste(image, (0, 0))
    
    amlgImg.save(folder+"/" + str(saveNameMod)+str(len(os.listdir(folder))) + ".png")
    
    


def savingImages(population,saveNameMod,folder):
    totalImageWidth = 0

#     print("Imagelist length ",str(len(ImageList)))
    tempImageList = []
    count=0
    for indevidual in population:
        if count <= 10:
            tempImageList.append(indevidual.img)
            with open(folder+"/"+saveNameMod, "a+") as outfile:
                outfile.write(" ".join(n[:4] for n in str(indevidual.number)))
                outfile.write("  ".join(str(indevidual.fitness)))
                outfile.write("\n")
                
        count+=1
    imageList = []
    for tmpIm in tempImageList:
        imageList.append(tmpIm)
        totalImageWidth += tmpIm.width
    amlgImg = PIL.Image.new('RGB', (totalImageWidth, imageList[0].height))
    print("imagelist length ",str(len(imageList)))
    imgWidthTracker = 0
    for imgT in imageList:
        amlgImg.paste(imgT, (imgWidthTracker, 0))
        imgWidthTracker += imgT.width
    amlgImg.save(folder+"/" + str(saveNameMod)+str(len(os.listdir(folder))) + ".png")
    
    

def savingImagesForDataSet(population,saveNameMod,folder):
    totalImageWidth = 0
    folderForDataSet = "gaTest"
#     print("Imagelist length ",str(len(ImageList)))
    tempImageList = []
    count=0
    for indevidual in population:
        if count <= 10:
            indevidual.img.save(folderForDataSet+"/"+str(saveNameMod)+str(len(os.listdir(folderForDataSet))) +".png")
            #tempImageList.append(indevidual.img)
        count+=1
    imageList = []
    for tmpIm in tempImageList:
        
        amlgImg.paste(tmpIm, (0,0))
        imageList.append(tmpIm)
        totalImageWidth += tmpIm.width
    amlgImg = PIL.Image.new('RGB', (totalImageWidth, imageList[0].height))
    amlgImg.save(folderForDataSet+"/" + str(saveNameMod)+str(len(os.listdir(folderForDataSet)) + ".jpg"), "JPEG")# + ".png")
    print("--------------------saving images ------------------------------------------")
    

#------------------ variabels-------------------

POPULATION_NUMBER = 40
CHANCE_OF_MUTATION = 0.6
# CLOTHING_IMAGE_PATH = "results/00034-project-real-images/image0002-stepSeed1000.pk.npy"
epoch = 15
folder = "GaTest"


    
#--------------------main-----------------------
graph_resize = tf.Graph()
with graph_resize.as_default():
    sess = tf.compat.v1.Session() 


graph_ClothCNN = tf.Graph()
with graph_ClothCNN.as_default():
       session_ClothCNN = tf.Session()
       with session_ClothCNN.as_default():
          modelClothCNN = predicting.loadModel("ConvolutionDatasett/nameOfModel2.h5")
        

graph_PersonCNN = tf.Graph()
with graph_PersonCNN.as_default():
       session_PersonCNN = tf.Session()
       with session_PersonCNN.as_default():
          modelPersonCNN  = keras.models.load_model('HumanEvaluationResources/fullModel_3.h5')

        
graph_Generate = tf.Graph()
with graph_Generate.as_default():
       session_Generate = tf.Session()
       with session_Generate.as_default():
          Gs,_D =predicting.loadingNetworks()
            


deletefiles(folder)
pathToCloatingPiceFolder = "results/00047-project-real-images"
pathToPersonFolder = "results/00027-generate-images"
pathesInPersonFolder = os.listdir(pathToPersonFolder)
personFolderList = []
for filepath in pathesInPersonFolder:
    print(filepath)
    if ".npy" in filepath:
#         print(filepath)
        personFolderList.append(filepath)

filesinClothingPiceFolder = os.listdir(pathToCloatingPiceFolder)
clothingPiceList = []
for filepath in filesinClothingPiceFolder:
    print(filepath)
    if  "10000.pk.npy" in filepath:
#         print(filepath)
        clothingPiceList.append(filepath)

    
random.shuffle(clothingPiceList)
for path in clothingPiceList:
    pathToClothingPice = "results/00047-project-real-images/"+path
    clothingPiceVector = np.load(pathToClothingPice)
    with graph_Generate.as_default():
        with session_Generate.as_default():
            clothing_img  = predicting.makeStartingImageFromVector(clothingPiceVector,Gs)
    Clothing_array = predicting.makeVectorFromTargetpath(path,Gs,sess,graph_resize)
#     Clothing_array = predicting.makeArrayFromImage(clothing_img,Gs,sess,graph_resize)
    
    
#     print(Clothing_array.shape)
    #exit()
    for i in range(5):
        
        population = makeStartingIndeviduels(POPULATION_NUMBER)
        randomStartingImagePath = random.choice(personFolderList)
        randomStartingImagePath = "results/00027-generate-images/"+randomStartingImagePath
        startsavingImages(path,clothingPiceVector,randomStartingImagePath,i,path,Gs)
        fittnes(population,Clothing_array,randomStartingImagePath,pathToClothingPice,Gs,sess,graph_resize)
    #     savingImages(population,"start",folder)
        print("starting population")
        printFitness(population)

        #loop
        for i in range(epoch):
            print("------------Bread-------- before")
            printFitness(population)
            population = bred(population,CHANCE_OF_MUTATION)
#             population= OnlyMutate(population)
            print("------------Bread-------- after")
            printFitness(population)

            fittnes(population,Clothing_array,randomStartingImagePath,pathToClothingPice,Gs,sess,graph_resize)
            print("-------------fittness-------after")
            printFitness(population)
            print("kill population")
            population=killHalf(population)
            if i%1==0:
                savingImages(population,"epoch"+str(i),folder)
#             savingImagesForDataSet(population,"test",folder)
            endLoopCounter = 0
            for i in population:
                if i.fitness >= 2.0 :
                    endLoopCounter+=1
            if endLoopCounter >= 20:
                break
            

        print("final fitness")
        printFitness(population)
        savingImages(population,"finish"+str(),folder)
