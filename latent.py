import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import IPython.display
from training import misc
import argparse
import PIL.Image
import re
import sys
import os
from io import BytesIO
import IPython.display
from math import ceil
from PIL import Image, ImageDraw
import os
import glob
import pretrained_networks



#----------------------- Functions ------------------------------

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
            
    
# makes a avrage vector of a series of vectors in folders
def loadAvrageVector(pathToFolder,pathToFile):
    files = os.listdir(pathToFolder)
    sumVec =np.load(pathToFile, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    counter=0
    for filepath in files:
        if filepath.endswith(".npy") and ( "10000" in filepath or "7000" in filepath) and filepath != pathToFile:
                editVec = np.load(pathToFolder+"/"+filepath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
                sumVec = np.add(sumVec,editVec)
                counter+=1
    returnVec = np.divide( sumVec,counter)
    return returnVec



#taken from https://github.com/dvschultz/stylegan2/blob/master/StyleGAN2_Projection.ipynb
# Generates a list of images, based on a list of latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_in_w_space(dlatents, truncation_psi,index,folder):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
    imgs = []
    counter=0
    for row, dlatent in enumerate(dlatents):
        dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
        row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
        tmpImgs = PIL.Image.fromarray(row_images[0], 'RGB')
#         tmpImgs.save(folder+"/image"+str(index),"jpeg")
        counter+=1
        imgs.append(tmpImgs)
    return tmpImgs


# takes a series of vectors from generator images and makes a series of vectors modified
def personsVectorModifier(personsFolderPath,vectorLists):
    seriesOfFiles = os.listdir(personsFolderPath)
    listOfFilePathToVector = []
    for files in seriesOfFiles:
        if files.endswith(".npy"):
            listOfFilePathToVector.append(files)
    count= 0
    for files in listOfFilePathToVector:
        if count<= 10:
            count+=1
            vectorList = []
            temp =  np.load(personsFolderPath+"/"+files)
            vectorList.append(temp)
            for i in  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,]:
                changedTemp= np.add(np.multiply(latentCloatingPice,i),temp)
                changedTemp = np.add(changedTemp ,np.multiply(latentAvgCloath, i*0.7))
                                    
                                    
                
                diff = np.subtract(changedTemp, latentAvgPerson)
                changedTemp = np.subtract(changedTemp, np.multiply(diff,i*0.01))
                vectorList.append(changedTemp)
            vectorLists.append(vectorList)


def interpolating(STEPS,latentCloatingPice,latentGenratedImage,latentAvgPerson,latentAvgCloath,current,vectorList):
    latentV=(latentCloatingPice*3+latentGenratedImage+latentAvgPerson*0.1-latentAvgCloath)
    # generate_images_in_w_space(np.stack([latentGenratedImage]),1,0)    
    diff = latentV - latentGenratedImage
    step = (diff / STEPS)
    count=0
    for i in range(STEPS):
    #  print("is there a nan steps",np.isnan(current).any())  
    #  print(count,current[0][0][0:10])  
      count+=1
      vectorList.append(current)
      current = current + step           
            
            

#some printlines that can be usefull when debuging
def debugprints(latentCloatingPice,latentGenratedImage,diff,step):
    print("latent cloathing pice",latentCloatingPice[0][0][0:10])
    print("generated image",latentGenratedImage[0][0][0:10])
    print("is there a nan cloathing piece",np.isnan(latentCloatingPice[0][0]).any())
    print("is there a nan",np.isnan(diff[0][0]).any())
    print("----------difference---",diff[0][0][0:30])
    print("is there a nan",np.isnan(step[0][0]).any())
    print("----------step---",step[0][0][0:30])
    

def savingImages(ImageList,saveNameMod):
    totalImageWidth = 0
#     print("Imagelist length ",str(len(ImageList)))
    imageList = []
    for tmpIm in ImageList:
        
        imageList.append(tmpIm)
        totalImageWidth += tmpIm.width

    amlgImg = PIL.Image.new('RGB', (totalImageWidth, imageList[0].height))
    print("imagelist length ",str(len(imageList)))
    imgWidthTracker = 0
    for imgT in imageList:
        amlgImg.paste(imgT, (imgWidthTracker, 0))
        imgWidthTracker += imgT.width
    amlgImg.save(folder+"/" + str(saveNameMod) + ".png")
    

def findlatestBuild(path,positionstart,positionEnd):
    files = glob.glob(path)
    count=0
    fileListInt = []
    for f in files:
        fileListInt.append([int(f[positionstart:positionEnd]),f])

    highestint = 0
    highestfile = ""                  
    for i in fileListInt:
        if i[0] >= highestint:
                highestint = i[0]
                highestfile = i[1]
    return highestfile
    

def loadPath():
    print( "the resoulution is ",resolution)
    latentCloatingPicePath= ""
    latentGenratedImagePath=""
    clothingPicesPath=""
    startingClothinPicePath=""
    personsFolderPath=''
    startingPersonPath=''
    networkPath=''
    if resolution == "256":
        latentCloatingPicePath= "results/00021-project-real-images/image0000-stepSeed1000.pk.npy"
        latentGenratedImagePath= "results/00025-project-generated-images/seed0005-stepSeed0200.pk.npy"
        clothingPicesPath="results/00026-project-real-images"
        startingClothinPicePath= "results/00026-project-real-images/image0001-stepSeed1000.pk.npy"
        personsFolderPath= "damagesDataset/dlatent_generated_images_256"
        startingPersonPath="damagesDataset/dlatent_generated_images_256/seed7000.npy"
        networkPath='results/00004-stylegan2-256resolution-8gpu-config-f/network-snapshot-018708.pkl'
    else:
        latentCloatingPicePath= "results/00047-project-real-images/image0006-stepSeed10000.pk.npy"
        latentGenratedImagePath="results/00027-generate-images/seed7000.npy"
        clothingPicesPath="results/00047-project-real-images/"
        startingClothinPicePath="results/00047-project-real-images/image0000-stepSeed1000.pk.npy"
        personsFolderPath="results/00027-generate-images"
        startingPersonPath= "results/00027-generate-images/seed7000.npy"
        networkPath=findlatestBuild(findlatestBuild("results/Transfer/*",18,22)+"/network-snapshot*",86,92)
    return latentCloatingPicePath,latentGenratedImagePath,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath


#--------------------variabels----------------------------------------


# pathes

#chose resolution 256 or 1024
resolution = "1024"
folder = "testing"


latentCloatingPicePath = ""
latentGenratedImagePath =""
clothingPicesPath =  ""
startingClothinPicePath =  ""
personsFolderPath =""
startingPersonPath = ""
networkPath = ""



STEPS = 20
vectorLists = []


#----------------------- main--------------------------------
#getting paths
latentCloatingPicePath,latentGenratedImagePath,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath = loadPath()


#loading vectors from files
latentCloatingPice = np.load(latentCloatingPicePath)
latentGenratedImage = np.load(latentGenratedImagePath)
latentAvgCloath=loadAvrageVector(clothingPicesPath,startingClothinPicePath)
latentAvgPerson=loadAvrageVector(personsFolderPath,startingPersonPath)
current = latentGenratedImage.copy()


#loading network
sc = dnnlib.SubmitConfig()
sc.num_gpus = 1
sc.submit_target = dnnlib.SubmitTarget.LOCAL
sc.local.do_not_copy_source_files = True
sc.run_dir_root = "/content/drive/My Drive/projects/stylegan2"
sc.run_desc = 'generate-images'
network_pkl = networkPath
print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)


deletefiles(folder)
infoVectorList = []
infoVectorList.append(latentAvgCloath)
infoVectorList.append(latentAvgPerson)
infoVectorList.append(latentCloatingPice)
vectorLists.append(infoVectorList)
interpolatingVectorList = []
interpolating(STEPS,latentCloatingPice,latentGenratedImage,latentAvgPerson,latentAvgCloath,current,interpolatingVectorList)
vectorLists.append(interpolatingVectorList)
personsVectorModifier(personsFolderPath,vectorLists)

    
#print("number of lists",str(len(vectorLists))," 0: ",str(len(vectorLists[0])))


for index,vectorList in enumerate(vectorLists):
    imageList = []
    count = 0
    for vectors in vectorList:
        imageList.append(generate_images_in_w_space(np.stack([vectors]),1,index,folder))
    savingImages(imageList,index)
    

# renamefiles(folder)

  