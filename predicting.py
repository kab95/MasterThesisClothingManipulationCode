from keras.models import model_from_json
import PIL.Image
from PIL import Image, ImageDraw
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import cv2
from skimage.transform import resize
import os
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import glob
import numpy as np
import cv2


# loads the model    
def loadModel(path):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path)
    print("Loaded model from disk")
    return loaded_model



def reSizeTo299(numpyVar,height,length,sess,graph_resize):
    #print("numpy var" numpyVar.shape())
    #exit()
    with graph_resize.as_default():
        with sess.as_default():
            INPUT_HEIGHT = height
            INPUT_WIDTH  = length
            INPUT_MEAN   = 127.5
            INPUT_STD    = 127.5

            #float_caster = tf.cast(image_reader, tf.float32)
            dims_expander = tf.expand_dims(np.asarray(numpyVar), 0)#float_caster, 0)
            #dims_expander = tf.expand_dims(np.asarray([numpyVar[0][0]]), 0)#float_caster, 0)
            resized = tf.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
            #resized = tf.image.resize(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
            normalized = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD]).eval(session=sess)
    #         result = sess.run(normalized)
    # cv2.resize(numpyVar, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)#result[0]
    return normalized
  

    

#taken from https://github.com/dvschultz/stylegan2/blob/master/StyleGAN2_Projection.ipynb
# Generates a list of images, based on a list of latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_in_w_space(dlatents, truncation_psi,index,Gs):
   
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
    imgs = []
    #counter=0
    for row, dlatent in enumerate(dlatents):
        #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
        dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
        row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
        #if row == 0: print(row_images)
        tmpImgs = PIL.Image.fromarray(row_images[0], 'RGB')
        #tmpImgs.save(folder+"/image"+str(index),"jpeg")
        #counter+=1
        imgs.append(tmpImgs)
    return tmpImgs







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

def loadPath(resolution):
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



# makes a avrage vector of a series of vectors in folders
def loadAvrageVector(pathToFolder,pathToFile):
    files = os.listdir(pathToFolder)
    sumVec =np.load(pathToFile, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    counter=0
    for filepath in files:
        if filepath.endswith(".npy") and ( "1000" in filepath or "7000" in filepath) and filepath != pathToFile:
                editVec = np.load(pathToFolder+"/"+filepath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
                sumVec = np.add(sumVec,editVec)
                counter+=1
    returnVec = np.divide( sumVec,counter)
    return returnVec



  
    
def get_dlatents(vector_scaler,latentCloatingPicePath,latentGenratedImagePath,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath):
    latentCloatingPice = np.load(latentCloatingPicePath)
    latentGenratedImage = np.load(latentGenratedImagePath)
    latentAvgCloath=loadAvrageVector(clothingPicesPath,startingClothinPicePath)
    latentAvgPerson=loadAvrageVector(personsFolderPath,startingPersonPath)
#     dlatent=latentGenratedImage*vector_scaler[0]+latentCloatingPice*vector_scaler[1]
    dlatent = []
#     print(latentGenratedImage.shape,"gen im")
  
    scaledGenvec =np.multiply(latentGenratedImage,vector_scaler[0])
#     print(scaledGenvec.shape,"scale im")
    scaledClothvec =np.multiply(latentCloatingPice,vector_scaler[1])
    scaledAvgClothVec = np.multiply( latentAvgCloath,vector_scaler[2]*0.1)
    scaledAvgPerVec = np.multiply( latentAvgPerson,vector_scaler[3]*0.1)
    
    dlatent = np.add( scaledClothvec,scaledGenvec)
    dlatent = np.add( dlatent,scaledAvgClothVec)
    dlatent = np.add( dlatent,scaledAvgPerVec)
#     print(dlatent.shape,"dlat")
#     for (item1, item2) in zip(latentGenratedImage, latentCloatingPice):
#         for (i1, n2) in zip(item1, item2):
#             for (i, n) in zip(i1, n2):
#                 dlatent.append(i*vector_scaler[0]+n*vector_scaler[1])
    
#     for i in latentGenratedImage:
#         i= i*vector_scaler[0]
    
#     +(latentGenratedImage-latentAvgCloath)*vector_scaler[2]-(latentGenratedImage-latentAvgPerson)*vector_scaler[3]
#     print(latentGenratedImage.shape,"her")
#     print(latentCloatingPice.shape,"kler")
#     print(latentAvgCloath.shape,"gjen kler")
#     print(latentAvgPerson.shape," gjen pers ")
#     dlatent =latentGenratedImage-(latentGenratedImage-latentCloatingPice)*vector_scaler[0] -(latentGenratedImage-latentAvgCloath)*vector_scaler[1] -  (latentGenratedImage-latentAvgPerson)*vector_scaler[2]

    
#     seriesOfFiles = os.listdir(personsFolderPath)
#     listOfFilePathToVector = []
#     for files in seriesOfFiles:
#         if files.endswith(".npy"):
#             listOfFilePathToVector.append(files)
#     count= 0
#     for files in listOfFilePathToVector:
#         if count<= 0:
#             count+=1
#             vectorList = []
#             temp +=  np.load(personsFolderPath+"/"+files)
            
#     dlatent+= temp*vector_scaler[3]
   
    return dlatent

def makeStartingImageFromVector(vector,Gs):
    img = generate_images_in_w_space(np.stack([vector]),1,1,Gs)
    return img
    
    
def makeVectorFromTargetpath(path,Gs,sess,graph_resize):
#     latentCloatingPicePath,latentGenratedImagePath,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath= loadPath("1024")
#     Gs,_D =loadingNetworks()
#     img = generate_images_in_w_space(np.stack([vector]),1,1,Gs)
    
    
    clothingPathes = os.listdir("results/00047-project-real-images/")
    pathToImage = ""
    for i in  clothingPathes:
#         print(i,"i and path :",path)
        if path[:9] in i[:9] :
#             print("her er jeg 1",i)
            if   "target" in i :
#                 print("her er jeg 2",i)
                pathToImage="results/00047-project-real-images/"+i
    
#     print(pathToImage)
#     with graph_resize.as_default():
#         with sess.as_default():

#             INPUT_HEIGHT = 256
#             INPUT_WIDTH  = 256
#             INPUT_MEAN   = 127.5
#             INPUT_STD    = 127.5


#             file_reader = tf.io.read_file(pathToImage, "file_reader")
#             image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
#             float_caster = tf.cast(image_reader, tf.float32)
#             dims_expander = tf.expand_dims(float_caster, 0)
#             resized = tf.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
#             #resized = tf.image.resize(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
#             normalized = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD])

#             result = normalized.eval(session=sess)
    result = PIL.Image.open(pathToImage).resize((256,256),resample=0,box=None)
#     result=cv2.resize( cv2.imread(pathToImage), dsize=(256, 256), interpolation=cv2.INTER_AREA)#img/
    return result
  
def makeArrayFromImage(image,Gs,sess,graph_resize):

    with graph_resize.as_default():
        with sess.as_default():

            INPUT_HEIGHT = 256
            INPUT_WIDTH  = 256
            INPUT_MEAN   = 127.5
            INPUT_STD    = 127.5


           
#             image_reader = tf.image.decode_jpeg(image, channels=3, name='jpeg_reader')
#             float_caster = tf.cast(image_reader, tf.float32)
            dims_expander = tf.expand_dims(np.asarray(image), 0)
            resized = tf.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
            #resized = tf.image.resize(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
            normalized = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD])

            result = normalized.eval(session=sess)
        #      cv2.resize(np.asarray(img), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)#img/
    return result



def loadingNetworks():
    
#loading network
    latentCloatingPicePath,latentGenratedImagePath,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath= loadPath("1024")

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = "/content/drive/My Drive/projects/stylegan2"
    sc.run_desc = 'generate-images'
    network_pkl = networkPath
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    
    return Gs,_D

def imageGeneration(population,randomStartingImage,path,Gs,sess,graph_resize):
    latentCloatingPicePath,latentGenratedImagePath,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath= loadPath("1024")
#     Gs,_D =loadingNetworks(networkPath)
    dlatentsList = []
    for i in population:
#         dlatents = get_dlatents(i.number,latentCloatingPicePath,latentGenratedImagePath,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath)
        dlatents =get_dlatents(i.number,path,randomStartingImage,clothingPicesPath,startingClothinPicePath,personsFolderPath,startingPersonPath,networkPath)
        dlatentsList.append(dlatents)
      
    
    img_list = []
    input_array_list = []
    for dlat in dlatentsList:
        img = generate_images_in_w_space(np.stack([dlat]),1,1,Gs)
        input_array = np.asarray(img)
        img_list.append(img)
        #print(input_array.reshape([-1,256,256,3]).shape)
        #exit()
        
#         print("her er D predict ",_D(img))
        #input_array_list.append(input_array.reshape([-1,256,256,3]))
#         input_array_list.append(cv2.resize(input_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))
        input_array_list.append(input_array)
      #  input_array_list.append(reSizeTo299(input_array,256,256,sess,graph_resize))
        
    
    #resized_img = cv2.resize(input_array,(256, 256) , interpolation = cv2.INTER_AREA)
#s    resized_array = resized_img.reshape([-1,256,256,3])
#     resized_array = resize(input_array,(254, 254))
    
    #prediction = model.predict(input_array, verbose=0)
  
    
    return input_array_list,img_list