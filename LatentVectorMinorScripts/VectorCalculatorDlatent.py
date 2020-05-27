import sys
sys.path.insert(0, "D:\PythonProjectsDDrive\stylegan2-master")

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os
import datetime

import pretrained_networks

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#----------------------------------------------------------------------------

def expand_seed(seeds, vector_size):
  result = []

  for seed in seeds:
    rnd = np.random.RandomState(seed)
    result.append( rnd.randn(1, vector_size) )
  return result

def generate_images(Gs, seeds, names, subfolder, truncation_psi):
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d/%d ...' % (seed_idx, len(seeds)))
        rnd = np.random.RandomState()
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(seed, None, **Gs_kwargs) # [minibatch, height, width, channel]
        path = f"D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\GeneratedImages\{subfolder}\image{seed_idx}_{names[seed_idx]}"
        PIL.Image.fromarray(images[0], 'RGB').save(path + ".png")
        np.save(path, seed)

def LoadImageFromVector(vecList, Gs, truncation_psi=0.5, saveBool=False, saveNameMod=""):
    imageList = []

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    rnd = np.random.RandomState()
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

    totalImageWidth = 0
    for vec in vecList:
        images = Gs.run(vec, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        tmpIm = PIL.Image.fromarray(images[0], 'RGB')
        imageList.append(tmpIm)
        totalImageWidth += tmpIm.width

    amlgImg = PIL.Image.new('RGB', (totalImageWidth, imageList[0].height))
    imgWidthTracker = 0
    for imgT in imageList:
        amlgImg.paste(imgT, (imgWidthTracker, 0))
        imgWidthTracker += imgT.width

    if saveBool: amlgImg.save("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ArithmeticResults\\" + datetime.datetime.now().timestamp().__str__().replace(".", "") + saveNameMod + ".png")
    amlgImg.show()

def LoadVectorAverage(dirPath):
    #totalVector = np.empty((1, 512))  # [] #a 512 dim numpy array
    totalVector = np.empty((1, 14, 512))
    vecCounter = 0
    for subdir, dirs, files in os.walk(dirPath):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".npy"):
                tempVec = np.load(filepath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
                totalVector += tempVec
                vecCounter += 1
    avrgVec = totalVector / vecCounter
    folderName = dirPath.split('\\')
    print(f"Average of {vecCounter} vectors taken from {folderName}")
    return avrgVec

def PerformVectorOperationAddSub(ImageEditDir, subVec, addVec, Gs):
    addVec *= 2.5
    for subdir, dirs, files in os.walk(ImageEditDir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".npy"):
                editVec = np.load(filepath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
                sumVec = editVec - subVec
                sumVec += addVec
                #LoadImageFromVector([editVec, subVec, addVec, sumVec], Gs, saveBool=True, saveNameMod="_Flip")
                LoadImageFromVector([subVec], Gs, saveBool=True, saveNameMod="_Flip")


def main():
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = "D:\PythonProjectsDDrive\stylegan2-master"
    sc.run_desc = 'generate-images'
    network_pkl = "D:\PythonProjectsDDrive\stylegan2-master\TrainedGANs\\256-model-network-snapshot-018708.pkl"#'D:\PythonProjectsDDrive\stylegan2-master\TrainedGANs\\network-snapshot-018708.pkl'

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs =pretrained_networks.load_networks(network_pkl)
    vector_size = Gs.input_shape[1:][0]

    # vec1 = np.load("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ToEditImages\image39_19039.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    # vec2 = np.load("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ToEditImages\image40_19040.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    # vec3 = np.load("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ToEditImages\image41_19041.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    # vec4 = np.load("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ToEditImages\image42_19042.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    # displayList = [vec1, vec2, vec3, vec4]
    # #LoadImageFromVector(displayList, Gs)
    #
    # avrgVecBack = LoadVectorAverage("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\BackImages")
    # avrgVecFront = LoadVectorAverage("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\FrontImages")

    avrgGreyShirtDL = LoadVectorAverage("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\MonoClothingIMages")
    avrgPersonDL = LoadVectorAverage("D:\PythonProjectsDDrive\stylegan2-master\\results\\00063-generate-images")
    PerformVectorOperationAddSub("D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ToEditImagesDL", avrgPersonDL, avrgGreyShirtDL, Gs)
    #LoadImageFromVector([avrgVecBack], Gs)

    #
    # seedData = range(19000,20000)
    # seeds = expand_seed(seedData, vector_size)
    # generate_images(Gs, seeds, seedData, "ArithmaticBaseImages", truncation_psi=0.5)

    # STEPS = 300
    # MeldSeeds = (0, 9)
    # diff = seeds[MeldSeeds[0]] - seeds[MeldSeeds[1]]
    # step = diff / STEPS
    # current = seeds[MeldSeeds[1]].copy()
    #
    # seeds2 = []
    # for i in range(STEPS):
    #     seeds2.append(current)
    #     current = current + step

    #generate_images(Gs, seeds2, [f"{seedData[MeldSeeds[0]]}-{seedData[MeldSeeds[1]]}-{i}" for i in range(len(seeds2))], "MeldGeneratedImages", truncation_psi=0.5)
    #generate_images(Gs, seeds2, ["" for i in range(len(seeds2))], "ArithmaticBaseImages", truncation_psi=0.5)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------