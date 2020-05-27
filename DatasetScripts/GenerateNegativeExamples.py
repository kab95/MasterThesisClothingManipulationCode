import sys
sys.path.insert(0, "D:\PythonProjectsDDrive\stylegan2-master")

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import datetime
import tensorflow as tf
import pretrained_networks
import random
#----------------------------------------------------------------------------

def load_seed_latents(seedData):
    latents = []
    for seedPath in seedData:
        latents.append(np.load(seedPath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII'))

    return np.array(latents)


def LoadImageFromVector(vecList, Gs, truncation_psi=0.5, saveBool=False, saveNameMod=""):
    imageList = []

    # noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    # Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    # Gs_kwargs.randomize_noise = False
    # if truncation_psi is not None:
    #     Gs_kwargs.truncation_psi = truncation_psi

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    dlatent_avg = Gs.get_var('dlatent_avg')  # [component]

    rnd = np.random.RandomState()
    #tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

    totalImageWidth = 0
    for indx, vec in enumerate(vecList):
        #images = Gs.run(vec, None, **Gs_kwargs)  # [minibatch, height, width, channel]

        # row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
        dl = (vec - dlatent_avg) * truncation_psi + dlatent_avg
        images = Gs.components.synthesis.run(vec, **Gs_kwargs)

        tmpIm = PIL.Image.fromarray(images[0], 'RGB')
        tmpIm.save("D:\\PythonProjectsDDrive\\MasterThesisClothesFittingwGANs\\DatasetProcesssing\\NegativeExamples\\" + str(indx).zfill(4) + "_0.jpg")
    #amlgImg.show()

def PosNeg():
    rand = random.randint(-1,1)
    return 4 if rand == 1 else -4

def main():
    #exampleDL = load_seed_latents(["D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ToEditImagesDL\seed8007.npy"])
    test = np.random.rand(1, 14, 512) * 8 - 4

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = "D:\PythonProjectsDDrive\stylegan2-master"
    sc.run_desc = 'generate-images'
    #network_pkl = 'D:\PythonProjectsDDrive\stylegan2-master\TrainedGANs\\network-snapshot-018708.pkl'
    network_pkl = 'D:\PythonProjectsDDrive\stylegan2-master\TrainedGANs\\256-model-network-snapshot-018708.pkl'
    example_img_pkl = "D:\PythonProjectsDDrive\MasterThesisClothesFittingwGANs\LatentSpaceImages\ToEditImagesDL\seed8007.npy"

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    vector_size = Gs.input_shape
    #
    #seedData = range(18000,19000)
    #startSeed = 190000
    #seeds = expand_seed(seedData, vector_size)
    #generate_images(Gs, seeds, seedData, "ArithmaticBaseImages", truncation_psi=0.5)
    dlatentVectorList = []#[tf.random_normal((1, 14, 512))]*10
    for i in range(10000):
        dlatentVectorList.append(np.random.rand(1, 14, 512) * 8 - 4)
    LoadImageFromVector(dlatentVectorList, Gs)

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

    # def generate_images_in_w_space(dlatents, truncation_psi, index, folder):
    #     Gs_kwargs = dnnlib.EasyDict()
    #     Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    #     Gs_kwargs.randomize_noise = False
    #     Gs_kwargs.truncation_psi = truncation_psi
    #     dlatent_avg = Gs.get_var('dlatent_avg')  # [component]
    #     imgs = []
    #     counter = 0
    #     for row, dlatent in enumerate(dlatents):
    #         # row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
    #         dl = (dlatent - dlatent_avg) * truncation_psi + dlatent_avg
    #         row_images = Gs.components.synthesis.run(dlatent, **Gs_kwargs)
    #         # if row == 0: print(row_images)
    #         tmpImgs = PIL.Image.fromarray(row_images[0], 'RGB')
    #         #         tmpImgs.save(folder+"/image"+str(index),"jpeg")
    #         counter += 1
    #         imgs.append(tmpImgs)
    #     return tmpImgs
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------