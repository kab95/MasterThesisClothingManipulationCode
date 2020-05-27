import sys
sys.path.insert(0, "D:\PythonProjectsDDrive\stylegan2-master")

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import pretrained_networks

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

def main():
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = "D:\PythonProjectsDDrive\stylegan2-master"
    sc.run_desc = 'generate-images'
    network_pkl = 'D:\PythonProjectsDDrive\stylegan2-master\TrainedGANs\\network-snapshot-018708.pkl'

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    vector_size = Gs.input_shape[1:][0]
    #
    seedData = range(18000, 19000)
    seeds = expand_seed(seedData, vector_size)
    generate_images(Gs, seeds, seedData, "ArithmaticBaseImages", truncation_psi=0.5)

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