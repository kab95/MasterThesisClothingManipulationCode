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

def generate_images(Gs, seeds, truncation_psi):
    #noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):#[0][0]):
        print('Generating image for seed %d/%d ...' % (seed_idx, len(seeds)))
        #rnd = np.random.RandomState()
        #tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(np.array([seed[0][0]]), None, **Gs_kwargs) # [minibatch, height, width, channel]
        #images = Gs.run(np.array([seed]), None, **Gs_kwargs)  # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').show()

vec4 = np.load("D:\PythonProjectsDDrive\stylegan2-master\\results\\00023-project-real-images\image0003-stepSeed15000.pk.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')

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


generate_images(Gs, [vec4], truncation_psi=0.5)