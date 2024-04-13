# how to update the codebook weight in autoencoder in to the BKRA module.

import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from argparse import ArgumentParser

def combine(args):
    config_ldm = OmegaConf.load(args.config)

    ##create ldm model and load weight
    model = instantiate_from_config(config_ldm.model)
    model.load_state_dict(torch.load(args.ldm_model_path)['state_dict'],strict=False)


    cache_state_dict=model.state_dict()
    cache_state_dict['model.SBG_module.bg_embed'] = rearrange(cache_state_dict['cond_stage_model.quantize.embedding.weight'], 'a b -> b a')

    # load a updated state_dict 
    model.load_state_dict(cache_state_dict)

    ## save the updated model
    torch.save({'state_dict':model.state_dict()}, args.savemodel)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--config', default="ldm/models/ldm/inpainting_big/config_LAKERED.yaml", help='Path to YAML ...')
    parser.add_argument('--ldm_model_path', default="ldm/models/ldm/inpainting_big/last.ckpt", help='Path to MODEL ...')
    parser.add_argument('--savemodel', default="ldm/models/ldm/inpainting_big/LAKERED_init.ckpt", help='Path to SAVE ...')
    
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    
    combine(args)