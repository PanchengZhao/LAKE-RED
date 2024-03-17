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

config_ldm = OmegaConf.load("/home/wwl/zpc/Code/Diffusion/LAKE-RED/ldm/models/ldm/inpainting_big/config_LAKERED.yaml")
ldm_model_path = "/home/wwl/zpc/Code/Diffusion/LAKE-RED/ldm/models/ldm/inpainting_big/last copy.ckpt"

##create ldm model and load weight
model = instantiate_from_config(config_ldm.model)
model.load_state_dict(torch.load(ldm_model_path)['state_dict'],strict=False)

# vqvae2 = torch.load('/home/ubuntu14/zpc/Code/Diffusion/vq-vae-2-pytorch/vqvae_560.pt')

cache_state_dict=model.state_dict()
# cache_state_dict['model.SBG_module.bg_embed'] = vqvae2['quantize_t.embed']
cache_state_dict['model.SBG_module.bg_embed'] = rearrange(cache_state_dict['cond_stage_model.quantize.embedding.weight'], 'a b -> b a')

# load a updated state_dict 
model.load_state_dict(cache_state_dict)

## save the updated model
torch.save({'state_dict':model.state_dict()}, "/home/wwl/zpc/Code/Diffusion/LAKE-RED/ldm/models/ldm/inpainting_big/LAKERED_init.ckpt")