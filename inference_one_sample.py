import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, Subset
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
transform_PIL = T.ToPILImage()
from einops import rearrange, repeat
from torchvision.utils import make_grid
from math import sqrt

import uuid

##create model
def create_model(device):
    #load config and checkpoint
    config = OmegaConf.load(yaml_path)
    config.model['params']['ckpt_path']=model_path
    model = instantiate_from_config(config.model)
    sampler = DDIMSampler(model)
    model = model.to(device)
    return model,sampler

def process_data(image_pth,mask_pth):
    mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)
    original_size = mask.shape
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    dilated_mask = Image.fromarray(dilated_mask) 
    dilated_mask = np.expand_dims(dilated_mask, axis=2)
    dilated_mask = dilated_mask.astype(np.float32) / 255.0#
    dilated_mask[dilated_mask < 0.1] = 0
    dilated_mask[dilated_mask >= 0.1] = 1
    dilated_mask = dilated_mask[None].transpose(0,3,1,2)
    dilated_mask = torch.from_numpy(dilated_mask)

    # normalzie and transform the image into tensor
    image = np.array(Image.open(image_pth).convert("RGB").resize((512,512)))
    image = image.astype(np.float32) / 255.0#
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = Image.open(mask_pth).convert("L").resize((512,512))    
    mask = np.expand_dims(mask, axis=2)
    mask = mask.astype(np.float32) / 255.0#
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask[None].transpose(0,3,1,2)
    mask = torch.from_numpy(mask)
    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": dilated_mask, "masked_image": masked_image}
    
    for k in batch:
        batch[k] = batch[k] * 2.0 - 1.0
    imagename = image_pth.split('/')[-1].split('.')[0]
    return batch, original_size, imagename


# config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yaml_path="/home/wwl/zpc/Code/Diffusion/LAKE-RED/ldm/models/ldm/inpainting_big/config_LAKERED.yaml"
model_path="/home/wwl/zpc/Code/Diffusion/LAKE-RED/ckpt/LAKERED.ckpt"
if not os.path.exists('./test_log'):
    os.makedirs('./test_log')
model,sampler=create_model(device)
model.eval()


image_pth = '/media/sde/zpc/Dataset/COD_Diff_/validation/images/COD_CAMO_camourflage_00012.jpg'
mask_pth = '/media/sde/zpc/Dataset/COD_Diff_/validation/masks/COD_CAMO_camourflage_00012.png'
logpath = os.path.join('./test_log', image_pth.split('/')[-1].replace('.','_'))
if not os.path.exists(logpath):
    os.makedirs(logpath)
batchsize = 4
try:
    batch, original_size, imgname = process_data(image_pth,mask_pth)
    
    # encode masked image and concat downsampled mask
    c = model.cond_stage_model.encode(batch["masked_image"].to(device))

    # the mask is frst being downsampled
    cc = torch.nn.functional.interpolate(batch["mask"].to(device),
                                        size=c.shape[-2:])
    # concat the masked image and downsampled mask
    c = torch.cat((c, cc), dim=1)
    
    shape = (c.shape[1]-1,)+c.shape[2:]
    
    c = c.expand(batchsize, -1,-1,-1)
    
    cond = c
    S = 50
    # diffusion process
    samples_ddim, _ = sampler.sample(S=S,
                                conditioning=cond,
                                batch_size=c.shape[0],
                                shape=shape,
                                verbose=False)

    # samples_ddim = c
    # decode the latent vector (output)
    x_samples_ddim = model.decode_first_stage(samples_ddim)


    # denormalize the output
    predicted_image_clamped = torch.clamp((x_samples_ddim+1.0)/2.0,
                                min=0.0, max=1.0)
    
    # imgsdst = os.path.join(dst_root, 'images', imgname)
    # if not os.path.exists(imgsdst):
    #     os.makedirs(imgsdst)
    
    all_samples= []
    # base_count = len(os.listdir(logpath))
    base_count = len(glob.glob(os.path.join(logpath, 'sample*')))
    grid_count = len(glob.glob(os.path.join(logpath, 'grid*')))
    
    for sample in predicted_image_clamped:
        output_PIL=transform_PIL(sample)
        output_PIL.save(os.path.join(logpath, f"sample_{base_count:05}.png"))
        base_count += 1
    all_samples.append(predicted_image_clamped)
    
    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=int(sqrt(batchsize)))

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    grid = Image.fromarray(grid.astype(np.uint8))
    grid.save(os.path.join(logpath, f'grid-{grid_count:04}.png'))
    grid_count += 1

    # mask = transform_PIL(1 - batch["mask"].squeeze())
    # mask.save(os.path.join(dst_root, 'masks', imgname+'.png'))
    
except RuntimeError:
    print('RuntimeError: ', image_pth)    
            

