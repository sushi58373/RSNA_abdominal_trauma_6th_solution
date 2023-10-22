import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import threading
import pydicom
import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import glob
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import albumentations
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import monai
from monai.transforms import Resize
import monai.transforms as transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
print(device, monai.__version__)

from timm.layers.conv2d_same import Conv2dSame
from conv3d_same import Conv3dSame
from typing import Any, Dict, Optional

DEBUG = False

# Config
kernel_type = 'test'
load_kernel = None
load_last = True
n_blocks = 4
n_folds = 5
backbone = 'resnet18d'

image_sizes = (128, 128, 128)

msk_size = image_sizes[0] # 128
image_size_cls = 224 # 224
n_slice_per_c = 15
n_ch = 5

init_lr = 3e-3
batch_size_seg = 1
drop_rate = 0.
drop_path_rate = 0.
loss_weights = [1,1]
p_mixup = 0.1

use_amp = True
num_workers = 16
out_dim = 5

seed = 42

result_dir = '../results/segmentations'
exp_dir = os.path.join(result_dir, kernel_type)
model_dir = os.path.join(exp_dir, 'model')

data_dir = '../data'
dcm_dir = os.path.join(data_dir, 'dataset/train_images')
seg_dir = os.path.join(data_dir, 'dataset/segmentations')
png_dir = os.path.join(data_dir, 'png_folder')

segmented_dir = os.path.join(data_dir, f'segmented') # output directory

transforms_valid = transforms.Compose([])

df_seg = pd.read_csv(os.path.join(data_dir, 'df_seg.csv'))
train_df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))


names = {    
    1 : "liver",
    2 : "spleen",
    3 : "left kidney",
    4 : "right kidney",
    5 : "bowel",
}




def gen_pickle():
    """
        generate pickle file for png_files by "patient-series id"
        to handle difference of image order between dcm files and png files that ordered by z-axis correctly.
    """
    if os.path.isfile(os.path.join(data_dir, 'd.pkl')):
        with open(os.path.join(data_dir, 'd.pkl'), 'rb') as f:
            d = pickle.load(f)
        print('load pickle file')
    else:
        print('generate pickle file')
        t = glob.glob(png_dir + '/*')
        print(len(t))

        d = {}
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            png_suffix = row['png_suffix']
            k = png_suffix.split('/')[-1]

            d[k] = [k for k in t if png_suffix in k]
            d[k] = sorted(d[k], key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

        with open(os.path.join(data_dir, 'd.pkl'), 'wb') as f:
            pickle.dump(d, f)
        print('generate pickle file')
    return d
        
d = gen_pickle()
p = seg_dir



def load_png_files(p_paths, png_suffix):
    n_scans = len(p_paths)
    indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., image_sizes[2])).round().astype(int)
    p_paths = [p_paths[i] for i in indices]
    
    images = []
    for filename in p_paths:
        data = cv2.imread(filename)
        data = cv2.resize(data, (image_sizes[0], image_sizes[1]), interpolation = cv2.INTER_LINEAR)
        images.append(data)
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images 


class SEGOutputDatasetALL(Dataset): # train_df
    def __init__(self, df, d, slices_num=128):
        self.df = df.reset_index()
        self.d = d
        self.slices_num = slices_num

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        ### using local cache (segmentation npy)
        psid = self.df.iloc[index]['png_suffix'].split('/')[-1]
        p_paths = self.d[psid]
        image = load_png_files(p_paths, row['png_suffix'])
        image = image.transpose(2,0,1,3)
        image = image / 255.
        image = torch.tensor(image).float()
        return image



# Model
class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False):
        super(TimmSegModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=3,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1,3,64,64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )
        self.segmentation_head = nn.Conv2d(
            decoder_channels[n_blocks-1], 
            out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
    
    def forward(self, x):
        global_features = [0] + self.encoder(x)[:n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


def convert_3d(module):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
            
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output


def load_bone(msk, cid, t_paths, cropped_images):
    """
        x,y,z : segmented mask
        xx, yy, zz : cropped
    """
    n_scans = len(t_paths)
    bone = []
    try:
        msk_b = msk[cid] > 0.2
        msk_c = msk[cid] > 0.05

        x = np.where(msk_b.sum(1).sum(1) > 0)[0]
        y = np.where(msk_b.sum(0).sum(1) > 0)[0]
        z = np.where(msk_b.sum(0).sum(0) > 0)[0]

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            x = np.where(msk_c.sum(1).sum(1) > 0)[0]
            y = np.where(msk_c.sum(0).sum(1) > 0)[0]
            z = np.where(msk_c.sum(0).sum(0) > 0)[0]

        x1, x2 = max(0, x[0] - 1), min(msk.shape[1], x[-1] + 1)
        y1, y2 = max(0, y[0] - 1), min(msk.shape[2], y[-1] + 1)
        z1, z2 = max(0, z[0] - 1), min(msk.shape[3], z[-1] + 1)
        zz1, zz2 = int(z1 / msk_size * n_scans), int(z2 / msk_size * n_scans)

        if cid != 4:
            inds = np.linspace(zz1 ,zz2-1, n_slice_per_c).astype(int) # 15 slices
            inds_ = np.linspace(z1 ,z2-1, n_slice_per_c).astype(int)
        else: # bowel
            mul = 2 
            inds = np.linspace(zz1 ,zz2-1, n_slice_per_c * mul).astype(int) # 30 slices
            inds_ = np.linspace(z1 ,z2-1, n_slice_per_c * mul).astype(int)

            # save bowel meta data
            output_slices_path = os.path.join(segmented_dir, 'bowel_slices')
            psid = "_".join(t_paths[0].split('/')[-1].split('_')[0:2])
            slices_d = {
                'psid':psid, 'n_scans':n_scans, 
                'z':[z1,z2], 'zz':[zz1,zz2], 
                'inds':inds, 'inds_':inds_
            }
            with open(os.path.join(output_slices_path, psid + '.pkl'), 'wb') as f:
                pickle.dump(slices_d, f)


        for sid, (ind, ind_) in enumerate(zip(inds, inds_)):

            msk_this = msk[cid, :, :, ind_]

            images = []
            for i in range(-n_ch//2+1, n_ch//2+1):
                try:
                    png = cv2.imread(t_paths[ind+1])[:,:,1] # use only 1 slices not +-2 channels.
                    # https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/448208
                    images.append(png)
                except:
                    images.append(np.zeros((512, 512)))
                
            data = np.stack(images, -1)
            data = data - np.min(data)
            data = data / (np.max(data) + 1e-4)
            data = (data * 255).astype(np.uint8)
            msk_this = msk_this[x1:x2, y1:y2]
            xx1 = int(x1 / msk_size * data.shape[0])
            xx2 = int(x2 / msk_size * data.shape[0])
            yy1 = int(y1 / msk_size * data.shape[1])
            yy2 = int(y2 / msk_size * data.shape[1])
            data = data[xx1:xx2, yy1:yy2]
            data = np.stack([cv2.resize(data[:, :, i], (image_size_cls, image_size_cls), # to (224, 224)
                interpolation = cv2.INTER_LINEAR) for i in range(n_ch)], -1)
            msk_this = (msk_this * 255).astype(np.uint8)
            msk_this = cv2.resize(msk_this, (image_size_cls, image_size_cls), interpolation = cv2.INTER_LINEAR)

            data = np.concatenate([data, msk_this[:, :, np.newaxis]], -1)
            
            bone.append(torch.tensor(data))

    except:
        for sid in range(n_slice_per_c):
            bone.append(torch.ones((image_size_cls, image_size_cls, n_ch+1)).int())
    
    cropped_images[cid] = torch.stack(bone, 0)




def runs(model, fold):
    outputs = []
    df_seg = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))
    valid_ = df_seg[df_seg['fold'] == fold].reset_index(drop=True)

    dataset_seg = SEGOutputDatasetALL(valid_, d)
    loader_seg = torch.utils.data.DataLoader(dataset_seg, 
        batch_size=1, shuffle=False, num_workers=num_workers)

    bar = tqdm(loader_seg)
    with torch.no_grad():
        for batch_id, (images) in enumerate(bar):
            images = images.cuda()

            # SEG
            pred_masks = []
            pmask = model(images).sigmoid()
            pred_masks.append(pmask)
            pred_masks = torch.stack(pred_masks, 0).mean(0).cpu().numpy()
            
            

            # Build cls input
            cls_inp = []
            threads = [None] * 5
            cropped_images = [None] * 5

            for i in range(pred_masks.shape[0]):
                row = valid_.iloc[batch_id*batch_size_seg+i]
                
                psid = row['png_suffix'].split('/')[-1]

                def load_cropped_images(msk, d_psid, png_suffix, n_ch=n_ch):                    
                    p_paths = d_psid # png paths

                    for cid in range(5): # cid : 5 organ (0,1,2,3,4)
                        threads[cid] = threading.Thread(
                            target=load_bone, args=(msk, cid, p_paths, cropped_images))
                        threads[cid].start()
                    for cid in range(5):
                        threads[cid].join()

                    return torch.cat(cropped_images, 0)
                
                cropped_images = load_cropped_images(pred_masks[i], d[psid], row['png_suffix']) # np.uint8
                cropped_images = cropped_images.permute(0, 3, 1, 2)
                cropped_images = cropped_images.detach().cpu().numpy()
                
                for j in range(5): # [liver, spleen, left kidney, right kidney, bowel]
                    output_name = os.path.join(output_dir, names_2[j+1], f"{psid}.npy")
                    start, end = j * 15, j * 15 + 15
                    if j == 4:
                        cropped_image = cropped_images[start:]
                    else:
                        cropped_image = cropped_images[start:end]
                    
                    # if DEBUG:
                    #     print(j, cropped_image.shape)

                    np.save(output_name, cropped_image) # save segmented volumes
                    
            if DEBUG:
                if batch_id == 3:
                    break




if __name__ == "__main__":
    print('length of png folder (nums of series_id) :' ,len(list(d.keys())))
    models_seg = []

    kernel_type = 'test'
    # 3D segmentation model directory
    model_dir_seg = f"../results/segmentations" 

    n_blocks = 4
    for fold in range(5):
        model = TimmSegModel(backbone, pretrained=False)
        model = convert_3d(model)
        model = model.to(device)
        load_model_file = os.path.join(model_dir_seg, kernel_type, 'model', f'test_fold{fold}_best.pth')
        sd = torch.load(load_model_file)
        if 'model_state_dict' in sd.keys():
            sd = sd['model_state_dict']
        sd = {k[7:] if k.startswith('module.') else k: sd[k] for k in sd.keys()}
        model.load_state_dict(sd, strict=True)
        model.eval()
        models_seg.append(model)

    print('nums of models :',len(models_seg))

    output_dir = segmented_dir

    os.makedirs(output_dir, exist_ok=True)
    names_2 = {}
    for k, organ in names.items():
        if 'kidney' in organ:
            organ = organ.replace(' ', '_')
        os.makedirs(os.path.join(output_dir, organ), exist_ok=True)
        if 'bowel' in organ:
            os.makedirs(os.path.join(output_dir, 'bowel_slices'), exist_ok=True)
        
        names_2[k] = organ
    print(names_2)


    # runs    
    for fold in range(5):
        print('fold :', fold)
        runs(models_seg[fold], fold)

        if DEBUG:
            break
