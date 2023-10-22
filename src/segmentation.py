""" segmentation.py

    this code for training '3D segmentation Model'

"""
import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
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

import argparse


DEBUG = False

# Config
kernel_type = 'test'
load_kernel = None
load_last = True
n_blocks = 4
n_folds = 5
backbone = 'resnet18d'

image_sizes = [128, 128, 128]
R = Resize(image_sizes)

init_lr = 1e-3 # 3e-3, 1e-3
batch_size = 4
drop_rate = 0.
drop_path_rate = 0.
loss_weights = [1,1]
p_mixup = 0.1

use_amp = True
num_workers = 16
out_dim = 5

n_epochs = 100

seed = 42

result_dir = '../results/segmentations'
exp_dir = os.path.join(result_dir, kernel_type)
log_dir = os.path.join(exp_dir, 'logs')
model_dir = os.path.join(exp_dir, 'model')

os.makedirs(exp_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


data_dir = '../data'
dcm_dir = os.path.join(data_dir, 'dataset/train_images')
seg_dir = os.path.join(data_dir, 'dataset/segmentations')
png_dir = os.path.join(data_dir, 'png_folder')

# Output dir
p = os.path.join(data_dir, 'segmentations')

d = pickle.load(open('../data/d.pkl', 'rb'))


transforms_train = transforms.Compose([
    transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
    transforms.RandAffined(
        keys=["image", "mask"], 
        translate_range=[int(x*y) for x, y in zip(image_sizes, [0.3, 0.3, 0.3])], 
        padding_mode='zeros', 
        prob=0.7
    ),
    transforms.RandGridDistortiond(
        keys=("image", "mask"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),    
])

transforms_valid = transforms.Compose([
])




def data_check(image, mask, row):
    fig, axes = plt.subplots(1,5,figsize=(20,4))
    i = 60
    sample = image.permute(1,2,0,3)[:,:,:,i]
    for k in range(5):
        axes[k].imshow(sample, alpha=0.7)
        axes[k].imshow(mask[k,:,:,i] * (k+1), alpha=0.3)
    plt.plot()
    # fig.savefig(f"data_check_{row['png_suffix'].split('/')[-1]}.png", dpi=100)
    fig.savefig(f"data_check.png", dpi=100)
    fig.clear()
    plt.close(fig)
    del fig, axes
    gc.collect()
    


def load_png_files(row, dcm_folder, png_suffix):
    """
        indices: select slices 
    """

    if os.path.isdir(dcm_folder):
        slices_numbers = sorted(os.listdir(dcm_folder), key=lambda x: int(x.split('/')[-1].split(".")[0]))
    else:
        psid = str(int(row['patient_id'])) + '_' + str(int(row['series_id']))
        slices_numbers = d[psid]
    
    n_scans = len(slices_numbers)
    indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., image_sizes[2])).round().astype(int)
    p_paths = [f"{png_suffix}_{i}.png" for i in indices] # by index

    images = []
    for filename in p_paths:
        data = cv2.imread(filename)
        data = cv2.resize(data, (image_sizes[0], image_sizes[1]), interpolation = cv2.INTER_LINEAR)
        images.append(data)
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images # (128, 128, 3, 128)


def load_sample(row, has_mask=True):
    image = load_png_files(row, row['dcm_folder'], row['png_suffix'])
    
    if image.ndim < 4:
        image = np.expand_dims(image, 0).repeat(3, 0) # to 3ch
    
    if has_mask:
        mask_org = nib.load(row['mask_file']).get_fdata()
        shape = mask_org.shape # (512, 512, 1022)
        
        mask_org = mask_org.transpose(1, 0, 2)[::-1, :, :]  # (d, w, h)
        
        mask = []
        for cid in range(5):
            mask.append((mask_org == (cid+1)))
        mask = np.stack(mask)
        mask = mask.astype(np.uint8) * 255
        mask = R(mask).numpy() # (5, 128, 128, 128)
        
        return image, mask
    else:
        return image
    

class SEGDataset(Dataset):
    """
        cache: 
            For training segmentation, need to convert .png files to 128 slices extracted and .nii files to mask for saving loading time.
            Because resizing xyz-axis by moani consume a lot of time.
            Initially running, load .png files and .nii files first, and convert these files.
            After running to tune 3D segmentation models repeatly, call saved 

            shape (e.g.)
                .png : (512, 512, 3)
                .nii : (512, 512, 64)

                images : (128, 128, 3, 128) ---> saved cache
                mask   : (5, 128, 128, 128) ---> saved cache
    """

    def __init__(self, df, mode, transform, slices_num=128):
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform
        self.slices_num = slices_num
        self.cache_folder = os.path.join(p, f"s_{slices_num}")

        self.makedirs()
    
    def makedirs(self):
        os.makedirs(self.cache_folder, exist_ok=True)
        os.makedirs(os.path.join(self.cache_folder, 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.cache_folder, 'mask'), exist_ok=True)
        print(f'make directory of {self.slices_num}')

    def check_cache(self, row):
        image_file = os.path.join(self.cache_folder, 'image', f"{row['png_suffix'].split('/')[-1]}.npy")
        mask_file = os.path.join(self.cache_folder, 'mask', f"{row['png_suffix'].split('/')[-1]}.npy")
        if os.path.isfile(image_file) and os.path.isfile(mask_file):
            return True
        else:
            return False
        
    def save_cache(self, row, image, mask):    
        save_img_path = os.path.join(p, f"s_{self.slices_num}", 'image')
        save_mask_path = os.path.join(p, f"s_{self.slices_num}", 'mask')
        save_img_name = os.path.join(save_img_path, f"{row['png_suffix'].split('/')[-1]}.npy")
        save_mask_name = os.path.join(save_mask_path, f"{row['png_suffix'].split('/')[-1]}.npy")
        img_save = np.save(save_img_name, image)
        mask_save = np.save(save_mask_name, mask)
        del img_save, mask_save, image, mask
        gc.collect()

    def load_cache(self, row):
        image_file = os.path.join(self.cache_folder, 'image', f"{row['png_suffix'].split('/')[-1]}.npy")
        mask_file = os.path.join(self.cache_folder, 'mask', f"{row['png_suffix'].split('/')[-1]}.npy")
        image = np.load(image_file).astype(np.float32)
        mask = np.load(mask_file).astype(np.float32)
        return image, mask

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.check_cache(row): # using local cache
            image, mask = self.load_cache(row)
        else:
            image, mask = load_sample(row, has_mask=True)
            self.save_cache(row, image, mask)
            gc.collect()
            print('save cache', end=' ')
        
        ### for transforms flip
        image = image.transpose(2,0,1,3) 

        res = self.transform({'image':image, 'mask':mask})
        image = res['image'] / 255.
        mask = res['mask']
        mask = (mask > 127).astype(np.float32)
        image, mask = torch.tensor(image).float(), torch.tensor(mask).float()

        return image, mask


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


# Loss & Metric

def binary_dice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: Optional[float] = None,
    nan_score_on_empty=False,
    eps: float = 1e-7,
) -> float:

    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    score = (2.0 * intersection) / (cardinality + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score


def multilabel_dice_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
):
    ious = []
    num_classes = y_pred.size(0)
    for class_index in range(num_classes):
        iou = binary_dice_score(
            y_pred=y_pred[class_index],
            y_true=y_true[class_index],
            threshold=threshold,
            nan_score_on_empty=nan_score_on_empty,
            eps=eps,
        )
        ious.append(iou)

    return ious


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def bce_dice(input, target, loss_weights=loss_weights):
    loss1 = loss_weights[0] * nn.BCEWithLogitsLoss()(input, target)
    loss2 = loss_weights[1] * dice_loss(input, target)
    return (loss1 + loss2) / sum(loss_weights)

criterion = bce_dice


# Train & Valid func
def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images, gt_masks in bar: 
        optimizer.zero_grad()
        images = images.cuda()
        gt_masks = gt_masks.cuda()

        do_mixup = False
        if random.random() < p_mixup:
            do_mixup = True
            images, gt_masks, gt_masks_sfl, lam = mixup(images, gt_masks)

        with amp.autocast():
            logits = model(images)
            loss = criterion(logits, gt_masks)
            if do_mixup:
                loss2 = criterion(logits, gt_masks_sfl)
                loss = loss * lam  + loss2 * (1 - lam)

        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    outputs = []
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    batch_metrics = [[]] * 7
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, gt_masks in bar:
            images = images.cuda()
            gt_masks = gt_masks.cuda()

            logits = model(images)
            loss = criterion(logits, gt_masks)
            valid_loss.append(loss.item())
            for thi, th in enumerate(ths):
                pred = (logits.sigmoid() > th).float().detach()
                for i in range(logits.shape[0]):
                    tmp = multilabel_dice_score(
                        y_pred=logits[i].sigmoid().cpu(),
                        y_true=gt_masks[i].cpu(),
                        threshold=0.5,
                    )
                    batch_metrics[thi].extend(tmp)
            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
            
    metrics = [np.mean(this_metric) for this_metric in batch_metrics]
    print('best th:', ths[np.argmax(metrics)], 'best dc:', np.max(metrics))

    return np.mean(valid_loss), np.max(metrics)


# Training
def run(df_seg, fold):

    log_file = os.path.join(log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')

    train_ = df_seg[df_seg['fold'] != fold].reset_index(drop=True)
    valid_ = df_seg[df_seg['fold'] == fold].reset_index(drop=True)
    dataset_train = SEGDataset(train_, 'train', transform=transforms_train)
    dataset_valid = SEGDataset(valid_, 'valid', transform=transforms_valid)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TimmSegModel(backbone, pretrained=True)
    model = convert_3d(model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scaler = torch.cuda.amp.GradScaler()
    from_epoch = 0
    metric_best = 0.
    loss_min = np.inf

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_epochs)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, n_epochs+1):
        scheduler_cosine.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss, metric = valid_func(model, loader_valid)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if metric > metric_best:
            print(f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_file)
            metric_best = metric

        # Save Last
        if not DEBUG:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'score_best': metric_best,
                },
                model_file.replace('_best', '_last')
            )

    del model
    torch.cuda.empty_cache()
    gc.collect()


def make_cache_files(df):
    df_show = df.copy()
    dataset_show = SEGDataset(
        df_show, 'train', transform=transforms_train, slices_num=128)

    loader_train = torch.utils.data.DataLoader(
        dataset_show, batch_size=1, shuffle=True, num_workers=4)
    
    for i, (image, mask) in enumerate(tqdm(loader_train, total=len(loader_train))):
        pass



if __name__ == "__main__":
    # DataFrame
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='train', help='run type for saving cache or training')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(data_dir, 'df_seg.csv'))
    if DEBUG:
        df = df.sample(20).reset_index(drop=True); print('debug')
    
    if args.run_type == 'save_cache':
        make_cache_files(df) # estimate : 160 min with num_workers=4 

    if args.run_type == 'train':
        run(df, 0)
        run(df, 1)
        run(df, 2)
        run(df, 3)
        run(df, 4)

    
        
