import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from glob import glob
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.utils.losses import DiceLoss


import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

seed = 42

DEBUG = False


# Config
kernel_type = 'extra-feat-sliding-seg'
load_kernel = None
load_last = True

n_folds = 5
backbone = 'seresnext50_32x4d'

image_size = 384
in_chans = 5

init_lr = 0.0001
eta_min = 0
batch_size = 64
drop_rate = 0.
drop_rate_last = 0.3
drop_path_rate = 0.
p_mixup = 0.5
p_rand_order_v1 = 0.2

use_amp = True
num_workers = 16
out_dim = 1 # [injury] binary classification

decoder_channels = (256, 128, 64, 32, 16)
# decoder_channels = (256, 160, 64, 48, 24)

seg_head_input = 32

class_weights = [6.] # [injury] (pos_weight)

n_epochs = 3

result_dir = '../results/stage2-extra-type1-seg'
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
mask_dir = os.path.join(data_dir, 'extra_mask')

segmented_dir = os.path.join(data_dir, "segmented")

feature_extracted_dir = os.path.join(data_dir, f'feature_extracted')
os.makedirs(feature_extracted_dir, exist_ok=True)

# transforms
transforms_train = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.RandomBrightness(limit=0.1, p=0.7),
    albumentations.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),

    albumentations.OneOf([
        albumentations.MotionBlur(blur_limit=3),
        albumentations.MedianBlur(blur_limit=3),
        albumentations.GaussianBlur(blur_limit=3),
        albumentations.GaussNoise(var_limit=(3.0, 9.0)),
    ], p=0.5),
    albumentations.OneOf([
        albumentations.OpticalDistortion(distort_limit=1.),
        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.5),

    albumentations.Cutout(
        max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), 
        num_holes=10, p=0.5),
])

transforms_valid = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
])



# Dataset
class CLSExtraDataset(Dataset):
    """
        for feature extractor for extravasation
    """
    def __init__(self, df, mode, transform):
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def load_file(self, path, png_number, start, end, n):
        if (png_number + n < 0) or (png_number + n > end):
            x = cv2.imread("_".join(path.split('_')[:-1]) + f"_{png_number}.png", 0)
        else:
            x = cv2.imread("_".join(path.split('_')[:-1]) + f"_{png_number+n}.png", 0)
        return x


    def __getitem__(self, index):
        row = self.df.iloc[index]
        psid = row['psid']
        path = row['png_folder']
        
        study_label = row['injury']
        image_label = row['image_label']

        start, end = row['png_start'], row['png_end']
        png_number = int(row['png_number'])
        
        x0 = self.load_file(path, png_number, start, end, n=-2)
        x1 = self.load_file(path, png_number, start, end, n=-1)
        x2 = cv2.imread(path, 0)
        x3 = self.load_file(path, png_number, start, end, n=+1)
        x4 = self.load_file(path, png_number, start, end, n=+2)

        x0 = np.expand_dims(x0, axis=2)
        x1 = np.expand_dims(x1, axis=2)
        x2 = np.expand_dims(x2, axis=2)
        x3 = np.expand_dims(x3, axis=2)
        x4 = np.expand_dims(x4, axis=2)
        image = np.concatenate([x0, x1, x2, x3, x4], axis=-1)

        mask_file = os.path.join(mask_dir, f"{row['psid']}_{png_number}.npy")
        if os.path.isfile(mask_file):
            mask = np.load(mask_file)
        else:
            mask = np.zeros((512,512), dtype=np.uint8)
            
        record = self.transform(image=image, mask=mask)
        image, mask = record['image'], record['mask']
        image = image.transpose(2,0,1).astype(np.float32) / 255.
        mask = mask.astype(np.float32)

        if self.mode != 'test':
            image = torch.tensor(image).float()
            mask = torch.tensor(mask).float()
            study_label = torch.tensor([study_label]).float()
            image_label = torch.tensor([image_label]).float()

            if self.mode == 'train' and random.random() < p_rand_order_v1:
                indices = torch.randperm(image.size(0)) # mix order of 5 images
                image = image[indices]

            return image, mask, study_label, image_label
        else:
            return torch.tensor(image).float() # (30, 6, sz, sz)



# Model
class TimmFeatExtractorSeg(nn.Module):
    def __init__(self, backbone, decoder_channels, pretrained=False):
        super(TimmFeatExtractorSeg, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans, # 3
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            output_dim = 1280
        elif 'seresnext50_32x4d' in backbone:
            output_dim = self.encoder.feature_info.channels()[-1]

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.feature_info.channels(),
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.seg_head = SegmentationHead(
            in_channels=seg_head_input,
            out_channels=1,
            activation=None,
            kernel_size=3
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.study_linear = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, out_dim)
        )
        self.image_linear = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, out_dim)
        )
        
        
    def forward(self, x): 
        features = self.encoder(x) 
        xseg = self.decoder(*features)
        xseg = self.upsample(xseg)
        xseg = self.seg_head(xseg) # (bs, 1, 384, 384)
        xseg = xseg.squeeze(1)

        pool = self.pool(features[-1])
        pool = pool.squeeze().squeeze()

        study_logit = self.study_linear(pool)
        image_logit = self.image_linear(pool)
        return study_logit, image_logit, xseg




bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(class_weights).cuda())

def criterion(logits, targets):
    losses = bce(logits.view(-1), targets.view(-1))
    return losses


class CustomDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()             
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
seg_criterion = CustomDiceLoss()


def mixup(input, mask, study_labels, image_labels, clip=[0, 1]):
    indices = torch.randperm(input.size(0)) # bs
    shuffled_input = input[indices]
    shuffled_mask = mask[indices]
    shuffled_study_labels = study_labels[indices]
    shuffled_image_labels = image_labels[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return (
        input, shuffled_mask,
        study_labels, shuffled_study_labels, 
        image_labels, shuffled_image_labels, 
        lam
    )


def train_func(model, loader_train, optimizer, scheduler, scaler=None):
    model.train()
    train_losses = []
    study_losses = []
    image_losses = []
    seg_losses = []
    bar = tqdm(loader_train)
    for images, mask, study_labels, image_labels in bar:
        optimizer.zero_grad()
        images = images.cuda()
        mask = mask.cuda()
        study_labels = study_labels.cuda()
        image_labels = image_labels.cuda()
        
        do_mixup = False
        if random.random() < p_mixup:
            do_mixup = True
            mixed_up = mixup(images, mask, study_labels, image_labels, clip=[0, 1])
            images, shuffled_mask, study_labels, study_labels_mix, image_labels, image_labels_mix, lam = mixed_up

        with amp.autocast():
            study_logits, image_logits, seg_logits = model(images) # (bs, 15)
            study_loss = criterion(study_logits, study_labels)
            image_loss = criterion(image_logits, image_labels)
            seg_loss = seg_criterion(seg_logits, mask)
            
            if do_mixup:
                study_loss11 = criterion(study_logits, study_labels_mix)
                image_loss11 = criterion(image_logits, image_labels_mix)
                seg_loss11 = seg_criterion(seg_logits, shuffled_mask)
                study_loss = study_loss * lam  + study_loss11 * (1 - lam)
                image_loss = image_loss * lam  + image_loss11 * (1 - lam)
                seg_loss = seg_loss * lam + seg_loss11 * (1-lam)
            
            loss = study_loss * 0.3 + image_loss * 0.3 + seg_loss * 0.4

        train_losses.append(loss.item())
        study_losses.append(study_loss.item())
        image_losses.append(image_loss.item())
        seg_losses.append(seg_loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]

        bar.set_description(f'lr:{lr:.8f} / smth:{np.mean(train_losses[-30:]):.4f} / study:{np.mean(study_losses[-30:]):.4f} / image:{np.mean(image_losses[-30:]):.4f} / seg:{np.mean(seg_losses[-30:]):.4f}')

    return np.mean(train_losses), np.mean(study_losses), np.mean(image_losses), np.mean(seg_losses)


def valid_func(model, loader_valid):
    model.eval()

    valid_losses, study_losses, image_losses, seg_losses = [], [], [], []
    
    study_gts, image_gts = [], []

    study_outputs, image_outputs = [], []

    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, mask, study_labels, image_labels in bar:
            images = images.cuda()
            mask = mask.cuda()
            study_labels = study_labels.cuda()
            image_labels = image_labels.cuda()

            study_logits, image_logits, seg_logit = model(images) 
            study_loss = criterion(study_logits, study_labels)
            image_loss = criterion(image_logits, image_labels)
            seg_loss = seg_criterion(seg_logit, mask)
            
            study_gts.append(study_labels.cpu())
            image_gts.append(image_labels.cpu())

            study_outputs.append(study_logits.cpu())
            image_outputs.append(image_logits.cpu())

            loss = study_loss * 0.3 + image_loss * 0.3 + seg_loss * 0.4

            valid_losses.append(loss.item())
            study_losses.append(study_loss.item())
            image_losses.append(image_loss.item())
            seg_losses.append(seg_loss.item())
            
            bar.set_description(f'smth:{np.mean(valid_losses[-30:]):.4f} / study:{np.mean(study_losses[-30:]):.4f} / image:{np.mean(image_losses[-30:]):.4f} / seg:{np.mean(seg_losses[-30:]):.4f}')

    study_outputs = torch.cat(study_outputs)
    image_outputs = torch.cat(image_outputs)
    study_gts = torch.cat(study_gts)
    image_gts = torch.cat(image_gts)

    valid_loss = np.mean(valid_losses)
    study_loss = criterion(study_outputs.cuda(), study_gts.cuda()).item()
    image_loss = criterion(image_outputs.cuda(), image_gts.cuda()).item()
    print(f'valid loss : {valid_loss}')
    print(f'study loss : {study_loss}')
    print(f'image loss : {image_loss}')
    print(f'seg   loss : {np.mean(seg_losses)}')

    return valid_loss, study_loss, image_loss, np.mean(seg_losses)


def run(df, df2, fold):

    log_file = os.path.join(log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')

    print('df  positive psid :', df[df['injury']==1]['psid'].nunique())
    print('df2 positive psid :', df2['psid'].nunique())

    train_ = df[df['fold'] != fold].reset_index(drop=True)
    # for upsampling
    train_2 = df2[df2['fold']!=fold].reset_index(drop=True)
    train_ = pd.concat([train_, train_2], axis=0).reset_index(drop=True)
    
    _zero = train_[train_['image_label']==0]
    _one = train_[train_['image_label']==1]
    _one = pd.concat([_one]*5)
    train_ = pd.concat([_zero, _one], axis=0).reset_index(drop=True)

    valid_ = df[df['fold'] == fold].reset_index(drop=True)

    print('train image_label')
    print(train_['image_label'].value_counts())
    print('valid image_label')
    print(valid_['image_label'].value_counts())
    
    dataset_train = CLSExtraDataset(train_, 'train', transform=transforms_train)
    dataset_valid = CLSExtraDataset(valid_, 'valid', transform=transforms_valid)
    
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TimmFeatExtractorSeg(backbone, decoder_channels, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    metric_best = np.inf
    loss_min = np.inf

    T_0 = n_epochs * len(loader_train)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0, eta_min=eta_min)

    print(len(dataset_train), len(dataset_valid))
    print('iter :', T_0)

    for epoch in range(1, n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)

        train_loss, train_s_loss, train_i_loss, train_seg_loss = train_func(model, loader_train, optimizer, scheduler_cosine, scaler)
        valid_loss, valid_s_loss, valid_i_loss, valid_seg_loss = valid_func(model, loader_valid)
        metric = valid_i_loss

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
        train_content = f"\t\t\ttrain loss: {train_loss:.6f}, study loss: {train_s_loss:.6f}, image loss: {train_i_loss:.6f}, seg loss: {train_seg_loss:.6f}"
        valid_content = f"\t\t\tvalid loss: {valid_loss:.6f}, study loss: {valid_s_loss:.6f}, image loss: {valid_i_loss:.6f}, seg loss: {valid_seg_loss:.6f}"
        print(content)
        print(train_content)
        print(valid_content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')
            appender.write(train_content + '\n')
            appender.write(valid_content + '\n')

        if metric < metric_best:
            print(f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
#             if not DEBUG:
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


def save_npy(df, fold, folder):
    out_path = os.path.join(data_dir, f'feature_extracted/sliding_{in_chans}')
    print('embedding save directory :', out_path)
    os.makedirs(out_path, exist_ok=True)

    valid_ = df[df['fold'] == fold].reset_index(drop=False)
    dataset_valid = CLSExtraDataset(valid_, 'valid', transform=transforms_valid)
    
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TimmFeatExtractorSeg(backbone, decoder_channels, pretrained=False)
    model_file = os.path.join(folder, "model", f"extra-feat-sliding-seg_fold{fold}_best.pth")
    print(f"load trained model : {model_file}")
    weight = torch.load(model_file)
    model.load_state_dict(weight)
    model = model.to(device)
    model.eval()

    bar = tqdm(enumerate(loader_valid), total=len(loader_valid))
    with torch.no_grad():
        for i, (images, mask, study_labels, image_labels) in bar:
            images = images.cuda()
            mask = mask.cuda()

            feats = model.encoder(images)
            feat = model.pool(feats[-1]).squeeze().squeeze().detach().cpu().numpy() # (bs, 1280)

            start = i*batch_size
            end = batch_size*(i+1) if len(images) == batch_size else i*batch_size+len(images)
            
            for k, j in enumerate(range(start, end)):
                row = valid_.iloc[j]
                psid = row['psid']
                path = row['png_folder']
                path = path.split('/')[-1].split('.')[0]

                np.save(os.path.join(out_path, f"{path}.npy"), feat[k])
            

def evaluation(df, model_folder):

    valid = []
    for fold in range(5):
        valid_ = df[df['fold'] == fold].reset_index(drop=True)

        dataset_valid = CLSExtraDataset(valid_, 'valid', transform=transforms_valid)
        
        loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = TimmFeatExtractorSeg(backbone, decoder_channels, pretrained=False)
        path = os.path.join(model_folder, 'model', f'extra-feat-sliding-seg_fold{fold}_best.pth')
        weight = torch.load(path)
        model.load_state_dict(weight)
        model = model.to(device)
        model.eval()
        
        image_outputs = []

        bar = tqdm(loader_valid)
        with torch.no_grad():
            for images, mask, study_labels, image_labels in bar:
                images = images.cuda()
                mask = mask.cuda()
                study_labels = study_labels.cuda()
                image_labels = image_labels.cuda()

                study_logits, image_logits, seg_logit = model(images) 
                
                image_outputs.append(image_logits.cpu())

        image_outputs = torch.cat(image_outputs)
        valid_['image_outputs'] = image_outputs
        valid.append(valid_)


        del model
        torch.cuda.empty_cache()
        gc.collect()
    valid = pd.concat(valid, axis=0)
    return valid


if __name__ == "__main__":
    name = os.path.join(data_dir, f"extra_sliding_{in_chans}_bbox.csv")
    name2 = os.path.join(data_dir, f'extra_sliding_{in_chans}_bbox_pos.csv')
    df = pd.read_csv(name)
    df2 = pd.read_csv(name2)
    print(f'load --- {name} ---')
    print(f'load --- {name2} ---')

    df = df.sample(12800).reset_index(drop=True) if DEBUG else df
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='train', help='training or get embedding')
    args = parser.parse_args()


    if args.run_type == 'train':
        run(df, df2, 0)
        run(df, df2, 1)
        run(df, df2, 2)
        run(df, df2, 3)
        run(df, df2, 4)

    if args.run_type == 'get_emb':
        folder = "../results/stage2-extra-type1-seg/extra-feat-sliding-seg"
        save_npy(df, 0, folder)
        save_npy(df, 1, folder)
        save_npy(df, 2, folder)
        save_npy(df, 3, folder)
        save_npy(df, 4, folder)
        
    if args.run_type == 'valid':
        model_folder = ""
        evaluation(df, model_folder)

