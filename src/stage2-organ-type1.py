"""
    small version (to gold zone) : effv2s, 224x224 input
    predicted masks to crop out.

    train a 2.5D classification with LSTM (type 1)

    -----

    kidney (left and right) -> resize and concat


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
kernel_type = 'organ'
load_kernel = None
load_last = True

n_folds = 5
backbone = 'tf_efficientnetv2_s_in21ft1k' # 'seresnext50_32x4d'

image_size = 224 
n_slice_per_c = 15
in_chans = 6

init_lr = 0.0001
eta_min = 0
batch_size = 8
drop_rate_last = 0.3
drop_rate = 0.
drop_path_rate = 0.

p_mixup = 0.5
p_rand_order_v1 = 0.2

use_amp = True
num_workers = 16
out_dim = 3 # [healthy, low, high]

class_weights = [1.,2.,4.] # [healthy, low, high]

n_epochs = 20

result_dir = '../results/stage2-organ-type1'
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

segmented_dir = os.path.join(data_dir, "segmented")

names = {    
    1 : "liver",
    2 : "spleen",
    3 : "left_kidney",
    4 : "right_kidney",
    5 : "bowel",
}
liver_dir = os.path.join(segmented_dir, 'liver')
spleen_dir = os.path.join(segmented_dir, 'spleen')
l_kidney_dir = os.path.join(segmented_dir, 'left_kidney')
r_kidney_dir = os.path.join(segmented_dir, 'right_kidney')


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

    albumentations.Cutout(max_h_size=int(image_size * 0.5), max_w_size=int(image_size * 0.5), num_holes=1, p=0.5),
])

transforms_valid = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
])


# Dataset
class CLSDataset(Dataset):
    """
        only for segmented organ (4)
    """
    def __init__(self, df, mode, transform):
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        psid = row['psid']
        organ = row['organ']
        
        labels = row[['healthy','low','high']].values.astype(float) # e.g. [0, 1, 0]

        images = []

        if organ == 'kidney':
            l = os.path.join(segmented_dir, f'left_{organ}/{psid}.npy') # int32
            r =  os.path.join(segmented_dir, f'right_{organ}/{psid}.npy')
            l_images = np.load(l).astype(np.uint8) # [15,6,sz,sz]
            r_images = np.load(r).astype(np.uint8) # [15,6,sz,sz]
            for ind in list(range(n_slice_per_c)): # 15
                l_image = l_images[ind].transpose(1,2,0)
                r_image = r_images[ind].transpose(1,2,0)
                l_image = cv2.resize(l_image, (image_size//2, image_size), interpolation=cv2.INTER_LINEAR)
                r_image = cv2.resize(r_image, (image_size//2, image_size), interpolation=cv2.INTER_LINEAR)
                image = np.concatenate([l_image, r_image], axis=1)
                image = self.transform(image=image)['image']
                image = image.transpose(2,0,1).astype(np.float32) / 255.
                images.append(image)
        else: # liver, spleen
            file_path = os.path.join(segmented_dir, f'{organ}/{psid}.npy')
            file = np.load(file_path).astype(np.uint8) # [15,6,sz,sz]
            for ind in list(range(n_slice_per_c)): # 15
                image = file[ind].transpose(1,2,0)
                image = self.transform(image=image)['image']
                image = image.transpose(2,0,1).astype(np.float32) / 255.
                images.append(image)

        images = np.stack(images, 0) # (15, 6, sz, sz)

        if self.mode != 'test':
            images = torch.tensor(images).float()
            labels = torch.tensor([labels] * n_slice_per_c).float()
            
            if self.mode == 'train' and random.random() < p_rand_order_v1:
                indices = torch.randperm(images.size(0)) # mix order of 15 images
                images = images[indices]

            return images, labels # torch.Size([15, 6, 224, 224]) torch.Size([15, 3])
        else:
            return torch.tensor(images).float() # (15, 6, sz, sz)


# Model
class TimmModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=out_dim,
            features_only=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        elif 'seresnext' in backbone or 'resnet' in backbone:
            hdim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()


        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, n_slice_per_c, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        feat = self.head(feat) # (2 * n_slice_per_c, 3)
        feat = feat.view(bs, n_slice_per_c, out_dim).contiguous()
        return feat # (bs, 15, 3)


# Loss & Metric
ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().cuda(), reduction='none')
ce2 = nn.CrossEntropyLoss(reduction='none')

def criterion(preds, targets, loss_type):
    """
        preds: (bs, 15, 3) (after softmax)
        targets: (bs, 15, 3)
    """
    if loss_type == 'ce':
        losses = ce(preds.view(-1, out_dim), targets.view(-1, out_dim)) # [bs * n_slice_per_c]
        return losses.mean()

    elif loss_type == 'll':
        losses = ce2(preds.view(-1, out_dim), targets.view(-1, out_dim)) # [bs * n_slice_per_c]
        w = torch.tensor(class_weights).repeat(len(targets), n_slice_per_c, 1).float().cuda()
        sw = torch.max((targets * w), axis=-1)[0] # (bs, n_slice_per_c)        
        losses *= sw.view(-1)
        return losses.sum() / sw.sum()

    elif loss_type == 'normal_loss':
        losses = ce2(preds.view(-1, out_dim), targets.view(-1, out_dim)) # [bs * n_slice_per_c]
        return losses.mean()


# train & valid func
def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam


def train_func(model, loader_train, optimizer, loss_type, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images, targets in bar:
        optimizer.zero_grad()
        images = images.cuda() # (15, 6, sz, sz)
        targets = targets.cuda()
        
        do_mixup = False
        if random.random() < p_mixup:
            do_mixup = True
            images, targets, targets_mix, lam = mixup(images, targets)

        with amp.autocast():
            logits = model(images) # (bs, 15, 3)
            loss = criterion(logits.cuda(), targets.cuda(), loss_type)

            if do_mixup:
                loss11 = criterion(logits.cuda(), targets.cuda(), loss_type)
                loss = loss * lam  + loss11 * (1 - lam)
        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid, loss_type):
    model.eval()
    valid_loss = []
    gts = []
    outputs = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, targets in bar:
            images = images.cuda()
            targets = targets.cuda()

            logits = model(images)
            loss = criterion(logits.cuda(), targets.cuda(), loss_type)
            
            gts.append(targets.cpu())
            outputs.append(logits.cpu())
            valid_loss.append(loss.item())
            
            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')

    outputs = torch.cat(outputs)
    gts = torch.cat(gts)
    valid_loss = criterion(outputs.cuda(), gts.cuda(), loss_type).item()
    
    print('ce_loss', criterion(outputs.cuda(), gts.cuda(), 'ce'))
    print('ll_loss', criterion(outputs.cuda(), gts.cuda(), 'll'))
    print('nl_loss', criterion(outputs.cuda(), gts.cuda(), 'normal_loss'))
    
    return valid_loss


def run(df, fold, loss_type):

    log_file = os.path.join(log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')

    train_ = df[df['fold'] != fold].reset_index(drop=True)
    valid_ = df[df['fold'] == fold].reset_index(drop=True)
    dataset_train = CLSDataset(train_, 'train', transform=transforms_train)
    dataset_valid = CLSDataset(valid_, 'valid', transform=transforms_valid)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TimmModel(backbone, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    metric_best = np.inf
    loss_min = np.inf

    with open(log_file, 'a') as appender:
        appender.write('\n'+kernel_type+'\t'+backbone+'\t'+str(image_size)+'\n')

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_epochs, eta_min=eta_min)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, n_epochs+1):
        scheduler_cosine.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, loss_type, scaler)
        valid_loss = valid_func(model, loader_valid, loss_type)
        metric = valid_loss

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if metric < metric_best:
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



def evalutation(df, folder, save=False):
    valid = []
    for fold in range(5):
        print(f"fold {fold} evaluation")
        if os.path.isdir(os.path.join(folder, 'model')):
            model_file = os.path.join(folder, 'model', f'organ_fold{fold}_best.pth')
        else:
            model_file = os.path.join(folder, f'organ_fold{fold}_best.pth')
        print(model_file)
        valid_ = df[df['fold'] == fold].reset_index(drop=True)
        
        dataset_valid = CLSDataset(valid_, 'valid', transform=transforms_valid)
        loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = TimmModel(backbone, pretrained=False)
        weight = torch.load(model_file)
        model.load_state_dict(weight)
        model = model.to(device)
        model.eval()

        gts = []
        outputs = []

        bar = tqdm(loader_valid, total=len(loader_valid))
        with torch.no_grad():
            for images, targets in bar:
                images = images.cuda()
                targets = targets.cuda()

                logits = model(images)
                
                gts.append(targets.cpu())
                outputs.append(F.softmax(logits, dim=-1).cpu())
                
        outputs = torch.cat(outputs)
        gts = torch.cat(gts)
        
        outputs = torch.mean(outputs, dim=1)
        valid_['healthy'] = outputs[:,0]
        valid_['low'] = outputs[:,1]
        valid_['high'] = outputs[:,2]

        valid.append(valid_)
    valid = pd.concat(valid).reset_index(drop=True)
    if save:
        valid.to_csv(os.path.join(folder, 'organ_valid.csv'), index=False)
        print('save organ validation')
    return valid
    



if __name__ == "__main__":
    df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))
    df['psid'] = df['patient_id'].astype(str) + '_' + df['series_id'].astype(str)
    df = df.sample(16).reset_index(drop=True) if DEBUG else df

    psid = []
    organ = []
    healthy = []
    low = []
    high = []
    fold = []
    for _, row in df.iterrows():
        for o in ['liver','spleen','kidney']:
            psid.append(row['psid'])
            organ.append(o)
            healthy.append(row[o + '_healthy'])
            low.append(row[o + '_low'])
            high.append(row[o + '_high'])
            fold.append(row['fold'])

    df = pd.DataFrame({
        'psid':psid,
        'organ':organ,
        'healthy':healthy,
        'low':low,
        'high':high,
        'fold':fold
    })
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_type', type=str, default='ll', help='loss type by folds') # ['normal', 'ce', 'll]
    parser.add_argument('--fold', type=int, default='5', help='select train folds with loss_type')
    parser.add_argument('--run_type', type=str, default='train', help='train or valid')
    args = parser.parse_args()

    # training
    if args.run_type == 'train':
        if args.fold == 5:
            print(f'training kfolds / loss_type : {args.loss_type}')
            run(df, 0, args.loss_type)
            run(df, 1, args.loss_type)
            run(df, 2, args.loss_type)
            run(df, 3, args.loss_type)
            run(df, 4, args.loss_type)
        else:
            print(f'training {args.fold} fold / loss_type : {args.loss_type}')
            run(df, args.fold, args.loss_type)
            
    # evaluation
    if args.run_type == 'valid':
        # model folder
        folder = "../results/stage2-organ-type1/organ" # default
        # folder = "../results/stage2-organ-type1/effv2s"
        # folder = "../results/stage2-organ-type1/seresnext50_32x4d"; backbone = 'seresnext50_32x4d'
        valid = evalutation(df, folder, save=True)
