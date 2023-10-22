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
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

seed = 42

DEBUG = True

# Config
kernel_type = 'bowel'
load_kernel = None
load_last = True

n_folds = 5
backbone = 'tf_efficientnetv2_s_in21ft1k'

image_size = 224
n_slice_per_c = 30
in_chans = 6

init_lr = 0.0001
eta_min = 0 
batch_size = 8
drop_rate = 0.
drop_rate_last = 0.3
drop_path_rate = 0.
p_mixup = 0.5
p_rand_order_v1 = 0.2

use_amp = True
num_workers = 16
out_dim = 1 # [injury] binary classification

class_weights = [2.] # [injury] (pos_weight)

n_epochs = 10

removed = ['43_36714', '8684_50377', '60744_397']


result_dir = '../results/stage2-bowel-type1'
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

segmented_dir = os.path.join(data_dir, f"segmented")

names = {    
    1 : "liver",
    2 : "spleen",
    3 : "left_kidney",
    4 : "right_kidney",
    5 : "bowel",
}
bowel_dir = os.path.join(segmented_dir, 'bowel')
bowel_slices_dir = os.path.join(segmented_dir, 'bowel_slices')



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
class CLSBowelDataset(Dataset):
    """
        only for segmented Bowel 
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
        assert row['organ'] == 'bowel'
        
        study_labels = row['injury']
        if study_labels == 1 and row.label_type is not np.nan:
            inds = pickle.load(open(os.path.join(bowel_slices_dir, f"{psid}.pkl"), 'rb'))['inds'].astype(np.float32) 
            # inds : png number
            _min = row.label_min # dcm
            _max = row.label_max # dcm
            inds += row.start # make png_number to dcm_number
            image_labels = np.where((inds < _min) | (inds > _max), 0, 1)

        else:
            image_labels = np.zeros(n_slice_per_c,)
            
        images = []
        file_path = os.path.join(segmented_dir, f'bowel/{psid}.npy')
        file = np.load(file_path).astype(np.uint8) # [30,6,sz,sz]
        for ind in list(range(n_slice_per_c)): # 30
            image = file[ind].transpose(1,2,0)
            image = self.transform(image=image)['image']
            image = image.transpose(2,0,1).astype(np.float32) / 255.
            images.append(image)

        images = np.stack(images, 0) # (15, 6, sz, sz)

        if self.mode != 'test':
            images = torch.tensor(images).float()
            study_labels = torch.tensor([study_labels]).float()
            image_labels = torch.tensor(image_labels).float()
            
            if self.mode == 'train' and random.random() < p_rand_order_v1:
                indices = torch.randperm(images.size(0)) # mix order of 30 images
                images = images[indices]
                image_labels = image_labels[indices]

            return images, study_labels, image_labels
        else:
            return torch.tensor(images).float() # (30, 6, sz, sz)

# Model
class TimmBowelModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmBowelModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=out_dim, # 1
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
        self.image_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, out_dim)
        ) # output : (bs * n_slice_per_c, 1)

        self.study_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, out_dim)
        )


    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, n_slice_per_c, -1)
        feat, _ = self.lstm(feat)

        # image level
        image_feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        image_logit = self.image_head(image_feat) # (bs * n_slice_per_c, 1)
        image_logit = image_logit.view(bs, n_slice_per_c, out_dim).contiguous() # (bs, n_slice_per_c, 1)

        # study level
        avg_pool = torch.mean(feat, 1)   # (bs, 512)
        max_pool = torch.max(feat, 1)[0] # (bs, 512)
        study_feat = torch.cat((max_pool, avg_pool), 1) # (bs, 1024)
        study_logit = self.study_head(study_feat)
        
        return study_logit, image_logit



bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(class_weights).cuda())
bce2 = nn.BCEWithLogitsLoss(reduction='none')

def criterion(logits, targets):
    losses = bce(logits.view(-1), targets.view(-1))
    return losses

def criterion2(logits, targets): # for study
    losses = bce2(logits.view(-1), targets.view(-1))
    w = targets + 1
    losses *= w.view(-1)
    return losses.sum() / w.sum()



def mixup(input, study_labels, image_labels, clip=[0, 1]):
    indices = torch.randperm(input.size(0)) # bs
    shuffled_input = input[indices]
    shuffled_study_labels = study_labels[indices]
    shuffled_image_labels = image_labels[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return (
        input, 
        study_labels, shuffled_study_labels, 
        image_labels, shuffled_image_labels, 
        lam
    )


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_losses = []
    study_losses = []
    image_losses = []
    bar = tqdm(loader_train)
    for images, study_labels, image_labels in bar:
        optimizer.zero_grad()
        images = images.cuda()
        study_labels = study_labels.cuda()
        image_labels = image_labels.cuda()
        
        do_mixup = False
        if random.random() < p_mixup:
            do_mixup = True
            images, study_labels, study_labels_mix, image_labels, image_labels_mix, lam = mixup(images, study_labels, image_labels, clip=[0, 1])
        
        with amp.autocast():
            study_logits, image_logits = model(images) # (bs, 15)
            study_loss = criterion(study_logits, study_labels)
            image_loss = criterion(image_logits, image_labels)
            
            if do_mixup:
                study_loss11 = criterion(study_logits, study_labels_mix)
                image_loss11 = criterion(image_logits, image_labels_mix)
                study_loss = study_loss * lam  + study_loss11 * (1 - lam)
                image_loss = image_loss * lam  + image_loss11 * (1 - lam)
            
            loss = study_loss * 0.5 + image_loss * 0.5

        train_losses.append(loss.item())
        study_losses.append(study_loss.item())
        image_losses.append(image_loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'smth:{np.mean(train_losses[-30:]):.4f} / study:{np.mean(study_losses[-30:]):.4f} / image:{np.mean(image_losses[-30:]):.4f}')

    return np.mean(train_losses), np.mean(study_losses), np.mean(image_losses)


def valid_func(model, loader_valid):
    model.eval()

    valid_losses, study_losses, image_losses = [], [], []
    
    study_gts, image_gts = [], []

    study_outputs, image_outputs = [], []

    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, study_labels, image_labels in bar:
            images = images.cuda()
            study_labels = study_labels.cuda()
            image_labels = image_labels.cuda()

            study_logits, image_logits = model(images) 
            study_loss = criterion(study_logits, study_labels)
            image_loss = criterion(image_logits, image_labels)
            
            study_gts.append(study_labels.cpu())
            image_gts.append(image_labels.cpu())

            study_outputs.append(study_logits.cpu())
            image_outputs.append(image_logits.cpu())

            loss = study_loss * 0.5 + image_loss * 0.5

            valid_losses.append(loss.item())
            study_losses.append(study_loss.item())
            image_losses.append(image_loss.item())
            
            bar.set_description(f'smth:{np.mean(valid_losses[-30:]):.4f} / study:{np.mean(study_losses[-30:]):.4f} / image:{np.mean(image_losses[-30:]):.4f}')

    study_outputs = torch.cat(study_outputs)
    image_outputs = torch.cat(image_outputs)
    study_gts = torch.cat(study_gts)
    image_gts = torch.cat(image_gts)

    valid_loss = np.mean(valid_losses)
    study_loss = criterion(study_outputs.cuda(), study_gts.cuda()).item()
    image_loss = criterion(image_outputs.cuda(), image_gts.cuda()).item()
    ll_loss = criterion2(study_outputs.cuda(), study_gts.cuda()).item()
    print(f'\tvalid loss : {valid_loss}')
    print(f'\tstudy loss : {study_loss}')
    print(f'\timage loss : {image_loss}')
    print(f'\tlog   loss : {ll_loss}')
    return valid_loss, study_loss, image_loss


def run(df, fold):

    log_file = os.path.join(log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')

    train_ = df[df['fold'] != fold].reset_index(drop=True)
    valid_ = df[df['fold'] == fold].reset_index(drop=True)
    dataset_train = CLSBowelDataset(train_, 'train', transform=transforms_train)
    dataset_valid = CLSBowelDataset(valid_, 'valid', transform=transforms_valid)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TimmBowelModel(backbone, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    metric_best = np.inf
    loss_min = np.inf

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_epochs, eta_min=eta_min)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, n_epochs+1):
        scheduler_cosine.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss, train_s_loss, train_i_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss, valid_s_loss, valid_i_loss = valid_func(model, loader_valid)
        metric = valid_s_loss

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_s_loss:.5f}, valid loss: {valid_s_loss:.5f}, metric: {(metric):.6f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

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



def evalutation(df, folder, save=False):
    valid = []
    for fold in range(5):
        print(f"fold {fold} evaluation")

        if os.path.isdir(os.path.join(folder, 'model')): # default
            model_file = os.path.join(folder, 'model', f'bowel_fold{fold}_best.pth')
        else:
            model_file = os.path.join(folder, f'bowel_fold{fold}_best.pth')
        print(model_file)
        valid_ = df[df['fold'] == fold].reset_index(drop=True)
        
        dataset_valid = CLSBowelDataset(valid_, 'valid', transform=transforms_valid)
        loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = TimmBowelModel(backbone, pretrained=True)
        weight = torch.load(model_file)
        model.load_state_dict(weight)
        model = model.to(device)
        model.eval()

        study_outputs = []

        bar = tqdm(loader_valid)
        with torch.no_grad():
            for images, study_labels, image_labels in bar:
                images = images.cuda()

                study_logits, image_logits = model(images) 
                
                study_outputs.append(study_logits.sigmoid().detach().cpu().numpy())
                
        study_outputs = np.concatenate(study_outputs)
        
        valid_['preds'] = study_outputs

        valid.append(valid_)
    valid = pd.concat(valid).reset_index(drop=True)
    if save:
        valid.to_csv(os.path.join(folder, 'bowel_valid.csv'), index=False)
        print('save bowel validation')
    return valid





if __name__ == '__main__':
    # get only 'bowel image level label' from 'image_level_labels.csv'
    image_label = pd.read_csv(os.path.join(data_dir, 'dataset/image_level_labels.csv'))
    image_label['psid'] = image_label['patient_id'].astype(str) + '_' + image_label['series_id'].astype(str)
    bowel = image_label[image_label['injury_name']=='Bowel']
    bowel = bowel.groupby("psid")['instance_number'].agg(['min','max']).reset_index()
    bowel['label_type'] = 'bowel'
    extra = image_label[image_label['injury_name']=='Active_Extravasation']
    extra = extra.groupby("psid")['instance_number'].agg(['min','max']).reset_index()
    extra['label_type'] = 'extra'
    image_label = pd.concat([bowel, extra], axis=0)
    image_label = image_label.rename(columns={'min':'label_min','max':'label_max'})
    bowel_image_label = image_label[image_label['label_type']=='bowel']

    dcm_number = pd.read_csv(os.path.join(data_dir, "dcm_number.csv"))
    dcm_number['psid'] = dcm_number['patient_id'].astype(str) + '_' + dcm_number['series_id'].astype(str)

    df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))
    df['psid'] = df['patient_id'].astype(str) + '_' + df['series_id'].astype(str)

    psid = []
    organ = []
    healthy = []
    injury = []

    fold = []
    for _, row in df.iterrows():
        for o in ['bowel']:
            psid.append(row['psid'])
            organ.append(o)
            healthy.append(row[o + '_healthy'])
            injury.append(row[o + '_injury'])
            fold.append(row['fold'])
            
    df = pd.DataFrame({
        'psid':psid,
        'organ':organ,
        'healthy':healthy,
        'injury':injury,
        'fold':fold
    })

    df = df.merge(dcm_number[['psid','incomplete_organ','start','end']], how='left', on='psid')
    df = df.merge(bowel_image_label, how='left', on='psid')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='train', help='train or valid')
    args = parser.parse_args()

    if args.run_type == 'train':
        if DEBUG:
            df = df.sample(100).reset_index(drop=True)
        print(df.shape)
        run(df, 0)
        run(df, 1)
        run(df, 2)
        run(df, 3)
        run(df, 4)

    if args.run_type == 'valid':
        folder = "../results/stage2-organ-type1/bowel" # default
        # folder = "../results/stage2-bowel-type1/effv2s"
        # folder = "../results/stage2-bowel-type1/seresnext50_32x4d"; backbone = 'seresnext50_32x4d'
        valid = evalutation(df, folder, save=True)


        


