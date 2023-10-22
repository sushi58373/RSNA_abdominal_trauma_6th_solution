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
kernel_type = 'extra-feat-sliding-seg-seq'
load_kernel = None
load_last = True

n_folds = 5
backbone = 'seresnext50_32x4d' # 'tf_efficientnetv2_s_in21ft1k'

image_size = 384
in_chans = 5

init_lr = 0.0001
eta_min = 0
batch_size = 32
drop_rate = 0.
drop_rate_last = 0.3
drop_path_rate = 0.
p_mixup = 0.5
p_rand_order = 0.2

use_amp = True
num_workers = 16
out_dim = 1 # [injury] binary classification

class_weights = [6.] # [injury] (pos_weight)

n_epochs = 20

m_size = 192 # 96 128 160 192 224

study_ratio = 1.0

effv2s_dim = 1280
seres_dim = 2048

lstm_in_dim = seres_dim
lstm_size = 512



result_dir = '../results/stage2-extra-type1-seg'
exp_dir = os.path.join(result_dir, kernel_type)
log_dir = os.path.join(exp_dir, 'logs')
model_dir = os.path.join(exp_dir, 'model')

os.makedirs(exp_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

data_dir = f"../data"

dcm_dir = os.path.join(data_dir, 'dataset/train_images')
seg_dir = os.path.join(data_dir, 'dataset/segmentations')
png_dir = os.path.join(data_dir, 'png_folder')
emb_dir = os.path.join(data_dir, f"feature_extracted/sliding_{in_chans}") # embedding directory

print(emb_dir)

d = pickle.load(open('../data/d.pkl', 'rb'))




def rand_order(input, mask, image_labels, clip=[0,1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_mask = mask[indices]
    shuffled_labels = image_labels[indices]
    return input, mask, image_labels


# Dataset
class CLSExtraSeqDataset(Dataset):
    """
        for feature extractor for extravasation
    """
    def __init__(self, df, psids, mode):
        self.df = df.reset_index()
        self.psids = psids
        self.mode = mode

    def __len__(self):
        return len(self.psids)

    def __getitem__(self, index):
        psid = self.psids[index]
        tmp = self.df[self.df['psid']==psid].reset_index(drop=True)
        study_label = tmp['injury'].values[0]

        embs = []
        image_labels = []
        
        for i, row in tmp.iterrows():
            name = row['png_folder'].split('/')[-1].split('.')[0]
            name = os.path.join(emb_dir, f"{name}.npy")
            emb = np.load(name)
            embs.append(emb)

            image_labels.append(row['image_label'])
            
        embs = np.stack(embs)
        image_labels = np.stack(image_labels).astype(np.float32)
        
        if len(embs) < m_size:
            pad_sz = (m_size - len(embs))
            pad = np.zeros((pad_sz, embs.shape[-1]))
            mask = np.concatenate([np.zeros(pad_sz,), np.ones(len(embs),)])
            image_labels = np.concatenate([np.zeros(pad_sz,), image_labels])
            embs = np.concatenate([pad, embs])
        else: # resize
            embs = cv2.resize(embs, (embs.shape[-1], m_size), interpolation=cv2.INTER_LINEAR)
            image_labels = cv2.resize(image_labels, (1, m_size), interpolation = cv2.INTER_LINEAR)
            mask = np.ones(len(embs),)

        diff1 = np.zeros_like(embs)
        diff1[1:] = embs[1:] - embs[:-1]
        diff2 = np.zeros_like(embs)
        diff2[:-1] = embs[:-1] - embs[1:]
        
        embs = np.concatenate([embs, diff1, diff2], axis=-1)

        study_label = np.array([study_label])
        image_labels = image_labels.squeeze()

        embs = torch.tensor(embs).float()
        mask = torch.tensor(mask).float()
        study_label = torch.tensor(study_label).float()
        image_labels = torch.tensor(image_labels).float()

        # sampling sequence
        if self.mode == 'train' and random.random() < p_rand_order:
            indices = torch.randperm(embs.size(0))
            embs = embs[indices]
            mask = mask[indices]
            image_labels = image_labels[indices]

        # embs : (m_size, emb_size * 3)

        return embs, mask, study_label, image_labels


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        # x.shape 1024
        feature_dim = self.feature_dim # 1024
        step_dim = self.step_dim # 192 (m_size)
        
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
        

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class SeqModel(nn.Module):
    def __init__(self):
        super(SeqModel, self).__init__()
        self.lstm1 = nn.GRU(lstm_in_dim*3, lstm_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(lstm_size*2, lstm_size, bidirectional=True, batch_first=True)
        self.image_linear = nn.Linear(lstm_size*2, 1)

        self.study_linear = nn.Sequential(
            nn.Linear(lstm_size*4, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1)
        )

        self.attention = Attention(lstm_size*2, m_size)

    def forward(self, x, mask): 
        # x = SpatialDropout(0.2)(x)
        feat, _ = self.lstm1(x) # (192, 1024)
        image_logits = self.image_linear(feat)
        feat, _ = self.lstm2(feat) # (192, 1024)
        max_pool, _ = torch.max(feat, 1) # (1024)
        att_pool = self.attention(feat, mask) # (1024)
        conc = torch.cat((max_pool, att_pool), 1) # (2048)
        logits = self.study_linear(conc)
        return logits, image_logits



# Loss & Metric
bce = nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights).float().cuda(), reduction='none')
bce2 = nn.BCEWithLogitsLoss(reduction='none')

def criterion(preds, targets):
    losses = bce(preds.view(-1, out_dim), targets.view(-1, out_dim))
    return losses.mean()


def criterion2(preds, targets, loss_type):
    """
        preds: (bs, 15, 3) (after softmax)
        targets: (bs, 15, 3)
    """
    if loss_type == 'ce':
        losses = bce(preds.view(-1, out_dim), targets.view(-1, out_dim))
        return losses.mean()
    elif loss_type == 'll':
        losses = bce2(preds.view(-1, out_dim), targets.view(-1, out_dim)).squeeze(-1)
        w = torch.where(targets.view(-1, out_dim) == 0, 1, 6).squeeze(-1)
        losses *= w
        return losses.sum() / w.sum()


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_losses = []
    study_losses = []
    image_losses = []

    bar = tqdm(loader_train)
    for embs, mask, study_label, image_labels in bar:
        optimizer.zero_grad()

        embs = embs.cuda()
        mask = mask.cuda()
        study_label = study_label.cuda()
        image_labels = image_labels.cuda()
        

        with amp.autocast():
            logits, image_logits = model(embs, mask)
            study_loss = criterion2(logits.cuda(), study_label.cuda(), 'll')
            image_loss = criterion2(image_logits.cuda(), image_labels.cuda(), 'll')
                
            loss = study_loss * study_ratio + image_loss * (1 - study_ratio)
        
        train_losses.append(loss.item())
        study_losses.append(study_loss.item())
        image_losses.append(image_loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'smth:{np.mean(train_losses[-30:]):.4f} / study:{np.mean(study_losses[-30:]):.4f} / image:{np.mean(image_losses[-30:]):.4f}')

    return np.mean(study_losses)


def valid_func(model, loader_valid):
    model.eval()
    valid_losses = []
    study_losses = []
    image_losses = []
    
    study_gts = []
    image_gts = []

    study_outputs = []
    image_outputs = []

    bar = tqdm(loader_valid)
    with torch.no_grad():
        for embs, mask, study_label, image_labels in bar:
            embs = embs.cuda()
            mask = mask.cuda()
            study_label = study_label.cuda()
            image_labels = image_labels.cuda()

            logits, image_logits = model(embs, mask)
            study_loss = criterion2(logits.cuda(), study_label.cuda(), 'll')
            image_loss = criterion2(image_logits.cuda(), image_labels.cuda(), 'll')
                
            loss = study_loss * study_ratio + image_loss * (1 - study_ratio)
            
            study_gts.append(study_label.cpu())
            image_gts.append(image_labels.cpu())
            study_outputs.append(logits.cpu())
            image_outputs.append(image_logits.cpu())
            
            valid_losses.append(loss.item())
            study_losses.append(study_loss.item())
            image_losses.append(image_loss.item())
            
            bar.set_description(f'smth:{np.mean(valid_losses[-30:]):.4f} / study:{np.mean(study_losses[-30:]):.4f} / image:{np.mean(image_losses[-30:]):.4f}')

    study_outputs = torch.cat(study_outputs)
    image_outputs = torch.cat(image_outputs).squeeze(-1)
    study_gts = torch.cat(study_gts)
    image_gts = torch.cat(image_gts)

    study_ce_loss = criterion2(study_outputs.cuda(), study_gts.cuda(), 'ce').cpu().item()
    study_ll_loss = criterion2(study_outputs.cuda(), study_gts.cuda(), 'll').cpu().item()
    print('study ce_loss', study_ce_loss)
    print('study ll_loss', study_ll_loss)

    image_ce_loss = criterion2(image_outputs.cuda(), image_gts.cuda(), 'ce').cpu().item()
    image_ll_loss = criterion2(image_outputs.cuda(), image_gts.cuda(), 'll').cpu().item()
    print('image ce_loss', image_ce_loss)
    print('image ll_loss', image_ll_loss)
    
    return study_ll_loss





def run(df, fold):

    log_file = os.path.join(log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')

    train_ = df[df['fold'] != fold].reset_index(drop=True)
    valid_ = df[df['fold'] == fold].reset_index(drop=True)
    train_psids = list(train_['psid'].unique())
    valid_psids = list(valid_['psid'].unique())

    dataset_train = CLSExtraSeqDataset(train_, train_psids, 'train')
    dataset_valid = CLSExtraSeqDataset(valid_, valid_psids, 'valid')
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SeqModel()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    metric_best = np.inf
    loss_min = np.inf

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_epochs, eta_min=eta_min)

    print(len(dataset_train), len(dataset_valid))

    with open(log_file, 'a') as appender:
        appender.write(str(backbone) + '\t' + str(m_size) + '\t' + str(init_lr) + '\n')
        

    for epoch in range(1, n_epochs+1):
        scheduler_cosine.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)
        
        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss = valid_func(model, loader_valid)
        metric = valid_loss

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
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








def run_evalsteps(df, fold):

    log_file = os.path.join(log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')

    train_ = df[df['fold'] != fold].reset_index(drop=True)
    valid_ = df[df['fold'] == fold].reset_index(drop=True)
    train_psids = list(train_['psid'].unique())
    valid_psids = list(valid_['psid'].unique())

    dataset_train = CLSExtraSeqDataset(train_, train_psids, 'train')
    dataset_valid = CLSExtraSeqDataset(valid_, valid_psids, 'valid')
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SeqModel()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    metric_best = np.inf
    loss_min = np.inf

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, n_epochs, eta_min=eta_min)


    print(len(dataset_train), len(dataset_valid))
    with open(log_file, 'a') as appender:
        appender.write('\n' + str(backbone) + '\t' + str(m_size) + '\t' + str(init_lr) + '\n')
    

    eval_steps = 0.1

    for epoch in range(1, n_epochs+1):
        
        scheduler_cosine.step(epoch-1)

        train_losses = []
        study_losses = []
        image_losses = []

        bar = tqdm(enumerate(loader_train), total=len(loader_train))
        for steps, (embs, mask, study_label, image_labels) in bar:
            model.train()
            optimizer.zero_grad()
            embs = embs.cuda()
            mask = mask.cuda()
            study_label = study_label.cuda()
            image_labels = image_labels.cuda()
            
            with amp.autocast():
                logits, image_logits = model(embs, mask)
                
                study_loss = criterion2(logits.cuda(), study_label.cuda(), 'll')
                image_loss = criterion2(image_logits.cuda(), image_labels.cuda(), 'll')
                
                loss = study_loss * study_ratio + image_loss * (1-study_ratio)
            
            train_losses.append(loss.item())
            study_losses.append(study_loss.item())
            image_losses.append(image_loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler_cosine.step()

            bar.set_description(f'smth:{np.mean(train_losses[-30:]):.4f} / study:{np.mean(study_losses[-30:]):.4f} / image:{np.mean(image_losses[-30:]):.4f}')

            if steps % int(len(loader_train) * eval_steps) == 0:
                print(f"{steps}/{len(loader_train)} validation")
                train_loss = np.mean(study_losses)
                valid_loss = valid_func(model, loader_valid)
                metric = valid_loss
                content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
                print(content + f'---e{epoch}:{steps}/{int(len(loader_train))}')
                with open(log_file, 'a') as appender:
                    appender.write(content + '\n')

                if metric < metric_best:
                    print(f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
                    torch.save(model.state_dict(), model_file)
                    metric_best = metric

        # after epoch
        print(time.ctime(), 'Epoch:', epoch)

        train_loss = np.mean(study_losses)
        valid_loss = valid_func(model, loader_valid) # study loss
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

        if epoch == 4:
            break


    del model
    torch.cuda.empty_cache()
    gc.collect()



def evaluation(df, model_folder, save=False):
    print('embedding folder :', emb_dir)
    study_outputs = []
    valid_psids_lst = []
    for fold in range(5):
        model_file = os.path.join(model_folder, 'model', f'{kernel_type}_fold{fold}_best.pth')

        valid_ = df[df['fold'] == fold].reset_index(drop=True)
        valid_psids = list(valid_['psid'].unique())

        dataset_valid = CLSExtraSeqDataset(valid_, valid_psids, 'valid')
        loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = SeqModel()
        weight = torch.load(model_file)
        model.load_state_dict(weight)
        model = model.to(device)
        model.eval()

        study_output = []
        bar = tqdm(loader_valid)
        with torch.no_grad():
            for embs, mask, study_label, image_labels in bar:
                embs = embs.cuda()
                mask = mask.cuda()
                study_label = study_label.cuda()

                logits, image_logits = model(embs, mask)
                
                study_output.append(logits.sigmoid().detach().cpu().numpy().squeeze(-1))

        study_output = np.concatenate(study_output)
        study_outputs.append(study_output)
        valid_psids_lst.append(valid_psids)
    
    study_outputs = np.concatenate(study_outputs)
    valid_psids_lst = np.concatenate(valid_psids_lst)
    print(study_outputs.shape, valid_psids_lst.shape)

    valid = pd.DataFrame({'psid':valid_psids_lst, 'extra_output':study_outputs})
    if save:
        valid.to_csv(os.path.join(model_folder, 'extra_valid.csv'), index=False)
        print('save extra output')
    return valid





if __name__ == "__main__":
    df = pd.read_csv(os.path.join(data_dir, f'extra_sliding_{in_chans}_bbox.csv'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='train', help='train or valid')
    parser.add_argument('--eval_type', type=str, default='epoch', help='training or get embedding')
    args = parser.parse_args()

    if args.run_type == 'train':
        if args.eval_type == 'epoch':
            run(df, 0)
            run(df, 1)
            run(df, 2)
            run(df, 3)
            run(df, 4)

        if args.eval_type == 'eval_steps':
            run_evalsteps(df, 0)
            run_evalsteps(df, 1)
            run_evalsteps(df, 2)
            run_evalsteps(df, 3)
            run_evalsteps(df, 4)
    
    if args.run_type == 'valid':
        # check other setting!
        model_folder = "../results/stage2-extra-type1-seg/extra-feat-sliding-seg-seq"
        valid = evaluation(df, model_folder, save=True)
