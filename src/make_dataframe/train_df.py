import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

seed = 42

DATA_DIR = '../../data'
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
DCM_DIR = os.path.join(DATASET_DIR, 'train_images')
PNG_DIR = os.path.join(DATA_DIR, 'png_folder')
SEG_DIR = os.path.join(DATASET_DIR, 'segmentations')

image_level_label = pd.read_csv(os.path.join(DATASET_DIR, 'image_level_labels.csv'))
train_series_meta = pd.read_csv(os.path.join(DATASET_DIR, 'train_series_meta.csv'))
train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))

df_seg = pd.read_csv(os.path.join(DATA_DIR, 'df_seg.csv'))

df_train = train_series_meta.merge(train, how='left', on='patient_id')

cols = [
        'any_injury',
        'bowel_injury',
        'extravasation_injury',
        'kidney_high',
        'spleen_high',
        'liver_high'
    ]

def get_train_df(df_train, df_seg, cols, save=False):
    df_tmp = df_train[~df_train['series_id'].isin(df_seg['series_id'].unique())]
    df_tmp_not = df_train[df_train['series_id'].isin(df_seg['series_id'].unique())]
    assert df_tmp['series_id'].nunique() + df_seg['series_id'].nunique() == df_train['series_id'].nunique()

    df_tmp_not = df_tmp_not.merge(df_seg[['patient_id', 'series_id', 'fold']], how='left', on=['patient_id','series_id'])

    k = df_tmp.groupby('patient_id').first().reset_index()

    sgkf = MultilabelStratifiedKFold(5, shuffle=True, random_state=seed)

    k['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(sgkf.split(X=k, y=k[cols[1:]])):
        k.loc[valid_idx, 'fold'] = fold

    k2 = df_tmp.merge(k[['patient_id', 'fold']], how='left', on='patient_id')

    train_df = pd.concat([k2, df_tmp_not], axis=0).reset_index(drop=True)
    print(train_df.groupby('patient_id')['fold'].nunique().value_counts())

    train_df['png_suffix'] = PNG_DIR[3:] + '/' + train_df['patient_id'].astype(str) + '_' + train_df['series_id'].astype(str)
    train_df['dcm_folder'] = DCM_DIR[3:] + '/' + train_df['patient_id'].astype(str) + '/' + train_df['series_id'].astype(str)

    if save:
        train_df.to_csv(os.path.join(DATA_DIR, 'train_df.csv'), index=False)
        print('save [train_df.csv]')
    return train_df

if __name__ == "__main__":
    train_df = get_train_df(df_train, df_seg, cols, save=True)