import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import glob

seed = 42

DATA_DIR = '../../data'
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
SEG_DIR = os.path.join(DATASET_DIR, 'segmentations')
PNG_DIR = os.path.join(DATA_DIR, 'png_folder')
DCM_DIR = os.path.join(DATASET_DIR, 'train_images')

train_meta = pd.read_csv(os.path.join(DATASET_DIR, 'train_series_meta.csv'))
train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
image_level_label = pd.read_csv(os.path.join(DATASET_DIR, 'image_level_labels.csv'))


def to_seg_df(save=False):
    seg_series_ids = sorted([int(p.split(".")[0]) for p in os.listdir(SEG_DIR)])
    print(len(seg_series_ids))

    seg_df = train_meta[train_meta['series_id'].isin(seg_series_ids)].reset_index(drop=True)
    tmp = train[train['patient_id'].isin(seg_df['patient_id'].unique())]
    seg_df = pd.merge(tmp, seg_df, how='left', on='patient_id')

    if save:
        seg_df.to_csv(os.path.join(DATA_DIR, f"seg_df.csv"), index=False)
        print('save [seg_df.csv]')
    return seg_df


def to_df_seg(seg_df, save=False):
    df_seg = pd.read_csv(os.path.join(DATA_DIR, 'seg_df.csv'))

    mask_files = os.listdir(SEG_DIR) # by series_id
    df_mask = pd.DataFrame({
        'mask_file': mask_files,
    })
    df_mask['series_id'] = df_mask['mask_file'].apply(lambda x: int(x[:-4]))
    df_mask['mask_file'] = df_mask['mask_file'].apply(lambda x: os.path.join(SEG_DIR[3:], x))
    df = df_seg.merge(df_mask, how='left', on='series_id')
    df['png_suffix'] = PNG_DIR[3:] + '/' + df['patient_id'].astype(str) + '_' + df['series_id'].astype(str)
    df['dcm_folder'] = DCM_DIR[3:] + '/' + df['patient_id'].astype(str) + '/' + df['series_id'].astype(str)
    df['mask_file'].fillna('', inplace=True)

    df = df.query('mask_file != ""').reset_index(drop=True)
    
    tmp = df.groupby('patient_id')['series_id'].agg(['count']).reset_index()
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    tmp['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(skf.split(tmp, tmp['count'])):
        tmp.loc[valid_idx, 'fold'] = fold
    
    df = df.merge(tmp, how='left', on='patient_id')
    if save:
        df.to_csv(os.path.join(DATA_DIR, 'df_seg.csv'), index=False)
        print('save [df_seg.csv]')
    return df



if __name__ == "__main__":
    seg_df = to_seg_df(save=True)
    df_seg = to_df_seg(seg_df, save=True)
