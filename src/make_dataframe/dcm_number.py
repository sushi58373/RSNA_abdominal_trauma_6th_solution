import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

seed = 42

DATA_DIR = '../../data'
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')

def make_dcm_number(save=False):
    train_series_meta = pd.read_csv(os.path.join(DATASET_DIR, 'train_series_meta.csv'))

    path = os.path.join(DATASET_DIR, 'train_images')

    starts, ends = [], []
    for i, row in tqdm(train_series_meta.iterrows(), total=len(train_series_meta)):
        files = glob.glob(os.path.join(path, str(int(row['patient_id'])), str(int(row['series_id'])), '*'))
        files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        _min = int(files[0].split('/')[-1].split('.')[0])
        _max = int(files[-1].split('/')[-1].split('.')[0])
        starts.append(_min)
        ends.append(_max)

    train_series_meta['start'] = starts
    train_series_meta['end'] = ends
    if save:
        train_series_meta.to_csv(os.path.join(DATA_DIR, 'dcm_number.csv'), index=False)
    return train_series_meta

if __name__ == "__main__":
    dcm_number = make_dcm_number(save=True)