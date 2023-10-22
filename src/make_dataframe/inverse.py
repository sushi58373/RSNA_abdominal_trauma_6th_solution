import os
import numpy as np
import pandas as pd
import pickle
import dicomsdl
import gdcm
from tqdm import tqdm
import glob


DATA_DIR = '../../data'
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
data_type = 'train'
test_dir = os.path.join(DATASET_DIR, f'{data_type}_images')

df = pd.read_csv(os.path.join(DATASET_DIR, f'{data_type}_series_meta.csv'))
df['psid'] = df['patient_id'].astype(str) + '_' + df['series_id'].astype(str)
df['image_folder'] = test_dir + '/' + df['patient_id'].astype(str) + '/' + df['series_id'].astype(str)

def dicomsdl_to_numpy_image(ds, index=0):
    info = ds.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')  # number of separate planes in this image
    shape = [info['Rows'], info['Cols']]
    dtype = info['dtype']
    outarr = np.empty(shape, dtype=dtype)
    ds.copyFrameData(index, outarr)
    return outarr

def check_inverse(t_paths):
    n_scans = len(t_paths)
    
    #check inversion
    min_index, max_index = t_paths[0], t_paths[-1]
    dcm0 = dicomsdl.open(min_index)
    dcmN = dicomsdl.open(max_index)
    sx0, sy0, sz0 = dcm0.ImagePositionPatient
    sxN, syN, szN = dcmN.ImagePositionPatient
    
    inversion = True if szN < sz0 else False
    if inversion:
        t_paths = t_paths[::-1]
        print()
    return t_paths, inversion

def get_inverse_df(df, save=False):
    d2 = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        psid = str(row['psid'])
        pid, sid = str(row['patient_id']), str(row['series_id'])
        t_paths = sorted(glob.glob(os.path.join(test_dir, pid, sid, "*")), key=lambda x: int(x.split('/')[-1].split(".")[0]))    
        t_paths, inverse = check_inverse(t_paths)
        d2[psid] = inverse

    df2 = pd.DataFrame.from_dict(d2, orient='index', columns=['inverse']).reset_index().rename(columns={'index':'psid'})
    if save:
        df2.to_csv(os.path.join(DATA_DIR, "inverse.csv"), index=False)
    return df2


if __name__ == "__main__":
    inverse = get_inverse_df(df, save=True)