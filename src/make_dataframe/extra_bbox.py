import numpy as np
import pandas as pd
import os
import glob
import pickle
from tqdm import tqdm

DATA_DIR = '../../data'
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')


def preprocess_bbox_df(bbox):
    remove = 63618

    bbox = bbox[bbox['series_id'] != remove].reset_index(drop=True)
    bbox['psid'] = bbox['pid'].astype(str) + '_' + bbox['series_id'].astype(str)
    bbox = bbox.merge(inverse, how='left', on='psid')

    bbox2 = bbox[['psid','instance_number','x1','y1','x2','y2','width','height','inverse']].merge(dcm_number[['psid','start','end']], how='left', on='psid')
    bbox2['instance_number_inv'] = bbox2['end'] - (bbox2['instance_number']-bbox2['start'])
    bbox2['final_instance'] = np.where(bbox2['inverse'], bbox2['instance_number_inv'], bbox2['instance_number'])

    bbox2['png_start'] = bbox2['start'] - bbox2['start']
    bbox2['png_end'] = bbox2['end'] - bbox2['start']
    bbox2['png_instance'] = bbox2['final_instance'] - bbox2['start']

    bbox2['image_label'] = 1
    
    return bbox2

def get_image_slices(d):
    psids, png_folders = [], []
    for k, v in tqdm(d.items(), total=len(list(d.keys()))):
        psids.append([k] * len(v))
        png_folders.append(v)
    psids = np.concatenate(psids)
    png_folders = np.concatenate(png_folders)
    len(psids), len(png_folders)

    df_image = pd.DataFrame({'psid':psids, 'png_folder':png_folders})
    df_image['png_number'] = df_image['png_folder'].apply(lambda x: int(x.split("_")[-1].split('.')[0]))

    # df :: start, end : dcm / label_min, label_max : dcm
    df_image2 = df_image.merge(df, how='left', on='psid')
    return df_image2


def save_mask(bbox, save=False):
    """
        save mask only image_label == 1
    """
    print('save mask') if save else print('only check')

    output_dir = os.path.join(DATA_DIR, 'extra_mask')
    os.makedirs(output_dir, exist_ok=True)

    sz = 512
    for i, row in tqdm(bbox.iterrows(), total=len(bbox)):
        path = os.path.join(output_dir, f"{str(row['psid'])}_{str(row['png_instance'])}.npy")
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        mask = np.zeros((512,512), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1.
        if save:
            np.save(path, mask)

def get_all_image_df(df_image, bbox):
    df_image = df_image.merge(bbox.rename(columns={'png_instance':'png_number'}), how='left', on=['psid','png_number'])
    df_image['image_label'] = df_image['image_label'].fillna(0)
    return df_image


# sliding slices (5 channels, 5 stride, +-2 chans)
def sliding_slices(df_image, save=False):
    n_s, unit = 5, 2 # 5, +-2
    psid_lst = df_image['psid'].unique()
    lst = []
    for _psid in tqdm(psid_lst, total=len(psid_lst)):
        tmp = df_image[df_image['psid']==_psid]
        if len(tmp) < 10:
            lst.append(tmp)
        else:
            tmp_ill = tmp['image_label'].values
            _lst = []
            for i, k in enumerate(tmp['png_number'].values):
                if i < unit or i >= len(tmp)-unit:
                    continue
                if k%n_s == unit:
                    row = tmp.iloc[k]
                    _lst.append(row)
            _lst = pd.DataFrame(_lst)
            lst.append(_lst)
    lst = pd.concat(lst, axis=0)
    
    if save:
        lst.reset_index(drop=True).to_csv(
            os.path.join(DATA_DIR, f"extra_sliding_{n_s}_bbox.csv"), index=False
        )
    return lst

def get_only_pos_df(df_image, idx, save=False):
    df_image_pos = df_image[df_image['image_label']==1]
    df_image_pos = df_image_pos[~df_image_pos.index.isin(idx)]

    if save:
        df_image_pos.to_csv(os.path.join(DATA_DIR, 'extra_sliding_5_bbox_pos.csv'), index=False)
    return df_image_pos



if __name__ == "__main__":

    d = pickle.load(open(os.path.join(DATA_DIR, 'd.pkl'), 'rb'))

    image_level_label = pd.read_csv(os.path.join(DATASET_DIR, 'image_level_labels.csv'))
    inverse = pd.read_csv(os.path.join(DATA_DIR, 'inverse.csv'))
    bbox = pd.read_csv(os.path.join(DATA_DIR, 'active_extravasation_bounding_boxes.csv')) # ian's label
    dcm_number = pd.read_csv(os.path.join(DATA_DIR, 'dcm_number.csv'))
    dcm_number['psid'] = dcm_number['patient_id'].astype(str) + '_' + dcm_number['series_id'].astype(str)

    train = pd.read_csv(os.path.join(DATA_DIR, 'train_df.csv'))
    df = train.copy()
    df['psid'] = df['patient_id'].astype(str) + '_' + df['series_id'].astype(str)

    psid = []
    organ = []
    healthy = []
    injury = []

    fold = []
    for _, row in df.iterrows():
        for o in ['extravasation']:
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
    
    bbox = preprocess_bbox_df(bbox)
    df_image = get_image_slices(d)

    # make extra mask
    save_mask(bbox, save=True)
    
    df_image = get_all_image_df(df_image, bbox)

    extra_sliding_5_bbox = sliding_slices(df_image, save=True)

    idx = extra_sliding_5_bbox.query('image_label == 1').index

    df_image_pos = get_only_pos_df(df_image, idx, save=True)
