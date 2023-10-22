import os
import pickle
import glob
import pandas as pd
from tqdm import tqdm

data_dir = '../../data'
png_dir = os.path.join(data_dir, 'png_folder')

train_df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))

def gen_pickle(save=False):
    """
        generate pickle file for png_files by "patient-series id"
        to handle difference of image order between dcm files and png files that ordered by z-axis correctly.
    """
    if os.path.isfile(os.path.join(data_dir, 'd.pkl')):
        with open(os.path.join(data_dir, 'd.pkl'), 'rb') as f:
            d = pickle.load(f)
        print('load pickle file')
    else:
        print('generate pickle file')
        t = glob.glob(png_dir + '/*')
        print(len(t))

        d = {}
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            png_suffix = row['png_suffix']
            k = png_suffix.split('/')[-1]

            d[k] = [k[3:] for k in t if png_suffix in k]
            d[k] = sorted(d[k], key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

        if save:
            with open(os.path.join(data_dir, 'd.pkl'), 'wb') as f:
                pickle.dump(d, f)
            print('save pickle file')
        print('generate pickle file')
    return d
        

if __name__ == "__main__":
    d = gen_pickle(save=True)