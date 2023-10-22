import os
import cv2
import glob
import gdcm
import pydicom
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut


TRAIN_PATH = "../../data/dataset/train_images/"

print('Number of training patients :', len(os.listdir(TRAIN_PATH)))


def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
#         pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2    
    
    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    return pixel_array


def process(patient, size=512, save_folder="", data_path=""):
    for study in sorted(os.listdir(data_path + patient)):
        imgs = {}
        for f in sorted(glob.glob(data_path + f"{patient}/{study}/*.dcm")):
            dicom = pydicom.dcmread(f)

            pos_z = dicom[(0x20, 0x32)].value[-1]

            img = standardize_pixel_array(dicom)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = 1 - img

            imgs[pos_z] = img


            print(pos_z, end=' ')

        for i, k in enumerate(sorted(imgs.keys())):
            img = imgs[k]

            if size is not None:
                img = cv2.resize(img, (size, size))

            if isinstance(save_folder, str):
                cv2.imwrite(save_folder + f"{patient}_{study}_{i}.png", (img * 255).astype(np.uint8))
            else:
                im = cv2.imencode('.png', (img * 255).astype(np.uint8))[1]
                save_folder.writestr(f'{patient}_{study}_{i:04d}.png', im)

if __name__ == "__main__":
    patients = os.listdir(TRAIN_PATH)

    print(len(patients))

    save_folder = '../../data/png_folder/'
    os.makedirs(save_folder, exist_ok=True)
    for patient in tqdm(patients):
        process(patient, size=None, save_folder=save_folder, data_path=TRAIN_PATH)