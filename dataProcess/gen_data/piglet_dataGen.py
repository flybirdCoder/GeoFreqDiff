import argparse
import numpy as np
import pydicom
from natsort import natsorted
from glob import glob
import os


def normalize(image, MIN_B=-1024.0, MAX_B=3072.0):
    image = image - 1024
    image[image < MIN_B] = MIN_B
    image[image > MAX_B] = MAX_B
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image


def read_dicom_image(file_list, file_ind, dir):
    data = pydicom.dcmread(file_list[file_ind])
    img = np.array(data.pixel_array).astype(np.int16)
    img[img < 0] = 0

    f = img
    f = normalize(f, args.norm_range_min, args.norm_range_max)
    f_name = 'piglet_' + dose + '_{:0>3d}_img.npy'.format(file_ind)
    np.save(os.path.join(dir, f_name), f)


def save_dataset(args):
    global dose
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
        print('Create path : {}'.format(args.save_root))

    train_dir = os.path.join(args.save_root, 'train')
    test_dir = os.path.join(args.save_root, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    test_indices = np.array(list(range(21, 871, 6)))
    train_indices = np.array([x for x in range(21, 871) if x not in test_indices])

    for folder_ind in range(33):
        file_list = natsorted(glob(os.path.join(args.data_root, "SE{}".format(folder_ind), "IM*")))
        data = pydicom.dcmread(file_list[0])
        recon_type = data.SeriesDescription
        if "FBP" in recon_type:
            if "FULL" in recon_type:
                dose = 'target'
            elif "25" in recon_type:
                dose = '25'

            for file_ind in train_indices:
                if file_ind < len(file_list):
                    read_dicom_image(file_list, file_ind, train_dir)

            for file_ind in test_indices:
                if file_ind < len(file_list):
                    read_dicom_image(file_list, file_ind, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='D:/Download/论文/SDM-Denoiser/dataset/piglet/PA0/ST0/')  # data format: dicom
    parser.add_argument('--save_root', type=str, default='D:/Download/论文/SDM-Denoiser/dataset/piglet_25_npy')
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    args = parser.parse_args()

    save_dataset(args)
