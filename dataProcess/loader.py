import os
from glob import glob

import numpy as np
from torch.utils.data import Dataset, DataLoader


def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
        patch_target_img = full_target_img[top:top + new_h, left:left + new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


class CTDataset(Dataset):
    def __init__(self, mode, processed_data, test_patient, val_patient,
                 patch_n=None, patch_size=None):
        assert mode in ['train', 'test', 'val'], "mode is 'train, val' or 'test'"
        # for mayo and clinic
        input_path = sorted(glob(os.path.join(processed_data, '*_input.npy')))
        target_path = sorted(glob(os.path.join(processed_data, '*_target.npy')))

        # for piglet
        # input_path = sorted(glob(os.path.join(processed_data, '*_25_*.npy')))
        # target_path = sorted(glob(os.path.join(processed_data, '*_target_*.npy')))

        self.patch_n = patch_n
        self.patch_size = patch_size

        if mode == 'train':
            input_train = [f for f in input_path if test_patient not in f and val_patient not in f]
            target_train = [f for f in target_path if test_patient not in f and val_patient not in f]

            self.input_ = input_train
            self.target_ = target_train

        elif mode == 'test':
            # for mayo and clinic
            input_test = [f for f in input_path if test_patient in f]
            target_test = [f for f in target_path if test_patient in f]

            # for piglet
            # input_test = [f for f in input_path if test_patient not in f]
            # target_test = [f for f in target_path if test_patient not in f]

            self.input_ = input_test
            self.target_ = target_test

        else:
            # validate
            input_test = [f for f in input_path if val_patient in f]
            target_test = [f for f in target_path if val_patient in f]

            self.input_ = input_test
            self.target_ = target_test

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = np.load(self.input_[idx]), np.load(self.target_[idx])

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return [target_patches, input_patches]
        else:
            return [target_img, input_img]


def get_loader(mode='train', processed_data=None, test_patient='L506', val_patient='L333',
               patch_n=None, patch_size=None,
               batch_size=32, num_workers=6, accerlator=None):
    dataset = CTDataset(mode, processed_data, test_patient, val_patient, patch_n, patch_size)
    if mode == 'train':
        data_loader = cycle(accerlator.prepare(DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)))
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader
