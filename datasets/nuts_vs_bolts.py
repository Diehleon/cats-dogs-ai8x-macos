import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as album
import cv2

import ai8x  # from ai8x-training

class NutsVsBolts(Dataset):
    """
    Nuts vs Bolts Dataset.

    Root directory structure:
        nuts_vs_bolts/
            train/
                nuts/
                bolts/
            test/
                nuts/
                bolts/
    """
    labels = ['nuts', 'bolts']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {'nuts': 'nuts', 'bolts': 'bolts'}

    def __init__(self, root_dir, d_type, transform=None,
                 resize_size=(128, 128), augment_data=False):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, d_type)

        if not self.__check_data_exist():
            print(f"ERROR: data directory not found: {self.data_dir}")
            self.__print_manual_instructions()
            sys.exit(1)

        self.__get_image_paths()

        # Define augmentation/transforms
        if d_type == 'train' and augment_data:
            self.album_transform = album.Compose([
                album.GaussNoise(var_limit=(1.0, 20.0), p=0.25),
                album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                album.ColorJitter(p=0.5),
                album.SmallestMaxSize(max_size=int(1.2 * min(resize_size))),
                album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                                       rotate_limit=15, p=0.5),
                album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                album.HorizontalFlip(p=0.5),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            ])
        else:
            self.album_transform = album.Compose([
                album.SmallestMaxSize(max_size=int(1.2 * min(resize_size))),
                album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            ])

        self.transform = transform

    def __check_data_exist(self):
        if not os.path.isdir(self.data_dir):
            return False
        for label in self.labels:
            folder = self.label_to_folder_map[label]
            path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(path):
                return False
        return True

    def __print_manual_instructions(self):
        print("******************************************")
        print("Make sure your data is organized as:")
        print("  <data_root>/nuts_vs_bolts/train/nuts")
        print("  <data_root>/nuts_vs_bolts/train/bolts")
        print("  <data_root>/nuts_vs_bolts/test/nuts")
        print("  <data_root>/nuts_vs_bolts/test/bolts")
        print("You passed root_dir:", self.root_dir)
        print("************************************************")

    def __get_image_paths(self):
        self.data_list = []
        for label in self.labels:
            folder = self.label_to_folder_map[label]
            image_dir = os.path.join(self.data_dir, folder)
            try:
                file_names = sorted(os.listdir(image_dir))
            except Exception as e:
                print(f"WARNING: cannot list directory {image_dir}: {e}")
                continue
            for fname in file_names:
                path = os.path.join(image_dir, fname)
                if os.path.isfile(path):
                    self.data_list.append((path, self.label_to_id_map[label]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, lbl = self.data_list[index]
        label = torch.tensor(lbl, dtype=torch.int64)

        image = cv2.imread(image_path)
        if image is None:
            # skip sample
            next_idx = (index + 1) % len(self)
            return self.__getitem__(next_idx)

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            next_idx = (index + 1) % len(self)
            return self.__getitem__(next_idx)

        if self.album_transform:
            image = self.album_transform(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_nuts_vs_bolts_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])

    train_ds = None
    test_ds = None
    if load_train:
        train_ds = NutsVsBolts(root_dir=os.path.join(data_dir, 'nuts_vs_bolts'),
                               d_type='train',
                               transform=transform,
                               augment_data=True)
    if load_test:
        test_ds = NutsVsBolts(root_dir=os.path.join(data_dir, 'nuts_vs_bolts'),
                              d_type='test',
                              transform=transform,
                              augment_data=False)
    return train_ds, test_ds

# Add to datasets list:
datasets = [
    {
        'name': 'nuts_vs_bolts',
        'input': (3, 128, 128),
        'output': ('nuts', 'bolts'),
        'loader': get_nuts_vs_bolts_dataset,
    },
]
