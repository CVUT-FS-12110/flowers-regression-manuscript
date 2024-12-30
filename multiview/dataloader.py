from __future__ import print_function
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A

from glob import glob
import cv2 as cv

import random
import numpy as np
import matplotlib.pylab as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def set_random_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value} for Python, NumPy, and PyTorch (CUDA).")




class CustomDataset(Dataset):

    def __init__(self, trees, data_path, augment=True, ground_truth=True):
        super(CustomDataset, self).__init__()

        res = 512
        self.augment = augment
        self.elms = len(trees)
        self.data_a = np.empty((self.elms, res, res, 3), dtype='uint8')
        self.data_b = np.empty((self.elms, res, res, 3), dtype='uint8')
        self.targets = np.empty((self.elms, 1), dtype="float32")
        self.names = []
        self.transform =  transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a tensor with values [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

        filenames = list(glob(data_path))

        for idx, tree in enumerate(trees):
            name_a = "_".join(tree) + "_a_"
            name_b = "_".join(tree) + "_b_"

            if idx % 2:
                name_a, name_b = name_b, name_a

            self.names.append("_".join(tree))

            for filename in filenames:
                if name_a in filename:
                    img = cv.imread(filename)
                    self.data_a[idx] = cv.resize(img, (res, res))
                if name_b in filename:
                    img = cv.imread(filename)
                    self.data_b[idx] = cv.resize(img, (res, res))
                    cells = os.path.basename(filename).replace(name_b, "").replace(".jpg", "").split("_")
                    raw_ground, raw_photo = cells[0], cells[1]
                    if ground_truth:
                        self.targets[idx] = float(raw_ground)
                    else:
                        self.targets[idx] = float(raw_photo)

    def augment_albu(self, image):
        t = A.Compose([
            A.ShiftScaleRotate(p=1, shift_limit=(-0.01, 0.01), scale_limit=(-0.01, 0.01), rotate_limit=(-1, 1)),
            # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1),
            # A.GaussianBlur(blur_limit=(3, 5), p=1),
            ]
        )
        augmented = t(image=image)
        return augmented['image']

    def __len__(self):
        return self.elms

    def __getitem__(self, idx):

        target = torch.tensor(self.targets[idx], dtype=torch.float)
        image_a = self.data_a[idx].copy()
        image_b = np.flip(self.data_b[idx].copy(), axis=1)
        if self.augment:
            image_a = self.augment_albu(image_a)
            image_b = self.augment_albu(image_b)

        image_a = self.transform(image_a.copy())
        image_b = self.transform(image_b.copy())

        # image_a = np.transpose(image_a, (2, 0, 1))
        # image_b = np.transpose(image_b, (2, 0, 1))

        return image_a, image_b, target, self.names[idx]

        # return image_a.astype('float32') / 255, image_b.astype('float32') / 255, target, self.names[idx]


if __name__ == "__main__":

    ground_truth = True

    data_path = "final_data/*.jpg"
    filenames = list(glob(data_path))

    observations = list(set([tuple(os.path.basename(name).split(".")[0].split("_")[0:3]) for name in filenames]))

    trees = list(set([observation[1:3] for observation in observations]))

    split_point = 8
    train_trees = trees[split_point:]
    train_samples = [observation for observation in observations if observation[1:3] in train_trees]

    test_trees = trees[:split_point]
    test_samples = [observation for observation in observations if observation[1:3] in test_trees]


    train_dataset = CustomDataset(train_samples, data_path, ground_truth=ground_truth)
    train_loader = torch.utils.data.DataLoader(train_dataset)

    test_dataset = CustomDataset(test_samples, data_path, ground_truth=ground_truth)

    test_loader = torch.utils.data.DataLoader(test_dataset)

    for batch_idx, (images_a, images_b, targets, names) in enumerate(test_loader):
        for image_a, image_b, target in zip(images_a, images_b, targets):
            plt.subplot(121)
            plt.imshow(np.transpose(image_a, (1, 2, 0)))
            plt.subplot(122)
            plt.imshow(np.transpose(image_b, (1, 2, 0)))
            plt.title(str(target))
            plt.show()

