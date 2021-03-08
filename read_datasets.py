import sys
import torch.utils.data as data
from os import listdir
import os
import glob
import re
import numpy as np
import fnmatch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Dataset(data.Dataset):
    def __init__(self, img_path, gt_path, image_shape):
        super(Dataset, self).__init__()
        self.gt_samples = [os.path.join(gt_path, x) for x in listdir(gt_path) if self.is_image_file(x)]
        self.img_samples = [os.path.join(img_path, x) for x in listdir(img_path) if self.is_image_file(x)]
        self.gt_samples.sort(key=natural_keys)
        self.img_samples.sort(key=natural_keys)
        self.img_path = img_path
        self.gt_path = gt_path
        self.image_shape = [image_shape[0], image_shape[1]]

    def __getitem__(self, index):
        gt_path = os.path.join(self.gt_path, self.gt_samples[index])
        img_path = os.path.join(self.img_path, self.img_samples[index])

        img = default_loader(img_path)
        gt = default_loader(gt_path)

        img = Image.fromarray(img)
        gt = Image.fromarray(gt)

        img = transforms.Resize(self.image_shape)(img)
        gt = transforms.Resize(self.image_shape)(gt)

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        gt = transforms.ToTensor()(gt)  # turn the image to a tensor

        return img, gt

    def __len__(self):
        return len(self.gt_samples)

    def is_image_file(self, filename):
        img_extension = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        filename_lower = filename.lower()
        return any(filename_lower.endswith(extension) for extension in img_extension)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def cv2_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def default_loader(path):
    return cv2_loader(path)


def build_dataloader(img_path, gt_path, image_shape, batch_size,
                     num_workers, shuffle=False):
    dataset = Dataset(
        img_path=img_path,
        gt_path=gt_path,
        image_shape=image_shape
    )

    print('Total instance number:', dataset.__len__())

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=shuffle,
        pin_memory=False
    )

    return dataloader
