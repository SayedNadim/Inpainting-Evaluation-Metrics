import re
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import cv2


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + list(
                glob.glob(flist + '/*.JPG'))
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list' % flist)
    return []

# read folder, return torch Tensor in NCHW, normalized
def _read_folder(foldername):
    input_image_list = (get_image_list(foldername))
    input_image_list.sort(key=natural_keys)
    img_list = []
    print('Reading Images from %s ...' % foldername)
    for file in tqdm(input_image_list):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)  # NHWC -> NCHW
        img_list.append(img)
    img_list_tensor = torch.Tensor(np.concatenate(img_list, axis=0))
    return img_list_tensor