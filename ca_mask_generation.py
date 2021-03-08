import os
import cv2
import numpy as np
import random
import scipy.ndimage as ndimage
import math
from PIL import Image, ImageDraw
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms


## https://arxiv.org/pdf/2010.01110.pdf
class Masks():
    @staticmethod
    def get_ca_mask(h, w, scale=None):

        if scale is None:
            scale = random.choice([1, 2, 4, 8, 16])

        height = h
        width = w
        mask = np.random.randint(2, size=(height // scale, width // scale))

        mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)
        if scale > 1:
            struct = ndimage.generate_binary_structure(2, 1)
            mask = ndimage.morphology.binary_dilation(mask, struct)


        return np.asarray(mask, np.float32)


if __name__ == '__main__':
    ca_save_dir_50_60 = './ca_mask/mask_50_60/'
    ca_save_dir_60_70 = './ca_mask/mask_60_70/'
    ca_save_dir_70_80 = './ca_mask/mask_70_80/'
    ca_save_dir_80_90 = './ca_mask/mask_80_90/'
    ca_count_50_60 = 1
    ca_count_60_70 = 1
    ca_count_70_80 = 1
    ca_count_80_90 = 1
    while not (ca_count_50_60 % 100 == 0 and ca_count_60_70 % 100 == 0 and ca_count_70_80 % 100 == 0 and ca_count_80_90 % 100 == 0):
        if not os.path.exists(ca_save_dir_50_60):
            os.makedirs(ca_save_dir_50_60)
        if not os.path.exists(ca_save_dir_60_70):
            os.makedirs(ca_save_dir_60_70)
        if not os.path.exists(ca_save_dir_70_80):
            os.makedirs(ca_save_dir_70_80)
        if not os.path.exists(ca_save_dir_80_90):
            os.makedirs(ca_save_dir_80_90)
        mask = Masks.get_ca_mask(256, 256)
        mask = transforms.ToTensor()(mask)
        mask = mask.unsqueeze(0)
        total_pixel = mask.shape[2] * mask.shape[3]
        ca_count_hole_pixels = torch.count_nonzero(mask)
        hole_percentage = (ca_count_hole_pixels * 100) / total_pixel
        if hole_percentage >= 50 and hole_percentage < 60:
            if ca_count_50_60 == 100:
                pass
            else:
                vutils.save_image(mask, ca_save_dir_50_60 + 'mask_{}.png'.format(ca_count_50_60),
                                  padding=0, normalize=True)
                ca_count_50_60 += 1

        if hole_percentage >= 60 and hole_percentage < 70:
            if ca_count_60_70 == 100:
                pass
            else:
                vutils.save_image(mask, ca_save_dir_60_70 + 'mask_{}.png'.format(ca_count_60_70),
                                  padding=0, normalize=True)
                ca_count_60_70 += 1

        if hole_percentage >= 70 and hole_percentage < 80:
            if ca_count_70_80 == 100:
                pass
            else:
                vutils.save_image(mask, ca_save_dir_70_80 + 'mask_{}.png'.format(ca_count_70_80),
                                  padding=0, normalize=True)
                ca_count_70_80 += 1

        if hole_percentage >= 80 and hole_percentage < 90:
            if ca_count_80_90 == 100:
                pass
            else:
                vutils.save_image(mask, ca_save_dir_80_90 + 'mask_{}.png'.format(ca_count_80_90),
                                  padding=0, normalize=True)
                ca_count_80_90 += 1

