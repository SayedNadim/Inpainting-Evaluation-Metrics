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
    def get_ff_mask(H, W):
        # Source: Generative Inpainting https://github.com/JiahuiYu/generative_inpainting
        min_num_vertex = 3
        max_num_vertex = 8
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 10
        max_width = 30

        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 8)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        return np.asarray(mask, np.float32)



if __name__ == '__main__':
    ff_save_dir_10_20 = './ff_mask/mask_10_20/'
    ff_save_dir_20_30 = './ff_mask/mask_20_30/'
    ff_save_dir_30_40 = './ff_mask/mask_30_40/'
    ff_save_dir_40_50 = './ff_mask/mask_40_50/'
    ff_count_10_20 = 1
    ff_count_20_30 = 1
    ff_count_30_40 = 1
    ff_count_40_50 = 1
    while not (ff_count_10_20 % 100 == 0 and ff_count_20_30 % 100 == 0 and ff_count_30_40 % 100 == 0 and ff_count_40_50 % 100 == 0):
        if not os.path.exists(ff_save_dir_10_20):
            os.makedirs(ff_save_dir_10_20)
        if not os.path.exists(ff_save_dir_20_30):
            os.makedirs(ff_save_dir_20_30)
        if not os.path.exists(ff_save_dir_30_40):
            os.makedirs(ff_save_dir_30_40)
        if not os.path.exists(ff_save_dir_40_50):
            os.makedirs(ff_save_dir_40_50)
        mask = Masks.get_ff_mask(256, 256)
        mask = transforms.ToTensor()(mask)
        mask = mask.unsqueeze(0)
        total_pixel = mask.shape[2] * mask.shape[3]
        ff_count_hole_pixels = torch.count_nonzero(mask)
        hole_percentage = (ff_count_hole_pixels * 100) / total_pixel
        if hole_percentage >= 10 and hole_percentage < 20:
            if ff_count_10_20 == 100:
                pass
            else:
                vutils.save_image(mask, ff_save_dir_10_20 + 'mask_{}.png'.format(ff_count_10_20),
                                  padding=0, normalize=True)
                ff_count_10_20 += 1

        if hole_percentage >= 20 and hole_percentage < 30:
            if ff_count_20_30 == 100:
                pass
            else:
                vutils.save_image(mask, ff_save_dir_20_30 + 'mask_{}.png'.format(ff_count_20_30),
                                  padding=0, normalize=True)
                ff_count_20_30 += 1

        if hole_percentage >= 30 and hole_percentage < 40:
            if ff_count_30_40 == 100:
                pass
            else:
                vutils.save_image(mask, ff_save_dir_30_40 + 'mask_{}.png'.format(ff_count_30_40),
                                  padding=0, normalize=True)
                ff_count_30_40 += 1

        if hole_percentage >= 40 and hole_percentage < 50:
            if ff_count_40_50 == 100:
                pass
            else:
                vutils.save_image(mask, ff_save_dir_40_50 + 'mask_{}.png'.format(ff_count_40_50),
                                  padding=0, normalize=True)
                ff_count_40_50 += 1

