import torch
import piq
import argparse
import torch.nn.functional as F
import torch.nn as nn
from image_read import _read_folder
from tqdm import trange

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@torch.no_grad()
def main(pred_path, gt_path):
    x = _read_folder(pred_path)
    y = _read_folder(gt_path)
    assert x.shape[0] == y.shape[0]
    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        x = x.cuda()
        y = y.cuda()
    x = F.interpolate(x, (256, 256), mode='bilinear')
    y = F.interpolate(y, (256, 256), mode='bilinear')
    psnr_tensor = 0
    ssim_tensor = 0
    l1_tensor = 0
    l2_tensor = 0
    lpips_tensor = 0
    count = 1

    print("Calculating image quality metrics ...")
    for i in trange(x.shape[0]):
        # lpips_loss: torch.Tensor = piq.LPIPS(reduction='mean')(x[i], y[i])
        ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(y[i], x[i], data_range=1.)
        psnr_index: torch.Tensor = piq.psnr(y[i], x[i], data_range=1., reduction='mean')
        l1_index = nn.L1Loss(reduction='mean')(y[i], x[i])
        l2_index = nn.MSELoss(reduction='mean')(y[i], x[i])

        ssim_tensor += ms_ssim_index
        psnr_tensor += psnr_index
        l1_tensor += l1_index
        l2_tensor += l2_index
        # lpips_tensor += lpips_loss.item()

        count += 1
    # print(
    #     "Avg. LPIPS: {} \nAvg. SSIM: {} \nAvg. PSNR: {} \nAvg. L1: {} \nAvg. L2: {} \n".format(lpips_tensor / count,
    #                                                                                            ssim_tensor / count,
    #                                                                                            psnr_tensor / count,
    #                                                                                            l1_tensor / count,
    #                                                                                            l2_tensor / count))

    print(
        "Avg. SSIM: {} \nAvg. PSNR: {} \nAvg. L1: {} \nAvg. L2: {} \n".format(
            ssim_tensor / count,
            psnr_tensor / count,
            l1_tensor / count,
            l2_tensor / count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--input_path',
                        default='/home/la_belva/PycharmProjects/Comparison_values/effects_ablation_tests/test_results_replace_sccp',
                        help='Path to output data', type=str)
    parser.add_argument('--gt_path', default='/home/la_belva/PycharmProjects/Comparison_values/val',
                        help='Path to ground truth data', type=str)
    # parser.add_argument('--fid_real_path', help='Path to real images when calculate FID', type=str)
    args = parser.parse_args()

    main(args.input_path, args.gt_path)
