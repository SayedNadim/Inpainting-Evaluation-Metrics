import torch
import piq
import tqdm
import time
import argparse
import torch.nn as nn
from read_datasets import build_dataloader

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@torch.no_grad()
def main():
    psnr_tensor = 0
    ssim_tensor = 0
    l1_tensor = 0
    l2_tensor = 0
    # lpips_tensor = 0
    count = 0
    t0 = time.time()
    print("Calculating image quality metrics ...")
    for batch in data_loader:
        img_batch, gt_batch = batch
        if torch.cuda.is_available():
            # Move to GPU to make computaions faster
            img_batch = img_batch.cuda()
            gt_batch = gt_batch.cuda()

        for i in range(gt_batch.shape[0]):
            gt, img = gt_batch[i], img_batch[i]

            # MS-SIM
            ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(gt, img, data_range=1.)
            # PSNR
            psnr_index: torch.Tensor = piq.psnr(gt, img, data_range=1., reduction='mean')
            # L1 Error
            l1_index = nn.L1Loss(reduction='mean')(gt, img)
            # L1 Error
            l2_index = nn.MSELoss(reduction='mean')(gt, img)
            # LPIPS
            # lpips_loss: torch.Tensor = piq.LPIPS(reduction='mean')(gt, img)

            # Adding for computing average value
            ssim_tensor += ms_ssim_index
            psnr_tensor += psnr_index
            l1_tensor += l1_index
            l2_tensor += l2_index
            # lpips_tensor += lpips_loss.item()

            count += 1

    t1 = time.time()

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
    print(count)
    print("Average processing time for each image (of total {} images): {} s".format(count, (t1 - t0) / count))


if __name__ == '__main__':
    # Parsing Arguments
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--input_path',
                        default='/home/la_belva/PycharmProjects/Comparison_models/generative_inpainting/test_results_gated_full_places/output/ff_mask/mask_40_50',
                        help='Path to output data', type=str)
    parser.add_argument('--gt_path', default='/home/la_belva/PycharmProjects/Comparison_values/places_val_small',
                        help='', type=str)
    parser.add_argument('--batch_size',
                        help='Batch Size for dataloader. Default = 4', type=int,
                        default=4)
    parser.add_argument('--image_width', type=int,
                        help='Image width for ground truth images and generated images. Default = 256', default=256)
    parser.add_argument('--image_height', type=int,
                        help='Image height for ground truth images and generated images. Default = 256', default=256)
    parser.add_argument('--threads',
                        help='Threads for multi-processing. Default = 4', type=int,
                        default=4)
    args = parser.parse_args()

    # Datasets
    print('===> Loading datasets')
    data_loader = build_dataloader(
        img_path=args.input_path,
        gt_path=args.gt_path,
        batch_size=args.batch_size,
        num_workers=args.threads,
        shuffle=False,
        image_shape=(args.image_height, args.image_width)
    )
    print('===> Loaded datasets')

    # Main function
    main()
