import os
import torch
import piq
import tqdm
import time
import argparse
import torch.nn as nn
import logging


from read_datasets import build_dataloader, natural_keys

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def init_logging(log_filename: str):
    rootLogger = logging.getLogger('quality metrics')

    LOG_DIR = os.getcwd() + '/' + log_filename
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    fileHandler = logging.FileHandler("{0}/{1}.log".format(LOG_DIR, "g2"))
    rootLogger.addHandler(fileHandler)

    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)

    return rootLogger


@torch.no_grad()
def main(logger):
    psnr_tensor = 0
    ssim_tensor = 0
    l1_tensor = 0
    l2_tensor = 0
    # lpips_tensor = 0
    count = 0
    dataset_name = None
    t0 = time.time()
    print("Calculating image quality metrics ...")
    for batch in data_loader:
        img_batch, gt_batch, dataset_name = batch
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

            # if count % 100 ==0 and count !=0:
            #     print("Processed {} images.. Please wait..".format(count))

    t1 = time.time()

    # print(
    #     "Avg. LPIPS: {} \nAvg. SSIM: {} \nAvg. PSNR: {} \nAvg. L1: {} \nAvg. L2: {} \n".format(lpips_tensor / count,
    #                                                                                            ssim_tensor / count,
    #                                                                                            psnr_tensor / count,
    #                                                                                            l1_tensor / count,
    #                                                                                            l2_tensor / count))

    print(
        "Image quality statistics for {} dataset....\nAvg. SSIM: {} \nAvg. PSNR: {} \nAvg. L1: {} \nAvg. L2: {} \n".format(
            dataset_name,
            ssim_tensor / count,
            psnr_tensor / count,
            l1_tensor / count,
            l2_tensor / count))
    # logger.debug(
    #     "Image quality statistics for {} dataset....\nAvg. SSIM: {} \nAvg. PSNR: {} \nAvg. L1: {} \nAvg. L2: {} \n".format(
    #         dataset_name,
    #         ssim_tensor / count,
    #         psnr_tensor / count,
    #         l1_tensor / count,
    #         l2_tensor / count))
    # print("Average processing time for each image (of total {} images): {} s".format(count, (t1 - t0) / count))


if __name__ == '__main__':
    # Parsing Arguments
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--input_path',
                        default='/media/la_belva/E1/output_full_gla_places/output/',
                        help='Path to output data', type=str)
    parser.add_argument('--gt_path', default='/home/la_belva/PycharmProjects/Comparison_values/places_val_small',
                        help='Path to gt data', type=str)
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

    list_of_subfolders = [dI for dI in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, dI))]
    # print(list_of_subfolders)
    ca_subfolders = [dI for dI in os.listdir(args.input_path + list_of_subfolders[0]) if
                     os.path.isdir(os.path.join(args.input_path + list_of_subfolders[0], dI))]
    ff_subfolders = [dI for dI in os.listdir(args.input_path + list_of_subfolders[1]) if
                     os.path.isdir(os.path.join(args.input_path + list_of_subfolders[1], dI))]
    ca_dirs_len = len(ca_subfolders)
    ff_dirs_len = len(ff_subfolders)
    ca_individual_foders_list = []
    ff_individual_foders_list = []
    for i in range(ca_dirs_len):
        ca_individual_foders_list.append(args.input_path + list_of_subfolders[0] + '/' + ca_subfolders[i])
    for j in range(ca_dirs_len):
        ff_individual_foders_list.append(args.input_path + list_of_subfolders[1] + '/' + ff_subfolders[j])

    ca_individual_foders_list.sort(key=natural_keys)
    ff_individual_foders_list.sort(key=natural_keys)

    # print(ca_individual_foders_list)
    # print(ff_individual_foders_list)

    for kk in range(len(ca_individual_foders_list)):
        # Datasets
        print("\n")
        print("#" * 80)
        print('===> Loading datasets for {}'.format(str(ca_individual_foders_list[kk])))
        data_loader = build_dataloader(
            img_path=ca_individual_foders_list[kk],
            gt_path=args.gt_path,
            batch_size=args.batch_size,
            num_workers=args.threads,
            shuffle=False,
            image_shape=(args.image_height, args.image_width),
            dataset_name=ca_individual_foders_list[kk],
            return_dataset_name=True
        )
        print('===> Loaded datasets')

        # Main function
        logger = init_logging(ca_individual_foders_list[kk])
        main(logger)

    for jj in range(len(ff_individual_foders_list)):
        # Datasets
        print('===> Loading datasets for {}'.format(str(ff_individual_foders_list[jj])))
        data_loader = build_dataloader(
            img_path=ff_individual_foders_list[jj],
            gt_path=args.gt_path,
            batch_size=args.batch_size,
            num_workers=args.threads,
            shuffle=False,
            image_shape=(args.image_height, args.image_width),
            dataset_name=ff_individual_foders_list[jj],
            return_dataset_name=True
        )
        print('===> Loaded datasets')

        # Main function
        logger = init_logging(ff_individual_foders_list[jj])
        main(logger)
