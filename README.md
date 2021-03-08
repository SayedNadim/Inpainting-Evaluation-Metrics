# Inpainting Evaluation Metrics
The goal of this repo is to provide a common evaluation script for image inpainting tasks. It contains some commonly used image quality metrics for inpainting (e.g., L1, L2, SSIM, PSNR and [LPIPS](https://github.com/richzhang/PerceptualSimilarity)). 

Pull requests and  corrections/suggestions will be cordially appreciated. 

### Please Note
- Images are scaled to [0,1]. If you need to change the data range, please make sure to change the data range in SSIM and PSNR.
- Number of generated images and ground truth images have to be exactly same. 
- I have resized the images to be (`256,256`). You can change the resolution based on your needs.
- Please make sure that all the images (generated and gt images) are in the corresponding folders. Currently,it can not calculate metrics if there are sub-folders. I will update the code to calculate for sub-folders as well. 
- LPIPS is a bit slow. So, if you have lots of images, it might take a lot of time. (For ~1000 images, it took around ~15-20 minutes on my personal setup (1 TitanXP). Other metrics are fast and took around ~20 seconds to compute.)

### Requirements
- PyTorch ( `>= 1.0` )
- Python ( `>=3.5` )
- [PyTorch Image Quality (PIQ)](https://github.com/photosynthesis-team/piq) ( `$ pip install piq` )

### Usage
- Usable Arguments
  - `--input_path` - path to your generated images (required).
  - `--gt_path` - path to your ground truth images (required).
  - `--batch_size` - batch size you want to use (Default to 4).
  - `--image_width` - width of the image (both generated image and ground truth images will be resized to this width. Default to 256).
  - `--image_height` - width of the image (both generated image and ground truth images will be resized to this width. Default to 256).
  - `--threads` - threads to be used for multi-processing (Default to 4).


- Please provide paths of the folders (i.e., folder of generated images and folder of ground truth images).

    `python main.py --input_path path/to/generated/images --gt_path path/to/ground/truth/images`

- If you need to save it in a `.txt` file, then simply run

    `python main.py --input_path path/to/generated/images --gt_path path/to/ground/truth/images >> results.txt`

### To-do
- [x] L1
- [x] L2
- [x] SSIM
- [x] PSNR
- [x] LPIPS
- [ ] FID
- [ ] IS

### Acknowledgement
Thanks to [PhotoSynthesis Team](https://github.com/photosynthesis-team/piq) for the wonderful implementation of the metrics. Please cite accordingly if you use PIQ for the evaluation.

Cheers!!
