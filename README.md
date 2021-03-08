# Inpainting Evaluation Metrics
The goal of this repo is the provide a common evaluation script for image inpainting tasks. It contains some of the commonly used image quality metrics for inpainting (e.g., L1, L2, SSIM, PSNR and [LPIPS](https://github.com/richzhang/PerceptualSimilarity)). 

**Note
- Images are scaled to [0,1]. If you need to change the data range, please make sure to change the data range in SSIM and PSNR.
- Number of generated images and ground truth images have to be exactly same. 
- I have resized the images to be (256,256). You can change the resolution based on your needs.
- Please make sure that all the images (generated and gt images) are in the corresponding folders. Currently,it can not calculate metrics if there are sub-folders. I will update the code to calculate for sub-folders as well. 
- LPIPS is a bit slow. So, if you have lots of images, it might take a lot of time. For `~1000 images`, it took around 15 minutes on my personal setup (1 TitanXP). Other metrics are fast and took around 5 seconds to compute. 

### Requirements
- PyTorch ( `>= 1.0` )
- Python ( `>=3.5` )
- piq ( `$ pip install piq` )

### Usage

- Please provide paths of the folders (i.e., folder of generated images and folder of ground truth images).

`python quality_metrics.py --input_path path/to/generated/images --gt_path path/to/ground/truth/images`

- If you need to save it in a `.txt` file, then simply run

`python quality_metrics.py --input_path path/to/generated/images --gt_path path/to/ground/truth/images >> results.txt`

### To-do
- [x] L1
- [x] L2
- [x] SSIM
- [x] PSNR
- [x] LPIPS
- [ ] FID
- [ ] IS

### Optional
I have two scripts to generate free-form masks and cellur automata masks for inpainting tasks. If you use, please cite the respective authors.
- `ca_mask_generation.py` generates [cellular automata masks](https://arxiv.org/abs/2010.01110) and saves the corresponding masks based on mask-ratios. 
- `ff_mask_generation.py` file generates [free-form masks](https://github.com/JiahuiYu/generative_inpainting) and saves the corresponding masks based on mask-ratios.

### Acknowledgement
Thank you [PhotoSynthesis Team](https://github.com/photosynthesis-team/piq) for the wonderful implementation of the metrics.
