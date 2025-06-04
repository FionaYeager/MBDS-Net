This is code for paper “MBDS-Net: A Multi-Scale Boundary-Enhanced Denoising Diffusion Network for Medical Image Segmentation”
=====
1.Prepare data
-----
Download REFUGE dataset from https://refuge.grand-challenge.org/. Your dataset folder under "data" should be like:<br>
```
/data/REFUGE/
├── train/
│   ├── imgs/
│   │   ├── case1.png
│   │   ├── case2.png
│   │   └── ...
│   └── masks/
│       ├── case1_mask.png
│       ├── case2_mask.png
│       └── ...
├── val/
│   ├── imgs/
│   │   ├── case1.png
│   │   ├── case2.png
│   │   └── ...
│   └── masks/
│       ├── case1_mask.png
│       ├── case2_mask.png
│       └── ...
└── test/
    ├── imgs/
    │   ├── case1.png
    │   ├── case2.png
    │   └── ...
    └── masks/
        ├── case1_mask.png
        ├── case2_mask.png
        └── ...
```
2.Data augmentation parameters
-----
- Histogram Equalization  
  - Applied to: Input RGB images  
  - Method: `ImageOps.equalize` applied separately on R, G, B channels  
  - Purpose: Enhance contrast before further processing.

- Resizing  
  - Applied to: Both images and masks  
  - Size: 256 × 256  
  - Purpose: Standardize input size for training and evaluation.

- Random Horizontal Flip  
  - Applied during: Training (if `transform=True`)  
  - Probability: 0.5  
  - Affects: Both image and mask simultaneously  
  - Implementation: `data_transforms.RandomHorizontalFlip(0.5)`

- Normalization  
  - Applied to: Input images only  
  - Mean: `[0.485, 0.456, 0.406]`  
  - Std: `[0.229, 0.224, 0.225]`  
  - Performed after: `ToTensor()`

- Mask Preprocessing  
  - Conversion: 0 (optic cup) → 1, 255 (background) → 0  
  - Type: Single-channel float tensor  
  - Purpose: Prepare binary mask for segmentation.

3.Training
-----
```
python scripts/segmentation_train_my.py \
  --data_dir /data/REFUGE/train \
  --val_dir /data/REFUGE/val \
  --out_dir /model_save/REFUGE/ \
  --batch_size 8 \
  --lr 1e-4 \
  --use_fp16 False \
  --gpu_dev 0 \
  --schedule_sampler uniform \
  --save_interval 100 \
  --log_interval 100
```
4.Testing
-----
```
python scripts/segmentation_sample_my.py \
  --data_dir /data/REFUGE/test \
  --batch_size 8 \
  --model_path /model_save/REFUGE/xxxx.pth \
  --num_ensemble 5 \
  --gpu_dev 0 \
  --out_dir /test_result/REFUGE/ \
```
4.Reference
-----
5.Cite
-----
