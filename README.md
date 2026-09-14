
### 1. Dependences
 Create a conda virtual environment and activate it.
 1) conda create --name MOD python=3.9
 2) conda activate MOD
 3) pip install -r requirements.txt

### 2. Datasets download
Download these datasets and create a dataset folder to hold them.
1) FLIR dataset: [FLIR](https://drive.google.com/file/d/1o9lchkdQcPaYqqEa_d_6l3QewyfkDTCx/view?usp=drive_link)
2) LLVIP dataset: [LLVIP](https://drive.google.com/file/d/1Bl1_D1T2x4JLu4__VbBjn6WJ3-T1Z99W/view?usp=drive_link)
3) M3FD dataset: [M3FD](https://drive.google.com/file/d/1FSfAQQ80UvwE7mXKDAxZZnabUrsM9HHD/view?usp=drive_link)
4) MFAD dataset: [MFAD](https://drive.google.com/file/d/1BusF0_NY3pahjZLTqwXYYcKWQ8nSXy97/view?usp=drive_link)

### 3. Pretrained weights
Download our LCAFNet weights and create a weights folder to hold them.
1) FLIR dataset: [LCAFNet_FLIR.pt](https://drive.google.com/file/d/1M6ZAq_ZMQa4_zoJ2zdh9GlXh9t8vimRj/view?usp=drive_link)
2) LLVIP dataset: [LCAFNet_LLVIP.pt](https://drive.google.com/file/d/1lA1-VQmHa6J81j_bagl957zNpRHRahIv/view?usp=drive_link)
3) M3FD dataset: [LCAFNet_M3FD.pt](https://drive.google.com/file/d/14HxymFlpwtq4eVr8-gn5QoivRs7AYieJ/view?usp=drive_link)
4) MFAD dataset: [LCAFNet_MFAD.pt](https://drive.google.com/file/d/1zqBAwpv7eJCY5GYvwz4QKyOVgXMcGFq7/view?usp=drive_link)

### 4. Training our LCAFNet
Dataset path, GPU, batch size, etc., need to be modified according to different situations.
```
python train.py
```

The current default is **Group D fine-tuning** on M3FD. It loads the converged
no-prior `exp_rgca_mask_gfb_uniform_lr/weights/best.pt`, initializes only the
304-parameter BECP controller, calibrates evidence for five prior-free epochs,
ramps the prior over the next ten epochs, and jointly fine-tunes for 50 epochs.
SGD differential learning rates are used and AFSS is disabled. Each run writes
its exact treatment to `EXPERIMENT_IDENTITY.md`.

### 5. Test our LCAFNet

```
python test.py
```


