# Deep Polarization Cues for Single-shot Shape and Subsurface Scattering Estimation


[Chenaho Li](https://ligoudaner377.github.io/), [Ngo Thanh Trung](http://www.am.sanken.osaka-u.ac.jp/~trung/), [Hajime Nagahara](https://www.is.ids.osaka-u.ac.jp/en/authors/hajime-nagahara/)

Osaka University

This project is the implementation of "Deep Polarization Cues for Single-shot Shape and Subsurface Scattering Estimation (ECCV 2024)".

[Paper](https://arxiv.org/abs/2407.08149) | [Data](https://drive.google.com/drive/folders/1RY1nZpi99fujUGfXC4JuNKslzSZvvF8I?usp=sharing) 

![teaser image](/imgs/teaser.png)
## Requirements

* Linux
* NVIDIA GPU + CUDA CuDNN
* Python 3 (tested on 3.8.8)
* torch (tested on 1.8.1)
* scipy 
* pandas
* torchvision
* dominate
* visdom
* pillow

## How to use

### Train
- Download the [dataset](https://drive.google.com/drive/folders/1RY1nZpi99fujUGfXC4JuNKslzSZvvF8I?usp=sharing). 
- Unzip it to ./datasets/
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train the first stage (change the gpu_ids based on your device)
```bash
python ./train.py --dataroot "./datasets/translucent" --name "4pol_minmax_reconstruct" --model "pol_shape_illu" --checkpoints_dir "./checkpoints" --input_list "4pol min max" --gpu_ids 0,1,2,3 
```
- Train the second stage 
```bash
python ./train.py --dataroot "./datasets/translucent" --name "4pol_minmax_reconstruct" --model "pol_sss"  --checkpoints_dir "./checkpoints" --use_reconstruction_loss --input_list "4pol min max" --gpu_ids 0,1,2,3
```

### Test on the synthetic dataset
- Dowload the [trained model](https://drive.google.com/file/d/1iin_0F3mXzwiGr_M5_OdhfeX3wiOz734/view?usp=drive_link) to ./checkpoints (skip this step if you have already trained the model)
```bash
python ./test.py --dataroot "./datasets/translucent" --name "4pol_minmax_reconstruct" --model "pol_sss" --results_dir "./results"  --input_list "4pol min max" --eval
```

### Test on real-world objects
```bash
python ./inference_real.py --dataroot "./datasets/real_sample" --dataset_mode 'real' --name "4pol_minmax_reconstruct" --model "pol_sss" --results_dir "./results" --input_list  "4pol min max" --eval 
```


### scripts.sh integrates all commands
```bash
bash ./scripts.sh
```

## Acknowledgements

Code derived and modified from:

- [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix")

