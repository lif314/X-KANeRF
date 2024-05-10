# xKANeRF: KAN-based NeRF with Different Basis Functions

[KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) is a promising challenger to traditional MLPs. We're thrilled about integrating KAN into [NeRF](https://www.matthewtancik.com/nerf)! Is KAN suited for **view synthesis** tasks? What challenges will we face? How will we tackle them? We provide our initial observations and future discussion!

# XKAN
| Done | Basis Functions | Mathtype |
|--------|---------| -------------|
| √ | B-Spline | $$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$|
| - | Fourier | $$\phi_k(x) = \sin(2\pi kx), \phi_k(x) = \cos(2\pi kx)$$ |
| √ | Gaussian RBF | $$b_{i}(u)=\exp(-(u-u_i)^2)$$|
| - | Spherical Harmonics | $$Y_l^m(\theta, \phi) = \sqrt{\frac{2l + 1}{4\pi} \frac{(l - m)!}{(l + m)!}} P_l^m(\cos\theta) e^{im\phi}$$|



# Installation

```bash
# create python env
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

# install torch
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# install tinycudann
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install nerfstudio
pip install nerfstudio
```

# Performance Comparision
