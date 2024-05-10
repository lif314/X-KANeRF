# xKANeRF: KAN-based NeRF with Different Basis Functions

[KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) is a promising challenger to traditional MLPs. We're thrilled about integrating KAN into [NeRF](https://www.matthewtancik.com/nerf)! Is KAN suited for **view synthesis** tasks? What challenges will we face? How will we tackle them? We provide our initial observations and future discussion!

# XKAN
| TODO | Basis Functions | Mathtype | Acknowledgement|
|:--------:|:---------:|:-------:|:------:|
| √ | B-Spline | $$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$| [Efficient-Kan](https://github.com/Blealtan/efficient-kan) |
| - | Fourier | $$\phi_k(x) = \sin(2\pi kx), \phi_k(x) = \cos(2\pi kx)$$ | [FourierKAN](https://github.com/GistNoesis/FourierKAN/) |
| √ | Gaussian RBF | $$b_{i}(u)=\exp(-(u-u_i)^2)$$| [FastKAN](https://github.com/ZiyaoLi/fast-kan) |
| - | Chebyshev Polynomials | $$\text{First Kind: }T_n(x) = \cos(n \cos^{-1}(x))\\ \text{Second Kind: } U_n(x) = \frac{\sin((n+1)\cos^{-1}(x))}{\sin(\cos^{-1}(x))}$$ | [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) |
|- | FCN | - | [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) |


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
