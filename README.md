# xKANeRF: KAN-based NeRF with Different Basis Functions

**Is there any basis function can explain the NeRF formula?!** 
$$\mathbf{c}, \sigma = F_{\Theta}(\mathbf{x}, \mathbf{d}),$$
 where $\mathbf{c}=(r,g,b)$ is RGB color, $\sigma$ is density, $\mathbf{x}$ is 3D position, $\mathbf{d}$ is the direction. 


[KAN: Kolmogorov-Arnold Networks](https://github.com/KindXiaoming/pykan) is a promising challenger to traditional MLPs. We're thrilled about integrating KAN into [NeRF](https://www.matthewtancik.com/nerf) based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)! 

# xKAN
| TODO | Basis Functions | Mathtype | Acknowledgement|
|:--------:|:---------:|:-------:|:------:|
| √ | B-Spline | $$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$| [Efficient-Kan](https://github.com/Blealtan/efficient-kan) |
| - | Fourier | $$\phi_k(x) = \sin(2\pi kx), \phi_k(x) = \cos(2\pi kx)$$ | [FourierKAN](https://github.com/GistNoesis/FourierKAN/) |
| √ | Gaussian RBF | $$b_{i}(u)=\exp(-(u-u_i)^2)$$| [FastKAN](https://github.com/ZiyaoLi/fast-kan) |
| - | Chebyshev Polynomials | $$\text{First Kind: }T_n(x) = \cos(n \cos^{-1}(x)), \\ \text{Second Kind: } U_n(x) = \frac{\sin((n+1)\cos^{-1}(x))}{\sin(\cos^{-1}(x))}$$ | [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) |
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
pip install nerfstudio==0.3.4

pip install torchmetrics==0.11.4

# Tab command
ns-install-cli 
```

# Performance Comparision on `RTX-3090`
- `nerf_synthetic: lego / 30k`

|Model| Params | Train Time | FPS | PSNR| SSIM | LPIPS | 
|:---:|:---:|:----:|:-----:|:-----:|:----:|:-----:|
|Nerfacto| - | 11 m, 35 s| 2.5| 33.69|0.973|0.0132|
|KAN: B-Spline|8092| 54 m, 13 s|0.19|32.33|0.965|0.0174|
|KAN: G-RBF|3748| 19 m, 37 s |0.50|32.39|0.967|0.01721|

# Docs
- [Universal Approximation Theorem vs. Kolmogorov–Arnold Theorem](docs/Theorem.md)


# Acknowledgement
- [KANeRF](https://github.com/Tavish9/KANeRF)
    ```bibtex
    @Manual{kanerf,
    title = {Hands-On NeRF with KAN},
    author = {Delin Qu, Qizhi Chen},
    year = {2024},
    url = {https://github.com/Tavish9/KANeRF},
    }
    ```