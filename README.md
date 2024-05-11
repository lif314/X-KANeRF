# X-KANeRF: KAN-based NeRF with Different Basis Functions

### Is there any basis function can explain the NeRF formula?! 
$$\mathbf{c}, \sigma = F_{\Theta}(\mathbf{x}, \mathbf{d}),$$
 where $\mathbf{c}=(r,g,b)$ is RGB color, $\sigma$ is density, $\mathbf{x}$ is 3D position, $\mathbf{d}$ is the direction. 

Thanks to the excellent work of [KANeRF](https://github.com/Tavish9/KANeRF), I humbly utilized [Kolmogorov-Arnold Networks (KAN)](https://github.com/KindXiaoming/pykan) with different basis functions to fit the [NeRF](https://www.matthewtancik.com/nerf) equation. 

### The code is very COARSE, welcome any suggestions and criticism!

# X-KAN
| TODO | Basis Functions | Mathtype | Acknowledgement|
|:--------:|:---------:|:-------:|:------:|
| √ | B-Spline | $$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$| [Efficient-Kan](https://github.com/Blealtan/efficient-kan) |
| √ | Fourier | $$\phi_k(x) = \sin(2\pi kx), \phi_k(x) = \cos(2\pi kx)$$ | [FourierKAN](https://github.com/GistNoesis/FourierKAN/) |
| √ | Gaussian RBF | $$b_{i}(u)=\exp(-(u-u_i)^2)$$| [FastKAN](https://github.com/ZiyaoLi/fast-kan) |
| √ | RBF | - | [RGFKAN](https://github.com/sidhu2690/RBF-KAN) |
| √ | FCN | - | [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) |
| √ | FCN-Interpolation | - | [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) |
| - | Chebyshev Polynomials | $$\text{First Kind: }T_n(x) = \cos(n \cos^{-1}(x)), \\ \text{Second Kind: } U_n(x) = \frac{\sin((n+1)\cos^{-1}(x))}{\sin(\cos^{-1}(x))}$$ | [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) |


# Performance Comparision on `RTX-3090`

- `nerf_synthetic: lego / 30k`

|Model| Params $\downarrow$ |Train Rays/Sec $\uparrow$ | Train Time $\downarrow$ | FPS $\uparrow$ | PSNR $\uparrow$| SSIM $\uparrow$ | LPIPS $\downarrow$ | 
|:---:|:---:|:----:|:----:|:-----:|:-----:|:----:|:-----:|
|Nerfacto| 8192 | - | ~14m | 2.5| 33.69|0.973|0.0132|
|Nerfacto-Tiny| 2176 |- | ~13m | 2.5| 32.67 |0.962|0.0186|
|KAN: B-Spline|8092| ~37K | ~54 m|0.19|32.33|0.965|0.0174|
|KAN: G-RBF|3748 | ~115K | ~19 m |0.50|32.39|0.967|0.0172|
|KAN: RBF| 3512 | ~140K | ~15m |0.71|32.57|0.966| 0.0177|
|KAN: Fourier| 5222 | ~80K | ~25 m |0.42 | 31.72 |0.956|0.0241|
|KAN: FCN| - | ~4K | ~9h| - | - | -| - |
|KAN: FCN-Interpolation| 6912 | ~52K | ~40m| 0.21 | 32.67 | 0.965 | 0.0187 |

- `360_v2: garden / 30k`, todo


# Installation
```bash
# create python env
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

# install torch
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit

# install tinycudann
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install nerfstudio
pip install nerfstudio==0.3.4

# pip install torchmetrics==0.11.4

# Tab command
ns-install-cli

# !!! If you use `ns-process-data`, please install this version opencv
pip install opencv-python==4.3.0.36
```

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
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
	```bibtex
	@inproceedings{nerfstudio,
		title = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
		author = {
			Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
			and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
			Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
			Angjoo
		},
		year = 2023,
		booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
		series = {SIGGRAPH '23}
	}
	```