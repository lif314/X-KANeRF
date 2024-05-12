# X-KANeRF: KAN-based NeRF with Different Basis Functions

### Is there any basis function can explain the NeRF formula?! 
$$\mathbf{c}, \sigma = F_{\Theta}(\mathbf{x}, \mathbf{d}),$$
 where $\mathbf{c}=(r,g,b)$ is RGB color, $\sigma$ is density, $\mathbf{x}$ is 3D position, $\mathbf{d}$ is the direction. 

Thanks to the excellent work of [KANeRF](https://github.com/Tavish9/KANeRF), I utilize [Kolmogorov-Arnold Networks (KAN)](https://github.com/KindXiaoming/pykan) with different basis functions to fit the [NeRF](https://www.matthewtancik.com/nerf) equation based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). 

### The code is very COARSE, welcome any suggestions and criticism!

# [X-KAN Models](./xKANeRF/xKAN/)
| TODO | Basis Functions | Mathtype | Acknowledgement|
|:--------:|:---------:|:-------:|:------:|
| √ | [B-Spline](https://en.wikipedia.org/wiki/B-spline) | $$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$| [Efficient-Kan](https://github.com/Blealtan/efficient-kan) |
| √ | [Fourier](https://en.wikipedia.org/wiki/Fourier_transform) | $$\phi_k(x) = \sin(2\pi kx), \phi_k(x) = \cos(2\pi kx)$$ | [FourierKAN](https://github.com/GistNoesis/FourierKAN/) |
| √ | [Gaussian RBF](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) | $$\phi(x, c) = e^{-\frac{\|x - c\|^2}{2\sigma^2}}$$| [FastKAN](https://github.com/ZiyaoLi/fast-kan) |
| √ | [Radial Basis Function](https://en.wikipedia.org/wiki/Radial_basis_function) | $$\phi(x, c) = f(\|x - c\|)$$ | [RGFKAN](https://github.com/sidhu2690/RBF-KAN) |
| √ | FCN | - | [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) |
| √ | FCN-Interpolation | - | [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) |
| √ | [1st Chebyshev Polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) | $$T_n(x) = \cos(n \cos^{-1}(x))$$ | [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) |
| √ | [2nd-Chebyshev Polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) | $$U_n(x) = \frac{\sin((n+1)\cos^{-1}(x))}{\sin(\cos^{-1}(x))}$$ | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| √ | [Jacobi polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials) | $$P_n^{(\alpha, \beta)}(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} \left[ (1-x)^{\alpha+n} (1+x)^{\beta+n} \right]$$ | [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN) |
| √ | [Hermite polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials)  | $$H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n}(e^{-x^2})$$  | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| √ | [Gegenbauer polynomials](https://en.wikipedia.org/wiki/Gegenbauer_polynomials) |$$C_{n+1}^{(\lambda)}(x) = \frac{2(n+\lambda)}{n+1}x C_n^{(\lambda)}(x) - \frac{(n+2\lambda-1)}{n+1}C_{n-1}^{(\lambda)}(x)$$| [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| √ | [Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials) | $$P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} \left( x^2 - 1 \right)^n$$  | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| - | [Laguerre polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials) | $$L_n(x) = \frac{e^x}{n!} \frac{d^n}{dx^n} \left( x^n e^{-x} \right)$$ | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| √ | [Bessel polynomials](https://en.wikipedia.org/wiki/Bessel_polynomials)  | $$J_n(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k!(n+k)!} \left( \frac{x}{2} \right)^{2k+n}$$  | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| √ | [Fibonacci polynomials](https://en.wikipedia.org/wiki/Fibonacci_polynomials) | $$F_n(x) = xF_{n-1}(x) + F_{n-2}(x), \quad \text{for } n \geq 2.$$ | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| More and More!!! | - | - | -|


# Performance Comparision on `RTX-3090`

**Model Setting** -> [train_blender.sh](https://github.com/lif314/X-KANeRF/blob/main/train_blender.sh)
|hidden_dim| hidden_dim_color |num_layers | num_layers_color | geo_feat_dim | appearance_embed_dim |
|:---:|:---:|:----:|:----:|:-----:|:-----:|
| 8 | 8 | 1 | 1 | 7 | 8|

- `nerf_synthetic: lego / 30k`

|Model| Layer Params $\downarrow$ |Train Rays/Sec $\uparrow$ | Train Time $\downarrow$ | FPS $\uparrow$ | PSNR $\uparrow$| SSIM $\uparrow$ | LPIPS $\downarrow$ | 
|:---:|:---:|:----:|:----:|:-----:|:-----:|:----:|:-----:|
|[Nerfacto-MLP](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/nerfacto_mlp.py)| 456 | ~200K | ~13m | 1.09| 31.90 |0.961|0.0207|
|[BSplines-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/bspine_kan.py)|8092| ~37K | ~54 m|0.19|32.33|0.965|0.0174|
|[GRBF-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/grbf_kan.py)|3748 | ~115K | ~19 m |0.50|32.39|0.967|0.0172|
|[RBF-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/rbf_kan.py)| 3512 | ~140K | ~15m |0.71|32.57|0.966| 0.0177|
|[Fourier-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fourier_kan.py)| 5222 | ~80K | ~25 m |0.42 | 31.72 |0.956|0.0241|
|[FCN-KAN(Iters: 4k)](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fcn_kan.py)| 5184 | ~4K | ~90m | 0.02 | 29.67 | 0.938 | 0.0401 |
|[FCN-Interpolation-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fcn_kan.py)| 6912 | ~52K | ~40m| 0.21 | 32.67 | 0.965 | 0.0187 |
|[1st Chebyshev-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/chebyshev_kan.py) | 4396 | ~53K | ~40m| 0.34 | 28.56| 0.924 | 0.0523 |
|[Jacobi-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/jacobi_kan.py) | 3532 | ~72K | ~30m| 0.37 | 27.88 | 0.915 |0.0553|
|[Bessel-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/bessel_kan.py) | 3532 | ~76K | ~28m| 0.33 | 25.79 | 0.878 |0.1156|
|[2nd Chebyshev-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/chebyshev2_kan.py) |  | ~55K | ~39m| 0. |  | 0. |0.|
|[Fibonacci-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fibonacci_kan.py) |  | ~K | ~m| 0. |  | 0. |0.|
|[Gegenbauer-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/gegenbauer_kan.py) |  | ~K | ~m| 0. |  | 0. |0.|
|[Hermite-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/xKANeRF/xKAN/hermite_kan.py) |  | ~K | ~m| 0. |  | 0. |0.|
|[Legendre-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/xKANeRF/xKAN/legendre_kan.py) |  | ~K | ~m| 0. |  | 0. |0.|


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

# Run
```bash
############# kan_basis_type #############
# mlp, bspline, grbf, rbf, fourier,
# fcn, fcn_inter, chebyshev, jacobi
# bessel, chebyshev2, finonacci, hermite
# legendre
bash train_blender.sh
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