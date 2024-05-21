# X-KANeRF [KANeRF-benchmarking]: KAN-based NeRF with Various Basis Functions

### Is there any basis function can explain the NeRF formula?! 
$$\mathbf{c}, \sigma = F_{\Theta}(\mathbf{x}, \mathbf{d}),$$
 where $\mathbf{c}=(r,g,b)$ is RGB color, $\sigma$ is density, $\mathbf{x}$ is 3D position, $\mathbf{d}$ is the direction. 

To explore this issue, I used [Kolmogorov-Arnold Networks (KAN)](https://github.com/KindXiaoming/pykan) with various basis functions to fit the [NeRF](https://www.matthewtancik.com/nerf) equation based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). 

### The code might be a bit COARSE, any suggestions and criticisms are welcome!

# [X-KAN Models](./xKANeRF/xKAN/) (Here are various KANs!)
| TODO | Basis Functions | Mathtype | Acknowledgement|
|:--------:|:---------:|:-------:|:------:|
| 1 | [B-Spline](https://en.wikipedia.org/wiki/B-spline) | $$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$| [Efficient-Kan](https://github.com/Blealtan/efficient-kan) |
| 2 | [Fourier](https://en.wikipedia.org/wiki/Fourier_transform) | $$\phi_k(x) = \sin(2\pi kx), \phi_k(x) = \cos(2\pi kx)$$ | [FourierKAN](https://github.com/GistNoesis/FourierKAN/) |
| 3 | [Gaussian RBF](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) | $$\phi(x, c) = e^{-\frac{\|x - c\|^2}{2\sigma^2}}$$| [FastKAN](https://github.com/ZiyaoLi/fast-kan) |
| 4 | [Radial Basis Function](https://en.wikipedia.org/wiki/Radial_basis_function) | $$\phi(x, c) = f(\|x - c\|)$$ | [RBFKAN](https://github.com/sidhu2690/RBF-KAN) |
| 5 | FCN | - | [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) |
| 6 | FCN-Interpolation | - | [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) |
| 7 | [1st Chebyshev Polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) | $$T_n(x) = \cos(n \cos^{-1}(x))$$ | [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) |
| 8 | [2nd-Chebyshev Polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) | $$U_n(x) = \frac{\sin((n+1)\cos^{-1}(x))}{\sin(\cos^{-1}(x))}$$ | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| 9 | [Jacobi polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials) | $$P_n^{(\alpha, \beta)}(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} \left[ (1-x)^{\alpha+n} (1+x)^{\beta+n} \right]$$ | [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN) |
| 10 | [Hermite polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials)  | $$H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n}(e^{-x^2})$$  | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| 11 | [Gegenbauer polynomials](https://en.wikipedia.org/wiki/Gegenbauer_polynomials) |$$C_{n+1}^{(\lambda)}(x) = \frac{2(n+\lambda)}{n+1}x C_n^{(\lambda)}(x) - \frac{(n+2\lambda-1)}{n+1}C_{n-1}^{(\lambda)}(x)$$| [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| 12 | [Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials) | $$P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} \left( x^2 - 1 \right)^n$$  | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| 13 | [Laguerre polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials) | $$L_n(x) = \frac{e^x}{n!} \frac{d^n}{dx^n} \left( x^n e^{-x} \right)$$ | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| 14 | [Bessel polynomials](https://en.wikipedia.org/wiki/Bessel_polynomials)  | $$J_n(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k!(n+k)!} \left( \frac{x}{2} \right)^{2k+n}$$  | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| 15 | [Fibonacci polynomials](https://en.wikipedia.org/wiki/Fibonacci_polynomials) | $$F_n(x) = xF_{n-1}(x) + F_{n-2}(x), \quad \text{for } n \geq 2.$$ | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
| 16 | [Lucas polynomials](https://en.wikipedia.org/wiki/Fibonacci_polynomials) | $$L_n(x) = xL_{n-1}(x) + L_{n-2}(x)$$ | [OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) |
|  17 | [Mexican hat wavelet](https://en.wikipedia.org/wiki/Ricker_wavelet) | $$\psi(x) = \frac{2}{\sqrt{3a}\pi^{\frac{1}{4}}} \left(1 - \frac{x^2}{a^2}\right) e^{-\frac{x^2}{2a^2}}$$ | [Wav-KAN](https://github.com/zavareh1/Wav-KAN)|
|  18 | [Morlet wavelet (Gabor wavelet)](https://en.wikipedia.org/wiki/Morlet_wavelet) |  $$\psi(t) = \pi^{-\frac{1}{4}} e^{i\omega_0 t} e^{-\frac{t^2}{2}}$$| [Wav-KAN](https://github.com/zavareh1/Wav-KAN)|
|  19 | [Difference of Gaussians(DoG)](https://en.wikipedia.org/wiki/Difference_of_Gaussians) |  $$\text{DoG}(x, y) = \frac{1}{\sqrt{2\pi}\sigma_1} e^{-\frac{x^2 + y^2}{2\sigma_1^2}} - \frac{1}{\sqrt{2\pi}\sigma_2} e^{-\frac{x^2 + y^2}{2\sigma_2^2}}$$| [Wav-KAN](https://github.com/zavareh1/Wav-KAN)|
|  20 | [Meyer wavelet](https://en.wikipedia.org/wiki/Meyer_wavelet) |  $$\psi(x) = \sqrt{\frac{2}{T}} \sum_{k=1}^{N} \left(1 - \left(\frac{k}{N}\right)^2\right) \left[ \cos\left(\frac{2\pi x k}{T}\right) - \frac{\sin(\pi x k / T)}{\pi x k / T}\right]$$| [Wav-KAN](https://github.com/zavareh1/Wav-KAN)|
|  21  | [Shannon wavelet](https://en.wikipedia.org/wiki/Shannon_wavelet) |  $$\psi(t) = \frac{\sin(\pi t) - \sin\left(\frac{\pi t}{2}\right)}{\pi t}$$| [Wav-KAN](https://github.com/zavareh1/Wav-KAN)|
|  22 | [Bump wavelet](https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html) | $$\psi(t) = e^{-\frac{1}{1 - t^2}}$$| [Wav-KAN](https://github.com/zavareh1/Wav-KAN)|
| More and More!!! | - | - | -|


# Performance Comparision on `RTX-3090`

- **Model Setting** -> [train_blender.sh](https://github.com/lif314/X-KANeRF/blob/main/train_blender.sh)

|Model|hidden_dim| hidden_dim_color | num_layers | num_layers_color | geo_feat_dim | appearance_embed_dim |
|:----:|:---:|:---:|:----:|:----:|:-----:|:-----:|
Nefacto-MLP-A| 32 | 32 | 2 | 2 | 7 | 8 |
Nefacto-MLP-B| 8 | 8 | 8 | 8 | 7 | 8 |
Others| 8 | 8 | 1 | 1 | 7 | 8|


- `nerf_synthetic: lego / 30k`
> Note that the current `Train Rays/Sec` and `Train Time(ETA Time)` are not accurate, they are the values ​​when the number of iterations reaches 100.

|Model| Layer Params $\downarrow$ |Train Rays/Sec $\uparrow$ | Train Time $\downarrow$ | FPS $\uparrow$ | PSNR $\uparrow$| SSIM $\uparrow$ | LPIPS $\downarrow$ | 
|:---:|:---:|:----:|:----:|:-----:|:-----:|:----:|:-----:|
|[Nerfacto-MLP](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/nerfacto_mlp.py)-A| 9902 | ~170K | ~14m | 0.71 | 32.53 | 0.968 | 0.0167 |
|[Nerfacto-MLP](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/nerfacto_mlp.py)-B | 3382 | ~165K | ~14m | 0.75 | 27.11 | 0.915 | 0.0621 |
|[Nerfacto-MLP](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/nerfacto_mlp.py)| 1118 | ~190K | ~13m | 0.99| 28.60 |0.952 |0.0346 |
|[BSplines-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/bspine_kan.py)|8092| ~37K | ~54 m|0.19|32.33|0.965|0.0174|
|[GRBF-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/grbf_kan.py)|3748 | ~115K | ~19 m |0.50|32.39|0.967|0.0172|
|[RBF-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/rbf_kan.py)| 3512 | ~140K | ~15m |0.71|32.57|0.966| 0.0177|
|[Fourier-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fourier_kan.py)| 5222 | ~80K | ~25 m |0.42 | 31.72 |0.956|0.0241|
|[FCN-KAN(Iters: 4k)](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fcn_kan.py)| 5184 | ~4K | ~90m | 0.02 | 29.67 | 0.938 | 0.0401 |
|[FCN-Interpolation-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fcn_kan.py)| 6912 | ~52K | ~40m| 0.21 | 32.67 | 0.965 | 0.0187 |
|[1st Chebyshev-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/chebyshev_kan.py) | 4396 | ~53K | ~40m| 0.34 | 28.56| 0.924 | 0.0523 |
|[Jacobi-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/jacobi_kan.py) | 3532 | ~72K | ~30m| 0.37 | 27.88 | 0.915 |0.0553|
|[Bessel-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/bessel_kan.py) | 3532 | ~76K | ~28m| 0.33 | 25.79 | 0.878 |0.1156|
|[2nd Chebyshev-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/chebyshev2_kan.py) | 4396 | ~55K | ~39m| 0.33 | 28.53 | 0.924 |0.0500|
|[Fibonacci-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fibonacci_kan.py) | 4396 | ~65K | ~32m| 0.34 | 28.30 | 0.922 |0.0521|
|[Gegenbauer-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/gegenbauer_kan.py) | 4396 | ~53K | ~40m| 0.32 |  28.39| 0.922 |0.0514|
|[Hermite-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/hermite_kan.py) | 4396 | ~55K | ~38m| 0.37 | 27.58 | 0.913 |0.0591|
|[Legendre-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/legendre_kan.py) | 4396 | ~55K | ~38m| 0.33 | 26.64 | 0.893 |0.0986|
|[Lucas-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/lucas_kan.py) | 3532 | ~75K | ~28m | 0.42 | 27.95 | 0.916 |0.0550 |
|[Laguerre-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/laguerre_kan.py) | 3532 | ~74K | ~28m | 0.39 | 27.39 | 0.912 |0.0593 |
|[MexicanHat-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/wav_kan.py) | 3614 | ~66K | ~32m | 0.35 | 31.23 | 0.961 |0.0221 |
|[Morlet-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/wav_kan.py) | 3614 |  ~67K| ~31m  | 0.38 | 13.06 |0.686 |0.2583 |
|[DoG-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/wav_kan.py) |3614 | ~75K | ~28m | 0.41 | 32.59 | 0.966  | 0.0174 |
|[Meyer-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/wav_kan.py) | 3614 | ~36K | ~55m | 0.17 | 11.91 | 0.728 | 0.2991 |
|[Shannon-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/wav_kan.py) | 3614 | ~73K | ~28m | 0.49 | 9.15 | 0.738 |0.4434 |
|[Bump-KAN](https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/wav_kan.py) | × | × |  × | × | × | × | × |

- `360_v2: garden / 30k`, todo


# [Nerfstudio Installation](https://docs.nerf.studio/quickstart/installation.html)
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
# Train
############# kan_basis_type #############
# mlp, bspline, grbf, rbf, fourier,
# fcn, fcn_inter, chebyshev, jacobi
# bessel, chebyshev2, finonacci, hermite
# legendre, gegenbauer, lucas, laguerre
# mexican_hat, morlet, dog, meyer, shannon, bump
bash train_blender.sh [kan_basis_type]

# eval
bash run_eval.sh [exp_path]

# render RGB & Depth
bash run_render.sh [exp_path]
```

# Docs
- [Universal Approximation Theorem vs. Kolmogorov–Arnold Theorem](docs/Theorem.md)

# [PAPER](https://github.com/lif314/X-KANeRF)
COMMING SOON! We will provide a more detailed discussion of the impact of KAN on NeRF in our paper.

# Citation
If you use this benchmark in your research, please cite this project.
```bibtex
@misc{xkanerf,
	title={X-KANeRF: KAN-based NeRF with Various Basis Functions},
	author={Linfei Li},
	howpublished = {\url{https://github.com/lif314/X-KANeRF}},
	year={2024}
}
```

# Acknowledgement
- [KANeRF](https://github.com/Tavish9/KANeRF), A big thank you for this awesome work!
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