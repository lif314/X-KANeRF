# Hands-On NeRF with KAN

[KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) is a promising challenger to traditional MLPs. We're thrilled about integrating KAN into [NeRF](https://www.matthewtancik.com/nerf)! Is KAN suited for **view synthesis** tasks? What challenges will we face? How will we tackle them? We provide our initial observations and future discussion!

https://github.com/Tavish9/KANeRF/assets/60593268/0d1ae7c6-e937-4552-b414-ba39f7624a3c

# Update News
* [x] Release the implementation of KANeRF based on [PyKAN](https://kindxiaoming.github.io/pykan/).
* [x] 🔥🔥🔥 Release an efficient implementation of KANeRF based on [Efficient-KAN](https://github.com/Blealtan/efficient-kan), achiving 15x speedup! 🎉
* [ ] 🎉🎉🎉 We will conduct more evaluation, stay tuned! : D
# Installation

KANeRF is buid based on [nerfstudio](https://docs.nerf.studio/quickstart/installation.html#) and [Efficient-KAN](https://github.com/Blealtan/efficient-kan).  Please refer to the website for detailed installation instructions if you encounter any problems.

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

# install efficient-kan
pip install git+https://github.com/Blealtan/efficient-kan.git
```

# Performance Comparision

We integrate KAN and [NeRFacto](https://docs.nerf.studio/nerfology/methods/nerfacto.html) and compare KANeRF with NeRFacto in terms of model parameters, training time, novel view synthesis performance, etc. on the [Blender dataset](https://github.com/bmild/nerf?tab=readme-ov-file#project-page--video--paper--data). Under the same network settings, KAN **slightly** outperforms MLP in novel view synthesis, suggesting that KAN possesses a more powerful fitting capability. However, KAN's inference and training speed are significantly** slower than those of MLP. Furthermore, with a comparable number of parameters, KAN underperforms MLP.

| Model                         | NeRFacto                                                                          | NeRFacto Tiny                                                                          | KANeRF                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Trainable Network Parameters  | 8192                                                                              | 2176                                                                                   | 7131                                                                            |
| Total Network Parameters      | 8192                                                                              | 2176                                                                                   | 10683                                                                           |
| hidden_dim                    | 64                                                                                | 8                                                                                      | 8                                                                               |
| hidden dim color              | 64                                                                                | 8                                                                                      | 8                                                                               |
| num layers                    | 2                                                                                 | 1                                                                                      | 1                                                                               |
| num layers color              | 2                                                                                 | 1                                                                                      | 1                                                                               |
| geo feat dim                  | 15                                                                                | 7                                                                                      | 7                                                                               |
| appearance embed dim          | 32                                                                                | 8                                                                                      | 8                                                                               |
| Training Time                 | 14m 13s                                                                           | 13m 47s                                                                                | 37m 20s                                                                         |
| FPS                           | 2.5                                                                               | ~2.5                                                                                   | 0.95                                                                            |
| LPIPS                         | 0.0132                                                                            | 0.0186                                                                                 | 0.0154                                                                          |
| PSNR                          | 33.69                                                                             | 32.67                                                                                  | 33.10                                                                           |
| SSIM                          | 0.973                                                                             | 0.962                                                                                  | 0.966                                                                           |
| Loss                          | ![1](asset/loss_nerfacto.png)                                                     | ![1](asset/loss_tiny_nerfactory.png)                                                   | ![1](asset/loss_kanerf.png)                                                      |
| result (rgb)   | <video src="https://github.com/Tavish9/KANeRF/assets/60593268/15eb4f45-256b-4ba8-ba5b-8f8c7c4d1b4f" width="512" height="512" controls>.</video>   | <video src="https://github.com/Tavish9/KANeRF/assets/60593268/951a1e02-cfa3-4605-ab58-4dad37d33104" width="512" height="512" controls>.</video>   | <video src="https://github.com/Tavish9/KANeRF/assets/60593268/c11dba85-0ff3-49c6-ae02-b22fa604e00b" width="512" height="512" controls>.</video>   |
| result (depth) | <video src="https://github.com/Tavish9/KANeRF/assets/60593268/53ac72e8-47fb-45c2-ac05-9fbc760326c6" width="512" height="512" controls>.</video> | <video src="https://github.com/Tavish9/KANeRF/assets/60593268/1a825a28-9faa-402d-b79a-5da7b0208318" width="512" height="512" controls>.</video> | <video src="https://github.com/Tavish9/KANeRF/assets/60593268/92e575d3-4927-4c67-9f4d-9ca39a35aa8f" width="512" height="512" controls>.</video> |

KAN has the potential for optimization, particularly with regard to accelerating its inference speed. We plan to develop a CUDA-accelerated version of KAN to further enhance its performance : D

* The Visualization of KanNeRF

<div style="text-align:center; ">
  <img src="asset/node.png" alt="Alt text" style="width:50%; height:auto; ">
</div>

## Contact us

```bibtex
@Manual{,
   title = {Hands-On NeRF with KAN},
   author = {Delin Qu, Qizhi Chen},
   year = {2024},
   url = {https://github.com/Tavish9/KANeRF},
 }
```
