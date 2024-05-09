# MNIST example using Fourier Kolmogorov-Arnold Networks

Super quick and dirty experiment using https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py with slight modifications: I adjusted the fftKAN.py code by adding gridsize as a learnable parameter and employing Xavier initialization for Fourier coefficients. This allows the model to dynamically adapt the grid size during training, potentially enhancing its performance.

# Current performance after 1 epoch (92%) 

```
Train Epoch: 1 [0/6000 (0%)]	Loss: 1.949768
Train Epoch: 1 [640/6000 (11%)]	Loss: 0.627734
Train Epoch: 1 [1280/6000 (21%)]	Loss: 0.441107
Train Epoch: 1 [1920/6000 (32%)]	Loss: 0.331095
Train Epoch: 1 [2560/6000 (43%)]	Loss: 0.231573
Train Epoch: 1 [3200/6000 (53%)]	Loss: 0.172317
Train Epoch: 1 [3840/6000 (64%)]	Loss: 0.305194
Train Epoch: 1 [4480/6000 (74%)]	Loss: 0.314672
Train Epoch: 1 [5120/6000 (85%)]	Loss: 0.366347
Train Epoch: 1 [5760/6000 (96%)]	Loss: 0.314822
Test set: Average loss: 0.0013, Accuracy: 9171/10000 (92%)
```
