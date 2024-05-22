import torch
import torch.nn as nn
from typing import List

# code modified from https://github.com/SynodicMonth/ChebyKAN

# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class slow_ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        # Compute the Chebyshev interpolation
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
    
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.epsilon = 1e-6

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)

        # Apply acos with numerical stability
        x = torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

# To avoid gradient vanishing caused by tanh
class ChebyKANLayerWithNorm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayerWithNorm, self).__init__()
        self.layer = ChebyKANLayer(input_dim=input_dim, output_dim=output_dim, degree=degree)
        self.layer_norm = nn.LayerNorm(output_dim) # To avoid gradient vanishing caused by tanh

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x
    
class Chebyshev_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        grid_size: int = 8, # placeholder
        spline_order=0. # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ChebyKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x