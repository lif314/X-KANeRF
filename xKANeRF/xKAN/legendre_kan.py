import torch
import torch.nn as nn
from typing import List

# code modified from https://github.com/Boris-73-TA/OrthogPolyKANs

class LegendreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LegendreKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.legendre_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.legendre_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        x = torch.tanh(x)  # Normalize input to [-1, 1] for stability in Legendre polynomial calculation

        # Initialize Legendre polynomial tensors
        legendre = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        legendre[:, :, 0] = 1  # P_0(x) = 1
        if self.degree > 0:
            legendre[:, :, 1] = x  # P_1(x) = x

        # Compute Legendre polynomials using the recurrence relation
        for n in range(2, self.degree + 1):
           # Recurrence relation without in-place operations
            legendre[:, :, n] = ((2 * (n-1) + 1) / (n)) * x * legendre[:, :, n-1].clone() - ((n-1) / (n)) * legendre[:, :, n-2].clone()

        # Compute output using matrix multiplication
        y = torch.einsum('bid,iod->bo', legendre, self.legendre_coeffs)
        y = y.view(-1, self.outdim)
        return y

# To avoid gradient vanishing caused by tanh
class LegendreKANLayerWithNorm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LegendreKANLayerWithNorm, self).__init__()
        self.layer = LegendreKANLayer(input_dim=input_dim, output_dim=output_dim, degree=degree)
        self.layer_norm = nn.LayerNorm(output_dim) # To avoid gradient vanishing caused by tanh

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x
    
class Legendre_kan(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        grid_size: int = 8, # placeholder
        spline_order=0. # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            LegendreKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x