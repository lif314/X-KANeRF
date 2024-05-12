import torch
import torch.nn as nn
from typing import List

# code modified from https://github.com/Boris-73-TA/OrthogPolyKANs

class Cheby2KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(Cheby2KANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby2_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby2_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Chebyshev polynomial tensors
        cheby2 = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby2[:, :, 1] = 2 * x
        for i in range(2, self.degree + 1):
            cheby2[:, :, i] = 2 * x * cheby2[:, :, i - 1].clone() - cheby2[:, :, i - 2].clone()
        # Compute the Chebyshev interpolation
        y = torch.einsum('bid,iod->bo', cheby2, self.cheby2_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

# To avoid gradient vanishing caused by tanh
class Cheby2KANLayerWithNorm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(Cheby2KANLayerWithNorm, self).__init__()
        self.layer = Cheby2KANLayer(input_dim=input_dim, output_dim=output_dim, degree=degree)
        self.layer_norm = nn.LayerNorm(output_dim) # To avoid gradient vanishing caused by tanh

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x
    
class Chebyshev2_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        grid_size: int = 8, # placeholder
        spline_order=0. # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Cheby2KANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x