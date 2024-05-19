import torch
import torch.nn as nn
from typing import List

# code modified from https://github.com/Boris-73-TA/OrthogPolyKANs

class LucasKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LucasKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # Initialize coefficients for the Lucas polynomials
        self.lucas_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.lucas_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize input x to [-1, 1] for stability in polynomial calculation

        # Initialize Lucas polynomial tensors
        lucas = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        lucas[:, :, 0] = 2  # L_0(x) = 2
        if self.degree > 0:
            lucas[:, :, 1] = x  # L_1(x) = x

        for i in range(2, self.degree + 1):
            # Compute Lucas polynomials using the recurrence relation
            lucas[:, :, i] = x * lucas[:, :, i - 1].clone() + lucas[:, :, i - 2].clone()

        # Normalize the polynomial outputs to prevent runaway values
        #lucas = torch.tanh(lucas)

        # Compute the Lucas interpolation
        y = torch.einsum('bid,iod->bo', lucas, self.lucas_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y
        
# Add orthogonalization ?


# To avoid gradient vanishing caused by tanh
class LucasKANLayerWithNorm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LucasKANLayerWithNorm, self).__init__()
        self.layer = LucasKANLayer(input_dim=input_dim, output_dim=output_dim, degree=degree)
        self.layer_norm = nn.LayerNorm(output_dim) # To avoid gradient vanishing caused by tanh

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x
    
class Lucas_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 3,
        grid_size: int = 8, # placeholder
        spline_order=0. # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            LucasKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x