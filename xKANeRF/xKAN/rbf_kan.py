import torch
import torch.nn as nn
from typing import List

# code modified from https://github.com/sidhu2690/RBF-KAN
class RBFLinear(nn.Module):
   def __init__(self, in_features, out_features, grid_min=-2., grid_max=2., num_grids=8, spline_weight_init_scale=0.1):
       super().__init__()
       self.grid_min = grid_min
       self.grid_max = grid_max
       self.num_grids = num_grids
       self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=False)
       self.spline_weight = nn.Parameter(torch.randn(in_features*num_grids, out_features)*spline_weight_init_scale)

   def forward(self, x):
       x = x.unsqueeze(-1)
       basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
       return basis.view(basis.size(0), -1).matmul(self.spline_weight)

class RBFKANLayer(nn.Module):
   def __init__(self, 
                input_dim: int, 
                output_dim: int,
                grid_min: float = -2.,
                grid_max: float = 2.,
                num_grids: int = 8,
                base_activation = nn.SiLU,
                spline_weight_init_scale: float = 0.1,
                use_base_update=True):
       super().__init__()
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.use_base_update = use_base_update
       self.base_activation = base_activation()
       self.spline_weight_init_scale = spline_weight_init_scale
       self.rbf_linear = RBFLinear(input_dim, output_dim, grid_min, grid_max, num_grids, spline_weight_init_scale)
       self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None

   def forward(self, x):
       ret = self.rbf_linear(x)
       if self.use_base_update:
           base = self.base_linear(self.base_activation(x))
           ret = ret + base
       return ret

class RBF_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        grid_size: int = 8,
        base_activation = nn.SiLU,
        use_base_update=True,
        spline_weight_init_scale: float = 0.1,
        spline_order=0.
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RBFKANLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=grid_size,
                use_base_update=use_base_update, 
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
