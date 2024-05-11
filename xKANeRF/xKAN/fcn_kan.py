import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

# code modified from https://github.com/Zhangyanbo/FCN-KAN

def heaviside_theta(x, mu, r):
    """Heaviside theta function with parameters mu and r.

    Args:
        x (torch.Tensor): Input tensor.
        mu (float): Center of the function.
        r (float): Width of the function.
    
    Returns:
        torch.Tensor: Output tensor.
    """
    x = x - mu
    return (torch.clamp(x + r, 0, r) - torch.clamp(x, 0, r)) / r

def _linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Note: This function is used to apply the linear interpolation to one element of the input tensor.
    For vectorized operations, use the linear_interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    mu = X
    r = X[1] - X[0]
    F = torch.vmap(heaviside_theta, in_dims=(None, 0, None))
    y = F(x, mu, r).reshape(-1) * Y
    return y.sum()

def linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    shape = x.shape
    x = x.reshape(-1)
    return torch.vmap(_linear_interpolation, in_dims=(-1, None, None), out_dims=-1)(x, X, Y).reshape(shape)

def phi(x, w1, w2, b1, b2, n_sin):
    """
    phi function that integrates sinusoidal embeddings with MLP layers.

    Args:
        x (torch.Tensor): Input tensor.
        w1 (torch.Tensor): Weight matrix for the first linear transformation.
        w2 (torch.Tensor): Weight matrix for the second linear transformation.
        b1 (torch.Tensor): Bias vector for the first linear transformation.
        b2 (torch.Tensor): Bias vector for the second linear transformation.
        n_sin (int): Number of sinusoidal functions to generate.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    omega = (2 ** torch.arange(0, n_sin, device=x.device)).float().reshape(-1, 1)
    omega_x = F.linear(x, omega, bias=None)
    x = torch.cat([x, torch.sin(omega_x), torch.cos(omega_x)], dim=-1)
    
    x = F.linear(x, w1, bias=b1)
    x = F.silu(x)
    x = F.linear(x, w2, bias=b2)
    return x

class FCNKANLayer(nn.Module):
    """
    A layer in a Kolmogorov–Arnold Networks (KAN).

    Attributes:
        dim_in (int): Dimensionality of the input.
        dim_out (int): Dimensionality of the output.
        fcn_hidden (int): Number of hidden units in the feature transformation.
        fcn_n_sin (torch.tensor): Number of sinusoidal functions to be used in phi.
    """
    def __init__(self, dim_in, dim_out, fcn_hidden=32, fcn_n_sin=3):
        """
        Initializes the KANLayer with specified dimensions and sinusoidal function count.
        
        Args:
            dim_in (int): Dimension of the input.
            dim_out (int): Dimension of the output.
            fcn_hidden (int): Number of hidden neurons in the for the learned non-linear transformation.
            fcn_n_sin (int): Number of sinusoidal embedding frequencies.
        """
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden, 1+fcn_n_sin*2))
        self.W2 = nn.Parameter(torch.randn(dim_in, dim_out, 1, fcn_hidden))
        self.B1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden))
        self.B2 = nn.Parameter(torch.randn(dim_in, dim_out, 1))

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = torch.tensor(fcn_n_sin).long()

        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        # apply zero bias
        nn.init.zeros_(self.B1)
        nn.init.zeros_(self.B2)
    
    def map(self, x):
        """
        Maps input tensor x through phi function in a vectorized manner.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after mapping through phi.
        """
        F = torch.vmap(
            # take dim_in out, -> dim_in x (dim_out, *)(1)
            torch.vmap(phi, (None, 0, 0, 0, 0, None), 0), # take dim_out out, -> dim_out x (*)
            (0, 0, 0, 0, 0, None), 0
            )
        return F(x.unsqueeze(-1), self.W1, self.W2, self.B1, self.B2, self.fcn_n_sin).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Summed output after mapping each dimensions through phi.
        """
        device = x.device
        self.W1 = self.W1.to(device)
        self.W2 = self.W2.to(device)
        self.B1 = self.B1.to(device)
        self.B2 = self.B2.to(device)
        self.fcn_n_sin = self.fcn_n_sin.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x) # [batch, dim_in, dim_out]

        return phis.sum(dim=1)
    
    def take_function(self, i, j):
        """
        Returns a phi function specific to the (i, j)-th elements of parameters.

        Args:
            i (int): Row index in parameter tensors.
            j (int): Column index in parameter tensors.

        Returns:
            function: A function that computes phi for specific parameters.
        """
        def activation(x):
            return phi(x, self.W1[i, j], self.W2[i, j], self.B1[i, j], self.B2[i, j], self.fcn_n_sin)
        return activation


class FCNKANInterpoLayer(nn.Module):
    """
    A layer in a Kolmogorov–Arnold Networks (KAN).

    Attributes:
        dim_in (int): Dimensionality of the input.
        dim_out (int): Dimensionality of the output.
        num_x (int): Number of x values to interpolate.
        x_min (float): Minimum x value.
    """
    def __init__(self, dim_in, dim_out, num_x=64, x_min=-2, x_max=2):
        """
        Initializes the KANLayer with specified dimensions and sinusoidal function count.
        
        Args:
            dim_in (int): Dimension of the input.
            dim_out (int): Dimension of the output.
            num_x (int): Number of x values to interpolate.
            x_min (float): Minimum x value.
        """
        super().__init__()
        # self.X = nn.Parameter(torch.randn(dim_in, dim_out, num_x)
        self.X = torch.linspace(x_min, x_max, num_x)
        self.Y = nn.Parameter(torch.randn(dim_in, dim_out, num_x))

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_uniform_(self.Y)
    
    def map(self, x):
        """
        Maps input tensor x through phi function in a vectorized manner.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after mapping through phi.
        """
        F = torch.vmap(
            # take dim_in out, -> dim_in x (dim_out, *)(1)
            torch.vmap(linear_interpolation, (None, None, 0), 0), # take dim_out out, -> dim_out x (*)
            (0, None, 0), 0
            )
        return F(x.unsqueeze(-1), self.X, self.Y).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Summed output after mapping each dimensions through phi.
        """
        device = x.device
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x) # [batch, dim_in, dim_out]

        return phis.sum(dim=1)
    
    def take_function(self, i, j):
        """
        Returns a phi function specific to the (i, j)-th elements of parameters.

        Args:
            i (int): Row index in parameter tensors.
            j (int): Column index in parameter tensors.

        Returns:
            function: A function that computes phi for specific parameters.
        """
        def activation(x):
            return linear_interpolation(x, self.X, self.Y[i, j])
        return activation


def smooth_penalty(model):
    p = 0
    if isinstance(model, FCNKANInterpoLayer):
        dx = model.X[1] - model.X[0]
        grad = model.Y[:, :, 1:] - model.Y[:, :, :-1]
        # grad = grad[:, :, 1:] - grad[:, :, :-1]
        return torch.norm(grad, 2) / dx

    for layer in model:
        if isinstance(layer, FCNKANInterpoLayer):
            dx = layer.X[1] - layer.X[0]
            grad = layer.Y[:, :, 1:] - layer.Y[:, :, :-1]
            # grad = grad[:, :, 1:] - grad[:, :, :-1]
            p += torch.norm(grad, 2) / dx
    return p


class FCN_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8, # placeholder
        spline_order: int = 0, #  placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FCNKANLayer(
                dim_in=in_dim,
                dim_out=out_dim,
                fcn_hidden=1, # default ?
                fcn_n_sin=1 # default ?
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FCN_InterpoKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8, # placeholder
        spline_order: int = 0, #  placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FCNKANInterpoLayer(
                dim_in=in_dim,
                dim_out=out_dim,
                num_x=8, # default 
                x_min=-2, # default
                x_max=2   # default
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x