def get_kan_model(kan_basis_type='bspline'):
    if kan_basis_type == 'mlp':
        from xKANeRF.xKAN.nerfacto_mlp import Nefacto_MLP
        return Nefacto_MLP
    if kan_basis_type == 'bspline':
        from xKANeRF.xKAN.bspine_kan import BSpline_KAN
        return BSpline_KAN
    elif kan_basis_type == 'grbf':
        from xKANeRF.xKAN.grbf_kan import GRBF_KAN
        return GRBF_KAN
    elif kan_basis_type == 'rbf':
        from xKANeRF.xKAN.rbf_kan import RBF_KAN
        return RBF_KAN
    elif kan_basis_type == 'fourier':
        from xKANeRF.xKAN.fourier_kan import Fourier_KAN
        return Fourier_KAN
    elif kan_basis_type == 'fcn':
        from xKANeRF.xKAN.fcn_kan import FCN_KAN
        return FCN_KAN
    elif kan_basis_type == 'fcn_inter':
        from xKANeRF.xKAN.fcn_kan import FCN_InterpoKAN
        return FCN_InterpoKAN
    elif kan_basis_type == 'chebyshev':
        from xKANeRF.xKAN.chebyshev_kan import Chebyshev_KAN
        return Chebyshev_KAN
    elif kan_basis_type == 'jacobi':
        from xKANeRF.xKAN.jacobi_kan import Jacobi_KAN
        return Jacobi_KAN
    elif kan_basis_type == 'bessel':
        from xKANeRF.xKAN.bessel_kan import Bessel_KAN
        return Bessel_KAN
    elif kan_basis_type == 'chebyshev2':
        from xKANeRF.xKAN.chebyshev2_kan import Chebyshev2_KAN
        return Chebyshev2_KAN
    elif kan_basis_type == 'finonacci':
        from xKANeRF.xKAN.fibonacci_kan import Fibonacci_KAN
        return Fibonacci_KAN
    elif kan_basis_type == 'hermite':
        from xKANeRF.xKAN.hermite_kan import Hermite_KAN
        return Hermite_KAN
    elif kan_basis_type == 'legendre':
        from xKANeRF.xKAN.legendre_kan import Legendre_kan
        return Legendre_kan
    else:
        print("Not Implemented!!!")