from typing import Literal, Optional, Tuple
import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion

# from efficient_kan import KAN
from KANeRF.kan_models.bspine_kan import KAN

class KANeRFactoField(NerfactoField):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        grid_size: int = 3,
        spline_order: int = 3,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            num_layers,
            hidden_dim,
            geo_feat_dim,
            num_levels,
            base_res,
            max_res,
            log2_hashmap_size,
            num_layers_color,
            num_layers_transient,
            features_per_level,
            hidden_dim_color,
            hidden_dim_transient,
            appearance_embedding_dim,
            transient_embedding_dim,
            use_transient_embedding,
            use_semantics,
            num_semantic_classes,
            pass_semantic_gradients,
            use_pred_normals,
            use_average_appearance_embedding,
            spatial_distortion,
            implementation,
        )
        self.implementation = implementation

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.mlp_base_mlp = KAN(
            layers_hidden=[self.mlp_base_grid.get_out_dim()]
            + [hidden_dim] * num_layers
            + [1 + self.geo_feat_dim],
            grid_size=grid_size,
            spline_order=spline_order,
        )

        if self.use_transient_embedding:
            self.mlp_transient = KAN(
                layers_hidden=[self.geo_feat_dim + self.transient_embedding_dim]
                + [hidden_dim_transient] * num_layers_transient
                +[hidden_dim_transient],
                grid_size=grid_size,
                spline_order=spline_order,
            )

        if self.use_semantics:
            self.mlp_semantics = KAN(
                layers_hidden=[self.geo_feat_dim]
                + [64, 64]
                + [hidden_dim_transient],
                grid_size=grid_size,
                spline_order=spline_order,
            )

        if self.use_pred_normals:
            self.mlp_pred_normals = KAN(
                layers_hidden=[self.geo_feat_dim + self.position_encoding.get_out_dim()]
                + [64] * 3
                + [hidden_dim_transient],
                grid_size=grid_size,
                spline_order=spline_order,
            )

        self.mlp_head_base = KAN(
            layers_hidden=[self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim]
            + [hidden_dim_color] * num_layers_color
            + [3],
            grid_size=grid_size,
            spline_order=spline_order,
        )

        self.mlp_head = nn.Sequential(self.mlp_head_base, nn.Sigmoid())

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        grid_encoding = self.mlp_base_grid(positions_flat)
        grid_encoding = grid_encoding.float() if self.implementation == "tcnn" else grid_encoding
        h = self.mlp_base_mlp(grid_encoding).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out
