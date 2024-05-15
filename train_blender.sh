#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1

############# kan_basis_type #############
# mlp, bspline, grbf, rbf, fourier,
# fcn, fcn_inter, chebyshev, jacobi
# bessel, chebyshev2, finonacci, hermite
# legendre, gegenbauer
kan_basis_type='mlp'  # `mlp` refers to nerfacto with torch MLP, no tcnn
ns-train xkanerf --data data/nerf_synthetic/lego \
    --experiment-name "$kan_basis_type-blender-lego" \
    --pipeline.model.kan_basis_type $kan_basis_type \
    --pipeline.model.background-color white \
    --pipeline.model.proposal-initial-sampler uniform \
    --pipeline.model.near-plane 2. \
    --pipeline.model.far-plane 6. \
    --pipeline.model.hidden_dim 8 \
    --pipeline.model.hidden_dim_color 8 \
    --pipeline.model.num_layers 1 \
    --pipeline.model.num_layers_color 1 \
    --pipeline.model.geo_feat_dim 7 \
    --pipeline.model.appearance_embed_dim 8 \
    --pipeline.datamanager.camera-optimizer.mode off \
    --pipeline.model.use-average-appearance-embedding False \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --pipeline.datamanager.eval-num-rays-per-batch 4096 \
    --pipeline.model.distortion-loss-mult 0 \
    --pipeline.model.disable-scene-contraction True \
    --vis viewer+tensorboard \
    blender-data
    