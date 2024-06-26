#!/bin/bash

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start Time: $start_time"

start_timestamp=$(date -d "$start_time" +%s)

# export CUDA_VISIBLE_DEVICES=1

############# kan_basis_type #############
# mlp, bspline, grbf, rbf, fourier,
# fcn, fcn_inter, chebyshev, jacobi
# bessel, chebyshev2, finonacci, hermite
# legendre, gegenbauer, lucas, laguerre
# mexican_hat, morlet, dog, meyer, shannon, bump
kan_basis_type=$1  # `mlp` refers to nerfacto with torch MLP, no tcnn
# kan_basis_type='mlp'
ns-train xkanerf --data data/nerf_synthetic/lego \
    --experiment-name "$kan_basis_type-blender-lego" \
    --max-num-iterations 30000 \
    --steps-per-eval-all-images 25000 \
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
    --viewer.quit-on-train-completion True \
    --vis viewer+tensorboard \
    blender-data


end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End Time: $end_time"
end_timestamp=$(date -d "$end_time" +%s)
total_seconds=$((end_timestamp - start_timestamp))
formatted_time=$(printf "%02d:%02d:%02d" $((total_seconds/3600)) $((total_seconds%3600/60)) $((total_seconds%60)))
echo "Total Time: $formatted_time"
    