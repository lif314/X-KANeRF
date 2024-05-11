#!/bin/bash

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start Time: $start_time"

start_timestamp=$(date -d "$start_time" +%s)


# export CUDA_VISIBLE_DEVICES=1
# ns-train xkanerf --data data/360_v2_8x/garden \
#     --experiment-name "bspline-360-garden" \
#     --pipeline.model.proposal-initial-sampler uniform \
#     --pipeline.model.hidden_dim 8 \
#     --pipeline.model.hidden_dim_color 8 \
#     --pipeline.model.num_layers 1 \
#     --pipeline.model.num_layers_color 1 \
#     --pipeline.model.geo_feat_dim 7 \
#     --pipeline.model.appearance_embed_dim 8 \
#     --pipeline.datamanager.train-num-rays-per-batch 4096 \
#     --pipeline.datamanager.eval-num-rays-per-batch 4096 \
#     --vis viewer+tensorboard \
#     colmap


ns-train nerfacto --data data/360_v2_8x/garden \
    --experiment-name "nerfacto-360-garden" \
    --pipeline.model.proposal-initial-sampler uniform \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --pipeline.datamanager.eval-num-rays-per-batch 4096 \
    --vis viewer+tensorboard \
    colmap


end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End Time: $end_time"
end_timestamp=$(date -d "$end_time" +%s)
total_seconds=$((end_timestamp - start_timestamp))
formatted_time=$(printf "%02d:%02d:%02d" $((total_seconds/3600)) $((total_seconds%3600/60)) $((total_seconds%60)))
echo "Total Time: $formatted_time"