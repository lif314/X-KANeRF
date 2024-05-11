#!/bin/bash

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start Time: $start_time"

start_timestamp=$(date -d "$start_time" +%s)

# export CUDA_VISIBLE_DEVICES=1
 ns-eval --load-config "config-path" \
         --output-path "output-path/eval.json"\

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End Time: $end_time"
end_timestamp=$(date -d "$end_time" +%s)
total_seconds=$((end_timestamp - start_timestamp))
formatted_time=$(printf "%02d:%02d:%02d" $((total_seconds/3600)) $((total_seconds%3600/60)) $((total_seconds%60)))
echo "Total Time: $formatted_time"