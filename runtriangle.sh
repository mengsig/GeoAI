#!/usr/bin/env bash
set -euo pipefail



# define your arrays
patch_sizes=(64 32 16 8 4 2)
strides=(64 32 16 8 4 2)

# loop and run only when patch_size > stride
for ps in "${patch_sizes[@]}"; do
  for st in "${strides[@]}"; do
    if [ "$ps" -ge "$st" ]; then
      echo "=== Running experiment: patch_size=$ps, stride=$st ==="
      python src/cnn_sliding_window_inner.py "$ps" "$st"
    fi
  done
done


# define your arrays
patch_sizes=(64 32 16 8 4)
strides=(32 16 8 4 2)


# loop and run only when patch_size > stride
for ps in "${patch_sizes[@]}"; do
  for st in "${strides[@]}"; do
    if (( ps > st )); then
      echo "=== Running experiment: patch_size=$ps, stride=$st ==="
      python src/cnn_sliding_window_distribution.py "$ps" "$st"
    fi
  done
done
