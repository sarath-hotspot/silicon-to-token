#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
mkdir -p results

echo "n,threads,blocks,latency_ms,bandwidth_GBps" > results/block_sizes.csv

for threads in 64 128 256 512 1024; do
  ./build/vector_add_bench 16777216 "$threads" \
    | awk -F= '/^(n|threads|blocks|latency_ms|bandwidth_GBps)=/ {
        printf "%s%s", separator, $2; separator=","
      } END { print "" }' >> results/block_sizes.csv
done

cat results/block_sizes.csv