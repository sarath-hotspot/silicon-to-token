#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
binary="$script_dir/build/strided_vector_add"
n="${1:-1000000000}"
output="${2:-$script_dir/results/profiles/$(date -u +%Y%m%dT%H%M%S)-$$}"

for tool in nsys ncu; do
  if ! command -v "$tool" >/dev/null; then
    echo "Missing required profiler: $tool" >&2
    exit 1
  fi
done
if [[ ! -x "$binary" ]]; then
  echo "Build the lab first with $script_dir/build.sh" >&2
  exit 1
fi
mkdir -p "$output"
failed=0

for stride in 1 2 3 4 5; do
  prefix="$output/stride-$stride"
  "$binary" "$n" "$stride" | tee "$prefix-benchmark.txt"
  nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    -o "$prefix-nsys" "$binary" "$n" "$stride"
  nsys stats --report cuda_gpu_kern_sum --format csv --output "$prefix-nsys" \
    "$prefix-nsys.nsys-rep"
  if ! ncu --kernel-name regex:strided_vector_add --launch-skip 10 --launch-count 1 \
    --cache-control all \
    --metrics gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    --export "$prefix-ncu" --page raw --csv --log-file "$prefix-ncu.csv" \
    "$binary" "$n" "$stride"; then
    echo "Nsight Compute failed for stride $stride; see $prefix-ncu.csv" >&2
    cat "$prefix-ncu.csv" >&2
    failed=1
  fi
done

printf 'Profiles saved to %s\n' "$output"
if (( failed )); then
  echo "Profiling incomplete: one or more Nsight Compute runs failed." >&2
  exit 1
fi
