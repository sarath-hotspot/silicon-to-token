# Vector addition lab

## Build

Requires an NVIDIA GPU with a working driver, CUDA Toolkit (`nvcc`), CMake
3.24 or newer, and a host C++ compiler supported by the toolkit.

```bash
nvidia-smi
nvcc --version
./build.sh
```

`build.sh` configures a Release build and builds both CUDA programs. It works
from any working directory when invoked by its path and stops if configuration
or compilation fails. Additional arguments are forwarded to CMake configuration.
The build targets the locally visible GPU by default. For compilation without
a visible GPU, specify your target architecture, for example:

```bash
./build.sh -DCMAKE_CUDA_ARCHITECTURES=86
```

Architecture 86 targets an NVIDIA A40. Running the programs and tests still
requires GPU access. Run the commands below from this directory.

Executables are written to `build/`; existing binaries in this directory are
not replaced.

## Run and check

```bash
./build/vector_add
# Expected: c[11] = 33

# Arguments: element count, threads per block, kernel mode
./build/vector_add_bench 16777216 256 direct
./build/vector_add_bench 16777216 256 grid_stride

ctest --test-dir build --output-on-failure
# Expected: all 3 tests pass
```

The benchmark validates every output element against a CPU reference before
printing each result field on a separate `key=value` line. A failed correctness check or checked
CUDA operation exits nonzero. The tests exercise the example, an irregular-size
direct launch, and a grid-stride launch that processes multiple elements per
thread.

Benchmark output:

```text
Kernel=direct
gpu=...
cuda_runtime=...
cuda_driver=...
n=16777216
threads=256
blocks=65536
latency_ms=...
bandwidth_GBps=...
```

Latency is the average of 100 timed iterations after 10 warmups. Bandwidth
counts two float reads and one float write per element, not measured DRAM
traffic. Compare results on the same GPU and input size.

## Block-size sweep with the new build

Use Bash. This writes a separate CSV, leaving the existing `block_sizes.csv`
unchanged. Repeating the commands replaces `cmake_block_sizes.csv`.

```bash
set -euo pipefail
mkdir -p results
echo "n,threads,blocks,latency_ms,bandwidth_GBps" > results/cmake_block_sizes.csv
for threads in 64 128 256 512 1024; do
  ./build/vector_add_bench 16777216 "$threads" direct \
    | awk -F= '/^(n|threads|blocks|latency_ms|bandwidth_GBps)=/ {
        printf "%s%s", separator, $2; separator=","
      } END { print "" }' >> results/cmake_block_sizes.csv
done
cat results/cmake_block_sizes.csv
```

`./bench.sh` also uses the CMake executable in `build/` and works from any
working directory. It writes `results/block_sizes.csv`, replacing the recorded
results there. Use the commands above to keep that file unchanged.

## Learnings

1. When neighboring threads in a warp access elements spaced apart in memory
   (strided access), the warp generally touches more memory sectors than with
   sequential access in the default vector add. This requires more memory
   transactions for the same useful data and can reduce effective bandwidth.
   This is different from our `grid_stride` kernel: each thread jumps by the
   total number of threads between iterations, but neighboring threads still
   access consecutive elements within each iteration, preserving coalescing.

### Measured direct versus grid-stride results

Measured on 2026-11-08 with an NVIDIA A40, CUDA runtime 12.8, and CUDA
driver API version 13.2. Both kernels used 16,777,216 FP32 elements and
256 threads per block. Each run checked correctness and reported the average
kernel time over 100 iterations after 10 warmups; host-device copies were
excluded. Three paired runs used:

```bash
for repeat in 1 2 3; do
  ./build/vector_add_bench 16777216 256 direct
  ./build/vector_add_bench 16777216 256 grid_stride
done
```

| Run | Kernel | Blocks | Average latency (ms) | Effective bandwidth (GB/s) |
|---|---|---:|---:|---:|
| 1 | direct | 65,536 | 0.346386 | 581.220 |
| 1 | grid_stride | 336 | 0.379691 | 530.238 |
| 2 | direct | 65,536 | 0.346421 | 581.161 |
| 2 | grid_stride | 336 | 0.379950 | 529.877 |
| 3 | direct | 65,536 | 0.346902 | 580.356 |
| 3 | grid_stride | 336 | 0.379755 | 530.149 |

Using the median of the three reported averages, grid-stride took
0.379755 ms versus 0.346421 ms for direct: about **9.6% longer**.
Effective bandwidth fell from 581.161 to 530.149 GB/s, about **8.8% lower**.
Both perform the same useful memory work: `3 * n * sizeof(float)`, or
201,326,592 bytes per iteration. More time for the same work means lower
effective bandwidth; this does not measure physical DRAM traffic directly.

**Why might grid-stride be slower here?** The grid-stride launch is capped at
four blocks per SM (336 blocks on this GPU), giving 86,016 threads. Each thread
processes up to 196 elements, with loop increments, comparisons, and repeated
address calculations. The direct kernel assigns one element to each thread
and has no such loop. The smaller grid also changes scheduling and the
opportunities to hide memory latency. These are plausible contributors, not
proven bottlenecks; profiler counters and a grid-size sweep would be needed to
separate them.

**This reduction is not evidence of non-coalesced strided access.** In both
kernels, neighboring threads access consecutive FP32 elements within each
iteration. A different kernel that spaces neighboring threads' accesses apart
would generally touch more memory sectors per warp instruction, transferring
more data than is useful and potentially reducing effective bandwidth.
Grid-stride itself is not inherently slower; these results apply to this
input size, GPU, and launch configuration.
