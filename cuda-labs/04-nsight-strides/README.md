# Nsight: strided vector addition

Separate profiling project; the original vector-add lab is unchanged.

## My notes / learnings

- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) are profiling tools,
  similar in purpose to JProfiler for Java.
- `nsys` gives me the big picture: where the application spends time across
  host CPU CUDA API calls, CPU-to-GPU and GPU-to-CPU copies, GPU kernels, and
  synchronization. It is not just a CPU profiler.
- The Systems UI is a timeline of operations. I can see what runs when,
  what overlaps, and where there are gaps or waits.
- `ncu` helps me look inside a GPU kernel: how efficiently it executes,
  how it accesses memory, and why warps stall, such as waiting for memory
  dependencies. Its UI focuses on kernel metrics and analysis rather than
  the whole application's timeline.
- My mental model: use `nsys` to find where time goes, then `ncu` to understand
  why a particular kernel is slow. For `cudaMemcpy` transfer profiling, use
  `nsys`.

## Experiment

Each measured launch uses **one block of 256 threads**:

```cpp
size_t i = blockIdx.x * blockDim.x + threadIdx.x;
size_t source = i * stride;
if (i < n && source < n) {
    c[i] = a[source] + b[source];
}
```

The stride is restricted to 1 through 5. Both inputs contain `n` FP32 elements;
the output contains 256. The default `n` is **1,000,000,000** (decimal 1B),
requiring about 8 GB of device memory and another 8 GB of host RAM for the
inputs together. Ordinary pageable host buffers are allocated and filled on
the CPU, then each full input is copied to its device buffer with synchronous
`cudaMemcpy` before any vector-add launches. This models host-resident weights
being loaded onto the GPU once and then reused for computation; it does not
model disk loading or asynchronous transfer/compute overlap. A free-device-memory
check is not a reservation; allocation can still fail if another process uses
memory concurrently. Host allocation failures, CUDA failures, and invalid
arguments are reported and exit nonzero.

**A large allocation does not mean this kernel processes all 1B elements.**
Each thread produces at most one output. With the default input size all 256
threads are active, even at stride 5. This experiment examines warp-level
coalescing, not sustained full-array bandwidth. Use a separate grid-stride-loop
experiment to process all elements with this launch size.

## Build and test

Requires a CUDA-capable GPU, CUDA Toolkit, CMake 3.24+, a supported C++17 host
compiler, and `nsys` / `ncu` for profiling.

```bash
./build.sh
ctest --test-dir build --output-on-failure
./build/strided_vector_add 1000000000 1
```

Run these commands from this lab directory. `build.sh` works from any directory
and forwards arguments to CMake, for example `-DCMAKE_CUDA_ARCHITECTURES=86`.
Release builds include `-lineinfo`.

The executable accepts `[n] [stride]`, defaults to `1000000000 1`, and checks all
256 outputs against a CPU reference. Out-of-bounds threads must leave their
output sentinel unchanged. Tests cover every stride with all threads active,
every stride with bounds limiting active threads, single-element inputs, and
invalid arguments.

Host initialization, allocation, and host-to-device input transfers are excluded
from benchmark timing, but the CUDA transfers are visible in Systems traces.
Ten warmups precede 100 launches timed together with CUDA events. The average
includes possible gaps between launches; it is not an isolated single-kernel
duration.

## Profile strides 1 through 5

```bash
./profile.sh
# Optional smaller allocation and custom report directory:
./profile.sh 1280 /tmp/nsight-strides-small
```

The script runs an unprofiled benchmark, Nsight Systems, and Nsight Compute
for each stride. Reports use a unique directory under `results/profiles/`
(ignored by Git) unless an explicit output directory is provided. Existing
reports are not forcibly overwritten.

- **Systems:** CUDA timeline including two full-input H2D copies, a small output
  sentinel H2D copy, warmups, all 100 measured launches, and the final output D2H
  copy. CPU buffer filling occurs before the input copies and is not a CUDA
  kernel. Inspect `strided_vector_add` for the stride comparison. CPU sampling
  and context-switch tracing are disabled for restricted/container environments.
- **Compute:** only the first post-warmup `strided_vector_add` launch is selected
  using a kernel-name filter and ten skipped matching launches. It collects GPU
  duration, global-load requests, and global-load sectors. Metric collection may
  replay the kernel. Cache flushing is explicitly enabled, unlike the naturally
  warmed benchmark and Systems runs. Counter permissions are required. Compute
  failures are printed and logged; the sweep continues so Systems reports for
  other strides are retained, but the script ultimately exits nonzero.

Open `.nsys-rep` in Nsight Systems and `.ncu-rep` in Nsight Compute. The script
also saves benchmark text and CSV exports alongside those reports.

## What to compare

For aligned FP32 accesses, one warp reading consecutive values needs four
32-byte sectors per input load instruction. Strides 1 through 5 ideally require
4, 8, 12, 16, and 20 sectors per warp per input, respectively. All eight warps
and both inputs give an expected 16 global-load requests and 64, 128, 192, 256,
and 320 requested load sectors per kernel. Verify these predictions with the
actual counters; requested sectors are not necessarily unique DRAM traffic
because sectors can be reused or served by caches.

Divide global-load sectors by global-load requests to compare coalescing.
Stores remain contiguous. One block uses only one SM, so low whole-GPU
utilization is expected, and microsecond-scale timing differences should not
be treated as a full-GPU bandwidth comparison. Compare profiler counters and
repeat unprofiled timings before drawing performance conclusions.

## GPU counter permissions

`ERR_NVGPUCTRPERM` means the driver denied performance-counter access. Being root
inside a container is not sufficient when the driver has
`RmProfilingAdminOnly=1` and the container lacks `CAP_SYS_ADMIN`. A host
administrator must configure approved profiling access, or the Compute commands
must run in an environment where counters are permitted. This lab does not
change driver security settings. Systems CUDA tracing can still work without
these counters; failed Compute logs are not successful counter measurements.

## Historical measurements (device-initialized inputs)

These measurements predate the change to CPU-filled inputs and H2D transfers;
the old setup used a separate `initialize_inputs` GPU kernel. They do not
measure the current host-loading path.

Measured on 2026-11-09 with an NVIDIA A40, CUDA Toolkit 12.8, and Nsight
Systems 2026.5.2. All five runs used `n=1000000000` and passed correctness.
The Systems traces independently confirm 110 strided-kernel launches per run,
each with a 1x1x1 grid and 256x1x1 block.

| Stride | Unprofiled event average (us) | Systems kernel average, all 110 launches (us) |
|---|---:|---:|
| 1 | 2.18944 | 1.3277 |
| 2 | 2.41120 | 1.3676 |
| 3 | 2.44608 | 1.3996 |
| 4 | 2.19680 | 1.4423 |
| 5 | 2.15584 | 1.4801 |

These are different timing boundaries: the event batch includes launch gaps,
while Systems averages individual kernel execution durations including warmups.
This is one sweep, not evidence of a stable speedup or slowdown. All five
Compute attempts failed with `ERR_NVGPUCTRPERM`; no load-sector counts were
measured. Expected sector counts above remain predictions.
