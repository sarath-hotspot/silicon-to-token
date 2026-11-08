# CUDA memory lab

Starter setup for memory experiments. The `memory` executable allocates device
memory, copies 1,024 FP32 elements from host to device and back, and checks that
every value is preserved. It is a correctness smoke test, not a bandwidth
benchmark.

## Build

Requires an NVIDIA GPU with a working driver, CUDA Toolkit (`nvcc`), CMake
3.24 or newer, and a supported host C++ compiler.

```bash
./build.sh
```

The script configures a Release build with C++17/CUDA17 and `-lineinfo`, then
builds into `build/`. It works from any working directory when invoked by its
path. Additional arguments are forwarded to CMake configuration.

Every `.cu` file directly in this directory builds as a separate executable
named after the file (for example, `memory.cu` produces `build/memory`).
Each file must provide its own `main()`. Adding or removing a `.cu` file is
detected on the next build; no CMake target edits are needed.

The default architecture is the locally visible GPU. To compile without a
visible GPU, specify an architecture, for example architecture 86 for an A40:

```bash
./build.sh -DCMAKE_CUDA_ARCHITECTURES=86
```

Running the program and test still requires GPU access.

## Run and check

Run from this directory:

```bash
./build/memory
# Expected: Memory round-trip check passed

ctest --test-dir build --output-on-failure
# Expected: 1 test passes
```

CUDA failures and incorrect results exit nonzero.

## Unified-memory computation timing

```bash
./build/unified-memory 16777216
```

The program checks GPU vector addition against a CPU reference, then prints
`GPU elapsed time (ms)=...` and `CPU elapsed time (ms)=...` on separate lines
at the end. CUDA events in the default stream measure the GPU kernel, and
`std::chrono::steady_clock` measures the CPU vector-add call. Allocation,
initialization, correctness checking, and cleanup are outside these intervals.

These are single-run timings without warmups or prefetching. GPU timing can
include unified-memory migration to the GPU; CPU timing can include migration
back to the host and first-touch costs for its output buffer. They are not
isolated steady-state arithmetic timings.

## Explicit-memory computation timing

```bash
./build/explicit-memory 16777216
```

This program prints the same GPU and CPU elapsed-time metrics after checking
that their results match. CUDA events measure only the GPU kernel, excluding
explicit host-to-device and device-to-host copies. A steady clock measures the
CPU vector-add call. Allocation, initialization, device-buffer zeroing,
correctness checking, and cleanup are outside the measured intervals.

These are single-run timings without warmups. CPU timing can include first-touch
costs for its output buffer. Unlike the unified-memory example, the GPU timing
does not include transferring the input arrays, so comparing GPU timings alone
is not an end-to-end comparison of the two memory approaches.
