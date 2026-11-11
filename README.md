# Silicon to Token

Hands-on CUDA labs for understanding GPU computation, memory, and transfer
performance.

## The stack: application to silicon

```text
┌──────────────────────────────────────────┐
│ Applications                             │  Chat UI, API clients, agents
├──────────────────────────────────────────┤
│ Serving and inference engines            │  vLLM, SGLang: APIs, scheduling,
│                                          │  batching, KV-cache management
├──────────────────────────────────────────┤
│ ML frameworks and GPU libraries          │  PyTorch, Transformers, cuBLAS
├──────────────────────────────────────────┤
│ GPU programming model and runtime        │  CUDA, memory and kernel launches
├──────────────────────────────────────────┤
│ GPU kernels                              │  Attention, matrix multiply, etc.
├──────────────────────────────────────────┤
│ Operating system and device drivers      │  Device access and command submission
├──────────────────────────────────────────┤
│ GPU hardware                             │  Compute units, Tensor Cores, HBM
└──────────────────────────────────────────┘
                     │
                     └── generated tokens return to the application
```

This is a conceptual view: serving engines can use several frameworks and
kernel backends, and some responsibilities span layers. These labs focus on
the CUDA, kernel, memory-transfer, and profiling layers below.

## Labs

- [01-vector-add](cuda-labs/01-vector-add/README.md): vector addition, grid-stride
  loops, kernel timing, and effective bandwidth.
- [02-memory](cuda-labs/02-memory/README.md): explicit and unified memory with
  GPU and CPU computation timing.
- [03-memory-transfers](cuda-labs/03-memory-transfers/README.md): pageable versus
  pinned transfers, cold-buffer setup costs. Experiments on at what buffer size it is worth using pinned vs pageable. 
- [04-nsight-strides](cuda-labs/04-nsight-strides/README.md): Nsight Systems and
  Compute profiling of strides 1 through 5 with one block of 256 threads.

## Build and test

Requires a CUDA-capable NVIDIA GPU, a working driver, CUDA Toolkit, CMake
3.24 or newer, and a supported host C++ compiler.

```bash
for lab in 01-vector-add 02-memory 03-memory-transfers 04-nsight-strides; do
  "./cuda-labs/$lab/build.sh"
  ctest --test-dir "./cuda-labs/$lab/build" --output-on-failure
done
```

Each lab builds into its own `build/` directory. See the lab READMEs for run
commands, timing boundaries, and plotting dependencies. Generated binaries,
build directories, and Python virtual environments are excluded from Git;
compact measured CSVs and graphs are included.
