# Silicon to Token

Hands-on CUDA labs for understanding GPU computation, memory, and transfer
performance.

## Labs

- [01-vector-add](cuda-labs/01-vector-add/README.md): vector addition, grid-stride
  loops, kernel timing, and effective bandwidth.
- [02-memory](cuda-labs/02-memory/README.md): explicit and unified memory with
  GPU and CPU computation timing.
- [03-memory-transfers](cuda-labs/03-memory-transfers/README.md): pageable versus
  pinned transfers, cold-buffer setup costs. Experiments on at what buffer size it is worth using pinned vs oageable. 

## Build and test

Requires a CUDA-capable NVIDIA GPU, a working driver, CUDA Toolkit, CMake
3.24 or newer, and a supported host C++ compiler.

```bash
for lab in 01-vector-add 02-memory 03-memory-transfers; do
  "./cuda-labs/$lab/build.sh"
  ctest --test-dir "./cuda-labs/$lab/build" --output-on-failure
done
```

Each lab builds into its own `build/` directory. See the lab READMEs for run
commands, timing boundaries, and plotting dependencies. Generated binaries,
build directories, and Python virtual environments are excluded from Git;
compact measured CSVs and graphs are included.
