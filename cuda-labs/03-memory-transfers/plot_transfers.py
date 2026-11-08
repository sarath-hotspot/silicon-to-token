import argparse
import csv
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
METRICS = (
    "host_allocation_ms",
    "host_initialization_ms",
    "H2D_elapsed_ms",
    "D2H_elapsed_ms",
)
DEFAULT_MIN_BYTES = 256
DEFAULT_MAX_BYTES = 64 * 1024**2


def parse_run(text, size):
    lines = text.splitlines()
    if not lines or lines[-1] != "All transfer correctness checks passed":
        raise ValueError("Benchmark did not report successful correctness checks")
    metadata = {}
    buffers = {}
    current = metadata
    for line in lines[:-1]:
        key, value = line.split("=", 1)
        if key == "Memory":
            if value not in ("pageable", "pinned") or value in buffers:
                raise ValueError(f"Unexpected or duplicate memory type: {value}")
            current = buffers[value] = {}
        else:
            if key in current:
                raise ValueError(f"Duplicate benchmark field: {key}")
            current[key] = value
    if int(metadata["bytes_per_transfer"]) != size:
        raise ValueError("Benchmark returned the wrong transfer size")
    if int(metadata["n"]) * 4 != size:
        raise ValueError("Benchmark returned the wrong element count")
    if set(buffers) != {"pageable", "pinned"}:
        raise ValueError("Benchmark must report both memory types")
    metrics = METRICS
    if metadata.get("experiment") == "cold":
        metrics += ("cold_setup_round_trip_ms",)
    for values in buffers.values():
        for key in metrics:
            value = float(values[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid timing: {key}={value}")
            values[key] = value
    return metadata, buffers


def median(rows, size, memory, metric):
    return statistics.median(
        row[metric] for row in rows
        if row["bytes"] == size and row["memory"] == memory
    )


def size_label(size):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:g} {unit}"
        size /= 1024


def sustained_threshold(rows, metric):
    sizes = sorted({row["bytes"] for row in rows})
    faster = [
        median(rows, size, "pageable", metric) >=
        1.05 * median(rows, size, "pinned", metric)
        for size in sizes
    ]
    for i, size in enumerate(sizes):
        if all(faster[i:]) and len(sizes) - i >= 3:
            return size
    return None


def check_host_memory(size):
    with Path("/proc/meminfo").open() as source:
        available = next(
            int(line.split()[1]) * 1024
            for line in source if line.startswith("MemAvailable:")
        )
    limit_path = Path("/sys/fs/cgroup/memory.max")
    if limit_path.exists():
        limit = limit_path.read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            available = min(available, max(0, int(limit) - used))
    # Leave room for CUDA staging, the driver, and other users/processes.
    if 2 * size > available * 0.75:
        raise RuntimeError(
            f"{size_label(size)} transfers require {size_label(2 * size)} of host "
            f"buffers; only {size_label(available)} is available. "
            "Refusing allocation to retain a 25% safety reserve."
        )


def save_rows(rows, path):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Measure and plot transfer latency")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cold", action="store_true",
                        help="Directly time host setup and the first round trip, without warmups")
    parser.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if (args.min_bytes < 4 or args.max_bytes < args.min_bytes or
        any(value & (value - 1) for value in (args.min_bytes, args.max_bytes))):
        parser.error("Size bounds must be powers of two, with 4 <= min <= max")
    sizes = [2**power for power in range(
        args.min_bytes.bit_length() - 1, args.max_bytes.bit_length()
    )]
    check_host_memory(args.max_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = "cold_transfer_latency" if args.cold else "transfer_latency"
    csv_path = args.output_dir / f"{filename}.csv"

    rows = []
    environment = None
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment = "cold" if args.cold else "warm"
    for index, size in enumerate(sizes):
        for repeat in range(args.repeats):
            check_host_memory(size)
            first = "pageable" if (index + repeat) % 2 == 0 else "pinned"
            result = subprocess.run(
                [str(ROOT / "build" / "memory_transfers"), str(size // 4), first,
                 experiment],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(result.stdout, file=sys.stderr, end="")
                print(result.stderr, file=sys.stderr, end="")
                result.check_returncode()
            metadata, buffers = parse_run(result.stdout, size)
            if (metadata.get("experiment") != experiment or
                int(metadata["warmups"]) != (0 if args.cold else 5) or
                int(metadata["iterations"]) != (1 if args.cold else 20)):
                raise ValueError("Benchmark returned the wrong experiment or timing settings")
            identity = (metadata["gpu"], metadata["warmups"], metadata["iterations"])
            if environment is not None and identity != environment:
                raise ValueError("GPU or timing settings changed during the sweep")
            environment = identity
            for memory, values in buffers.items():
                setup = values["host_allocation_ms"] + values["host_initialization_ms"]
                round_trip = values["H2D_elapsed_ms"] + values["D2H_elapsed_ms"]
                rows.append({
                    "timestamp_utc": timestamp,
                    "gpu": metadata["gpu"],
                    "warmups": int(metadata["warmups"]),
                    "iterations": int(metadata["iterations"]),
                    "bytes": size,
                    "repeat": repeat + 1,
                    "first_memory": first,
                    "memory": memory,
                    "experiment": experiment,
                    **{key: values[key] for key in METRICS},
                    "host_setup_ms": setup,
                    "round_trip_ms": round_trip,
                    **({"cold_setup_round_trip_ms": values["cold_setup_round_trip_ms"]}
                       if args.cold else {"setup_plus_round_trip_ms": setup + round_trip}),
                })
            save_rows(rows, csv_path)
        print(f"Measured {size_label(size)} ({args.repeats} paired runs)", flush=True)

    panels = ((
        "cold_setup_round_trip_ms",
        "Measured host allocation + initialization + first round trip",
    ),) if args.cold else (
        ("H2D_elapsed_ms", "Host to device (reused buffers)"),
        ("D2H_elapsed_ms", "Device to host (reused buffers)"),
        ("setup_plus_round_trip_ms", "Host setup + one warmed round trip (estimate)"),
    )
    figure, axes = plt.subplots(1, 2 if args.cold else 3,
                               figsize=(14 if args.cold else 17, 5),
                               constrained_layout=True)
    for axis, (metric, title) in zip(axes, panels):
        for memory, color in (("pageable", "tab:orange"), ("pinned", "tab:blue")):
            samples = [
                [row[metric] for row in rows
                 if row["bytes"] == size and row["memory"] == memory]
                for size in sizes
            ]
            axis.plot(sizes, [statistics.median(values) for values in samples],
                      marker=".", label=memory, color=color)
            axis.fill_between(sizes, [min(values) for values in samples],
                              [max(values) for values in samples], color=color, alpha=0.15)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        ticks = sizes[::max(1, (len(sizes) + 6) // 7)]
        axis.set_xticks(ticks, [size_label(size) for size in ticks], rotation=35)
        axis.set_xlabel("Bytes per transfer")
        axis.set_ylabel("Latency (ms, lower is better)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    if args.cold:
        axis = axes[1]
        ratios = [
            median(rows, size, "pageable", "cold_setup_round_trip_ms") /
            median(rows, size, "pinned", "cold_setup_round_trip_ms")
            for size in sizes
        ]
        axis.plot(sizes, ratios, marker=".", color="tab:green")
        axis.axhline(1, color="gray", linestyle="--", label="Equal latency")
        axis.axhline(1.05, color="tab:blue", linestyle=":", label="5% pinned speedup")
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks, [size_label(size) for size in ticks], rotation=35)
        axis.set_xlabel("Bytes per transfer")
        axis.set_ylabel("Pageable / pinned median latency")
        axis.set_title("Above 1: pinned faster; below 1: pageable faster")
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle(
        f"{environment[0]}: {'cold buffers, no warmups' if args.cold else 'pageable vs pinned'}"
        f" | median of {args.repeats} runs; "
        "shading = min/max"
    )
    for extension in ("png", "svg"):
        figure.savefig(args.output_dir / f"{filename}.{extension}", dpi=160)
    plt.close(figure)

    print("\nObserved sustained thresholds (pageable/pinned median latency >= 1.05):")
    for metric, title in panels:
        threshold = sustained_threshold(rows, metric)
        print(f"{title}: {size_label(threshold) if threshold else 'none observed'}")
    print("Requires advantage at every larger sampled size, with at least 3 sizes.")
    print("These are sampled observations, not universal hardware thresholds.")
    if args.cold:
        print("\nDirectly measured host setup + first round trip (median ms):")
        for size in sizes:
            pageable = median(rows, size, "pageable", "cold_setup_round_trip_ms")
            pinned = median(rows, size, "pinned", "cold_setup_round_trip_ms")
            print(f"{size_label(size)}: pageable={pageable:.6f}, pinned={pinned:.6f}, "
                  f"pageable/pinned={pageable / pinned:.3f}")
        print(f"\nCSV and plots saved in {args.output_dir.resolve()}")
        return
    print("\nEstimated round trips needed to amortize host setup:")
    for size in sizes:
        saved = (median(rows, size, "pageable", "round_trip_ms") -
                 median(rows, size, "pinned", "round_trip_ms"))
        extra_setup = (median(rows, size, "pinned", "host_setup_ms") -
                       median(rows, size, "pageable", "host_setup_ms"))
        count = max(1, math.floor(extra_setup / saved) + 1) if saved > 0 else None
        print(f"{size_label(size)}: {count if count else 'no transfer advantage'}")
    print(f"\nCSV and plots saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
