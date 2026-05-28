# 📊 Benchmarking

This page explains how to collect and interpret throughput and efficiency
metrics for MMIRAGE pipeline runs.

---

## Overview

MMIRAGE includes built-in benchmarking that measures:

- wall-clock runtime
- throughput (rows per second)
- token generation speed per GPU
- GPU utilisation
- estimated compute cost (GPU-days per billion tokens)

Metrics are collected on each compute node during processing and recorded in the shard state directory (e.g. `<state_dir>/shard_<id>/status.json` under the `stats` key).
---

## Enabling benchmarking

Pass `--stats` to any command that submits or runs shards:

```bash
# Local run
mmirage run --config configs/config.yaml --stats

# SLURM array submission
mmirage submit --config configs/config.yaml --stats

# Retry failed shards with stats
mmirage retry --config configs/config.yaml --stats
```

When `--stats` is set, MMIRAGE polls GPU utilisation at a fixed interval
during processing and records token counts reported by the SGLang engine.

---

## Viewing collected metrics

After a run (or after shards complete), aggregate and display the metrics:

```bash
mmirage stats --config configs/config.yaml
```

This reads per-shard stats files from the state directory and prints a
summary to stdout in JSON format:

```json
{
  "per_shard": [],
  "aggregate": {
    "wall_clock_runtime_seconds": 3247.8,
    "overall_throughput_rows_per_sec": 12.4,
    "tokens_per_sec_per_gpu": 1850.3,
    "gpu_days_per_billion_tokens": 0.63,
    "mean_gpu_util_pct": 91.2
  }
}
```

---

## Metrics reference

| Metric | Unit | Description |
|---|---|---|
| `runtime_seconds` | seconds | Total wall-clock time across all shards |
| `overall_throughput_rows_per_sec` | rows/s | Dataset samples processed per second |
| `tokens_per_sec_per_gpu` | tokens/s/GPU | Token generation throughput normalised per GPU |
| `gpu_days_per_billion_tokens` | GPU-days | Compute cost estimate: GPU-days needed to generate one billion tokens |
| `mean_gpu_util_pct` | % | Average GPU utilisation during processing |

---

## Interpreting results

**`tokens_per_sec_per_gpu`** is the primary efficiency indicator.
Higher is better. Typical values for a well-configured SGLang engine on modern
hardware range from 1 000 to 5 000+ tokens/s/GPU depending on model size and
batch size.

**`gpu_days_per_billion_tokens`** normalises cost across different GPU counts
and runtimes, making it easy to compare runs with different configurations.

**`mean_gpu_util_pct`** should ideally stay above 80 %.
Low values (< 60 %) may indicate:
- `batch_size` is too small relative to the model's throughput
- heavy I/O overhead between batches
- slow JMESPath extraction or prompt rendering

---

## Tuning for throughput

If benchmarking reveals low efficiency, try:

| Symptom | Remedy |
|---|---|
| Low GPU util, high I/O wait | Increase `batch_size` |
| OOM errors at large batch size | Reduce `batch_size` or `max_new_tokens` |
| Low tokens/s/GPU | Tune `extra_engine_args` (e.g. `chunked_prefill_size`, `max_running_requests`) |
| High variance across shards | Check dataset skew (very long samples in one shard) |

---

## DataTrove benchmark

MMIRAGE includes a reference config for the DataTrove benchmark dataset:

```
configs/config_benchmark_datatrove.yaml
```

Use it to establish a baseline throughput on your hardware and compare
against published numbers.

---

## See also

- [CLI Reference](cli.md) — `--stats` flag and `mmirage stats` command details
- [SLURM & Cluster Deployment](slurm.md) — collecting stats in SLURM mode
- [Configuration Reference](configuration.md) — `extra_engine_args` for SGLang tuning
