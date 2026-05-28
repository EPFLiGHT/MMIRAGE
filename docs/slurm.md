# 🔀 SLURM & Cluster Deployment

This page explains how to run MMIRAGE pipelines on HPC clusters using SLURM.

For background on sharding and execution modes, see [Concepts](concepts.md).

---

## Overview

MMIRAGE has native SLURM support.
When `execution_params.mode` is set to `slurm`, the CLI generates and submits
an `sbatch` array job where each array task processes one shard.

The CLI then polls SLURM for job completion, checks per-shard status files,
retries failed shards (if configured), and optionally merges outputs.

---

## Prerequisites

- SLURM must be available on your cluster (`sbatch`, `squeue`, `srun` in `$PATH`).
- Your cluster nodes must have access to the model weights and dataset paths.
- MMIRAGE and its dependencies (`gpu` extra) must be installed on the nodes.

To prepare a portable environment for cluster nodes, use the helper script:

```bash
bash scripts/generate_env.sh
```

This creates a self-contained environment archive you can distribute to compute
nodes or load via a module system.

---

## SLURM configuration

Set `execution_params.mode: slurm` and fill in the SLURM-specific fields:

```yaml
execution_params:
  mode: slurm
  account: my_account           # SLURM account / allocation
  job_name: mmirage-pipeline    # Job name shown in squeue
  nodes: 1                      # Nodes per shard task
  ntasks_per_node: 1            # Tasks per node
  gpus: 4                       # GPUs per task (= tp_size)
  cpus_per_task: 64             # CPU cores per task
  time_limit: "11:59:59"        # Wall-clock limit per task
  retry: true                   # Retry failed shards
  merge: true                   # Merge outputs after success
  max_retries: 3                # Maximum retry attempts per shard
  settle_time_seconds: 30       # Seconds to wait after job ends
```

The `gpus` value should match `processors[0].server_args.tp_size` so that the
SGLang engine uses all GPUs allocated to the task.

---

## Submission workflow

### 1. Submit

```bash
mmirage run --config configs/slurm_config.yaml
```

This generates an `sbatch` script, submits it as a job array, and enters a
polling loop until all shards finish.

To submit without waiting, use `mmirage submit`:

```bash
mmirage submit --config configs/slurm_config.yaml
```

### 2. Monitor

Check which shards have succeeded, are pending, or have failed:

```bash
mmirage check --config configs/slurm_config.yaml
```

This reads the state directory and prints a per-shard status table.

### 3. Retry

Resubmit only the shards that failed:

```bash
mmirage retry --config configs/slurm_config.yaml
```

Retry respects `max_retries`: shards that have already been retried the
maximum number of times are skipped and reported as exhausted.

### 4. Merge

Once all shards succeed, merge their outputs into a single dataset:

```bash
mmirage merge --config configs/slurm_config.yaml
```

This combines all `shard_<id>/` directories under `output_dir` into
`<output_dir>/merged/`.

---

## Shard ID resolution

In SLURM mode, set `loading_params.shard_id` to the SLURM array task ID:

```yaml
loading_params:
  num_shards: 16
  shard_id: "$SLURM_ARRAY_TASK_ID"
```

MMIRAGE resolves `$SLURM_ARRAY_TASK_ID` from the environment at config load
time, so each array task automatically processes the correct slice of the data.

---

## Tips for HPC environments

**Shared filesystem writes**
MMIRAGE uses atomic temp-then-rename writes to avoid partial files on shared
filesystems. No extra configuration is needed.

**Tensor parallelism**
Match `tp_size` to the number of GPUs per task.
For large models (70B+), use `tp_size: 8` and request 8 GPUs per task.

**Wall-clock budget**
Set `time_limit` generously for the first run.
Once you know how long a shard takes, you can tighten it.

**Environment modules**
If your cluster uses modules, activate them before submitting:

```bash
module load python/3.12 cuda/12.4
mmirage run --config configs/slurm_config.yaml
```

---

## See also

- [Concepts](concepts.md) — shards, state directory, retry and merge
- [Pipeline](pipeline.md) — how SLURM mode fits into the pipeline
- [CLI Reference](cli.md) — `submit`, `check`, `retry`, `merge` subcommands
- [Configuration Reference](configuration.md) — full `execution_params` reference
- [Benchmarking](benchmarking.md) — measuring throughput on cluster jobs
