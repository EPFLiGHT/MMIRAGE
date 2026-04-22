# CLI Reference

All MMIRAGE commands share the pattern:

```
mmirage <subcommand> [flags]
```

Common flags available on most subcommands:

| Flag | Description |
|---|---|
| `--config PATH` | Path to a MMIRAGE YAML config file (**required**) |
| `--log-level LEVEL` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |

---

## `mmirage run`

Run the pipeline according to `execution_params.mode` and `execution_params.retry`.

```bash
mmirage run --config configs/config.yaml [--force-retry] [--shard-id N]
```

| Flag | Description |
|---|---|
| `--force-retry` | Enable retry orchestration even if `execution_params.retry` is `false` |
| `--shard-id N` | Run a single specific shard locally, ignoring the execution mode |

**Behaviour summary:**

- `mode: local` — Runs shards in the current Python environment.
- `mode: slurm` — Submits an sbatch array job.
- `retry: true` — After each run, automatically retries failed shards until all succeed or the retry budget is exhausted.
- `merge: true` — After all shards succeed, merges outputs into `<output_dir>/merged/`.

---

## `mmirage submit`

Submit a single SLURM array job without the retry/merge orchestration loop.

```bash
mmirage submit --config configs/config.yaml [--shard-ids 0,2,3] [--wait]
```

| Flag | Description |
|---|---|
| `--shard-ids N,N,...` | Comma-separated shard IDs to submit instead of the full array |
| `--wait` | Block until the submitted job finishes |

---

## `mmirage check`

Inspect shard status from the state directory and optionally submit retries.

```bash
mmirage check --config configs/config.yaml [--retry] [-y]
```

| Flag | Description |
|---|---|
| `--retry` | Submit a retry job for any failed shards |
| `-y`, `--yes` | Submit retries without prompting for confirmation |

Exits with code `0` if all shards succeeded, `1` otherwise.

---

## `mmirage retry`

Submit a retry job for failed shards without inspecting status interactively.

```bash
mmirage retry --config configs/config.yaml [-y]
```

| Flag | Description |
|---|---|
| `-y`, `--yes` | Submit retries without prompting |

---

## `mmirage merge`

Merge shard outputs for all datasets listed in `loading_params.datasets`.

```bash
mmirage merge --config configs/config.yaml [--output-root /path/to/merged]
```

| Flag | Description |
|---|---|
| `--output-root PATH` | Root directory for merged outputs. MMIRAGE creates one subdirectory per dataset. If omitted, each dataset is merged into `<dataset.output_dir>/merged` |

---

## `mmirage merge-dir`

Merge shard outputs directly from a directory, without a config file.

```bash
mmirage merge-dir --input-dir /path/to/shards --output-dir /path/to/merged
```

| Flag | Description |
|---|---|
| `--input-dir PATH` | Directory containing `shard_*` folders (one dataset), or a parent directory containing multiple dataset subdirectories |
| `--output-dir PATH` | Output directory for the merged dataset(s) |

If `shard_*` folders are present **directly** in `--input-dir`, MMIRAGE treats it as a single dataset and merges it there, ignoring nested internal folders such as `_pipeline_state`.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | All shards completed successfully |
| `1` | One or more shards failed or exceeded the retry budget |
