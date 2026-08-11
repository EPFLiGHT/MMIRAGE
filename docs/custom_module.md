# Custom Processor (Dynamic Python Functions)

The **Custom Processor** allows you to inject arbitrary, user-defined Python logic to process your dataset. MMIRAGE will execute it in a optimized, asynchronous process pool.

## Key Architectural Features

* **Memory Isolated:** Worker pools are initialized using a `"spawn"` multiprocessing context by default (see `start_method`). Your custom logic runs in separate processes, so a memory leak, a segfault or a fatal crash inside your script cannot corrupt MMIRAGE's main process.
* **Concurrency:** Inside one shard, many workers can work at the same time, independently.
* **Strict Order Preservation:** The processor guarantees that the output of each row is written at their original position into the batch.
* **Fault Tolerance:**
  * **Soft Fail:** If rows throw a standard Python exception or time out, the pipeline catches it, logs the error, injects your predefined `fallback_value`, and keeps the batch moving.
  * **Circuit Breaker (Hard Fail):** If the script behaves wrongly and hits a configured threshold of number of timeouts (`max_timeouts`) or exceptions (`max_errors`), the processor intentionally trips a circuit breaker, halts the pool, and cleanly fails the shard to prevent infinite pipeline hangs.
  * **Fatal Worker Crash:** If a worker process dies outright (OOM kill, segfault, `os._exit()`), the circuit breaker trips **immediately**, regardless of `max_errors`. A dead worker is not a recoverable row-level error, so no `fallback_value` is applied and the shard fails.
* **Seamless Local Imports:** Your custom script can safely import other local helper modules. Your script is loaded at runtime, and its folder is temporarily added to Python’s module search path (sys.path).

---

## How to use it?

### 1. Writing Your Custom Script

Your custom script must contain a target function that accepts **exactly one argument: a dictionary** representing the current row's data (`VariableEnvironment`). It should return the value you want written to the pipeline's output variable.

**Example**: `my_custom_logic.py`

```python
import re

def extract_address(row: dict) -> str:
    """
    Extracts eth addresses from the original text.
    """
    # Extract your target variable from the dictionary
    text = row.get("original_text_column", "")

    # Perform your custom logic
    addresses = re.findall(r'\b0x[a-fA-F0-9]{40}\b', text)

    if not addresses:
        return "NO_ADDRESS"

    return ", ".join(addresses)

```

> **Warning:** The pipeline will always pass the full dictionary of the current row environment. Extract what you need using `.get("variable_name")`.

#### What your function can return

Return plain data only — strings, numbers, lists, dicts, None. Returning a function, a class, or an instance of a class defined in your script will break the worker.

#### Beware of module-level code

Your script is imported **once per worker process**, not once per run. Anything at module scope — compiling regex patterns, reading a lookup file — runs `max_workers` times at pool startup, and once more each time a worker is replaced after a timeout:

```python
import json, re

# Runs once in EVERY worker process, then reused for all its rows
PATTERNS = [re.compile(p) for p in (r"\b\d{3}-\d{2}-\d{4}\b", r"\b0x[a-fA-F0-9]{40}\b")]
BLOCKLIST = set(json.load(open("./data/blocklist.json")))

def scrub(row: dict) -> str:
    text = row.get("text", "")
    for pattern in PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return " ".join(w for w in text.split() if w.lower() not in BLOCKLIST)
```

That is the right place for setup that is worth reusing across the rows a worker handles, as long as duplicating it `max_workers` times is cheap — compiled patterns, a small lookup table, a config file. It is the wrong place for anything large, since each worker keeps its own full copy in memory.

>Note : also that workers share no memory: a global counter or cache mutated by your function is local to one worker and is not visible to the others or to the main process.


---

### 2. Pipeline Configuration

To use the custom processor, register it in your MMIRAGE YAML configuration file. You must define the processor execution parameters, the input mapping, and the output schema.

Because local custom processors write to intermediate `.arrow` shards by default, it is highly recommended to set `merge: true` in your execution parameters so MMIRAGE automatically generates your final `.jsonl` file.

> **Note:** A relative `script_path` is resolved against the **current working directory of the run**, not against the location of the YAML file. Launch the job from your project root (as in the example below), or use an absolute path if you need the config to be location-independent.

```yaml
execution_params:
  merge: true                          # Automatically merge .arrow

processors:
  - type: "custom"
    script_path: "./my_custom_logic.py"  # Path to your python file
    function_name: "extract_address"    # Target function to execute inside the file
    max_workers: 4                       # Number of isolated worker processes running at the same time
    timeout_ms: 2000                     # Max execution time (in millisecond) per row
    max_timeouts: 5                      # Trip circuit breaker after 5 timeouts
    max_errors: 3                        # Trip circuit breaker after 3 script crashes
    fallback_value: "PIPELINE_ERROR"     # Injected if a row times out or crashes

loading_params:
  num_shards: 1
  batch_size: 500
  datasets:
    - type: "jsonl"
      path: "./data/input_data.jsonl"
      output_dir: "./output_data"

processing_params:
  inputs:
    - name: "my_text"  
      key: "original_text_column"  

  outputs:
    - name: "custom_result"
      type: "custom"  

  output_schema:  
    source_text: "{{ my_text }}"
    analysis_result: "{{ custom_result }}"

```

### 3. Running It

```bash
mmirage run --config configs/my_config.yaml
```

See the [CLI Reference](cli.md) for the full set of flags.

### Configuration Parameters Reference

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | `str` | *None* | **Required.** Must be set to `"custom"` to trigger the custom module processor. |
| `script_path` | `str` | *None* | **Required.** Path to your `.py` file. Relative paths are resolved against the working directory of the run, not against the config file. |
| `function_name` | `str` | *None* | **Required.** The exact name of the callable function inside your script. |
| `max_workers` | `int` | `1` | Number of concurrent processes spawned. Scale this based on CPU availability. |
| `timeout_ms` | `int` | *None* | Maximum time (in milliseconds) a single row is allowed to process before soft-failing. Left unset, rows run untimed and a hanging script stalls the shard indefinitely. |
| `max_timeouts` | `int` | `1` | Number of `TimeoutError` occurrences allowed before the circuit breaker trips and fails the shard. Counted cumulatively over the whole shard, the counter is never reset between batches. Inert when `timeout_ms` is unset. |
| `max_errors` | `int` | `1` | Number of standard `Exceptions` allowed before the circuit breaker trips. Cumulative over the shard, same as `max_timeouts`. |
| `fallback_value` | `Any` | `None` | The default value safely written to the output variable if the script soft-fails. Should have the same type as a normal return value. |
| `start_method` | `str` | `"spawn"` | How worker processes are created: `spawn`, `fork` or `forkserver`. See below. |

> **Tip:** The two counters are cumulative and never reset, so on a large shard a tolerant `max_errors: 3` will eventually trip on any script that fails even occasionally. Size them against the total number of rows in a shard, not against a batch — and keep them low deliberately if you would rather fail fast than produce a file silently full of `fallback_value`.

### Choosing a `start_method`

This only affects how long it takes to create a worker: rows are processed at the same speed under all three. Workers are created at pool startup, and a new one replaces any worker killed by a timeout or a fatal crash, so a script that times out pays this cost again — up to `max_timeouts` times before the shard fails.

* **`spawn` (default):** each worker is a fresh interpreter that re-imports `mmirage` and your script and inherits nothing from the shard process. Slowest to start, correct in every pipeline.
* **`fork`:** copies the shard process, so the pool starts much faster. But it copies memory without threads, and a lock held by one of those threads at fork time stays locked forever in the worker — the worker hangs, and you see timeouts instead of the real cause. Only safe when `custom` is the first entry of `processors:`; after `llm` or `image_gen` the shard process is already multi-threaded.
* **`forkserver`:** forks from a clean server process, avoiding the inherited locks. That server is started fresh for the single pool a shard creates, so it costs about as much as `spawn` without the safety of it.

Keep `spawn` unless shard startup is measurably a problem, which happens only with many very short shards.
