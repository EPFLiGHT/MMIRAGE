# Custom Processor (Dynamic Python Functions)

The **Custom Processor** allows you to inject arbitrary, user-defined Python logic to process your dataset. MMIRAGE will execute it in a highly optimized, asynchronous process pool.

## Key Architectural Features

* **Memory Isolated:** Worker pools are initialized using a strict `"spawn"` multiprocessing context. This completely isolates your custom logic from MMIRAGE's main process preventing for example  memory leaks.
* **Concurrency:** Inside one shard, many workers can work at the same time, independently.
* **Strict Order Preservation:** The processor guarantees that the output of each row is outputed at their original position into the batch.
* **Dual-Layer Fault Tolerance:**
* **Soft Fail:** If rows throw a standard Python exceptions or timeouts, the pipeline catches it, logs the error, injects your predefined `fallback_value`, and keeps the batch moving.
* **Circuit Breaker (Hard Fail):** If the script behaves wrongly and hits a configured threshold of consecutive timeouts (`max_timeouts`) or exceptions (`max_errors`), the processor intentionally trips a circuit breaker, halts the pool, and cleanly fails the shard to prevent infinite pipeline hangs.


* **Seamless Local Imports:** Your custom script can safely import other local helper modules. Your script is loaded at runtime, and its folder is temporarily added to Python’s module search path (sys.path).

---

## How to use it ?

### 1. Writing Your Custom Script

Your custom script must contain a target function that accepts **exactly one argument: a dictionary** representing the current row's data (`VariableEnvironment`). It should return the value you want written to the pipeline's output variable.

**Example: `my_custom_logic.py**`

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

---

### 2. Pipeline Configuration

To use the custom processor, register it in your MMIRAGE YAML configuration file. You must define the processor execution parameters, the input mapping, and the output schema.

Because local custom processors write to intermediate `.arrow` shards by default, it is highly recommended to set `merge: true` in your execution parameters so MMIRAGE automatically generates your final `.jsonl` file.

```yaml
execution_params:
  merge: true                          # Automatically merge .arrow

processors:
  - type: "custom"
    script_path: "./my_custom_logic.py"  # Path to your python file
    function_name: "extract_metadata"    # Target function to execute inside the file
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

### Configuration Parameters Reference

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | `str` | *None* | **Required.** Must be set to `"custom"` to trigger the custom module processor. |
| `script_path` | `str` | *None* | **Required.** The relative or absolute path to your `.py` file. |
| `function_name` | `str` | *None* | **Required.** The exact name of the callable function inside your script. |
| `max_workers` | `int` | `1` | Number of concurrent processes spawned. Scale this based on CPU availability. |
| `timeout_ms` | `int` | `1000` | Maximum time (in milliseconds) a single row is allowed to process before soft-failing. |
| `max_timeouts` | `int` | `1` | Number of `TimeoutError` occurrences allowed before the circuit breaker trips and fails the shard. |
| `max_errors` | `int` | `1` | Number of standard `Exceptions` allowed before the circuit breaker trips. |
| `fallback_value` | `Any` | `None` | The default value safely written to the output variable if the script soft-fails. |