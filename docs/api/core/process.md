# `mmirage.core.process` — Processors

## Variables

```{eval-rst}
.. automodule:: mmirage.core.process.variables
   :members:
   :undoc-members:
   :show-inheritance:
```

## Base processor

```{eval-rst}
.. automodule:: mmirage.core.process.base
   :members:
   :undoc-members:
   :show-inheritance:
```

## Mapper

```{eval-rst}
.. automodule:: mmirage.core.process.mapper
   :members:
   :undoc-members:
   :show-inheritance:
```

## LLM processor

### Configuration

```{eval-rst}
.. automodule:: mmirage.core.process.processors.llm.config
   :members:
   :undoc-members:
   :show-inheritance:
```

### Implementation

```{eval-rst}
.. automodule:: mmirage.core.process.processors.llm.llm_processor
   :members:
   :undoc-members:
   :show-inheritance:
```

## Batch API processor

### Configuration

```{eval-rst}
.. automodule:: mmirage.core.process.processors.batch_api.config
   :members:
   :undoc-members:
   :show-inheritance:
```

### Implementation

```{eval-rst}
.. automodule:: mmirage.core.process.processors.batch_api.batch_api_processor
   :members:
   :undoc-members:
   :show-inheritance:
```

## Batch processing

The batch subsystem handles asynchronous, provider-backed inference (e.g. OpenAI Batch API). It is activated by setting `batch_provider` in the processor configuration.

### Orchestrator

```{eval-rst}
.. automodule:: mmirage.core.process.batch.orchestrator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Adapter (provider-neutral interface)

```{eval-rst}
.. automodule:: mmirage.core.process.batch.adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### OpenAI adapter

```{eval-rst}
.. automodule:: mmirage.core.process.batch.openai_adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Anthropic adapter

```{eval-rst}
.. automodule:: mmirage.core.process.batch.anthropic_adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Chunking

```{eval-rst}
.. automodule:: mmirage.core.process.batch.chunking
   :members:
   :undoc-members:
   :show-inheritance:
```

### Status checker

```{eval-rst}
.. automodule:: mmirage.core.process.batch.status_checker
   :members:
   :undoc-members:
   :show-inheritance:
```
