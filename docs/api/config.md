# `mmirage.config` — Configuration

## `mmirage.config.config`

Main configuration dataclasses for the MMIRAGE pipeline.

```{eval-rst}
.. automodule:: mmirage.config.config
   :members:
   :undoc-members:
   :show-inheritance:
```

## `mmirage.config.loading`

Dataset loading and sharding configuration.

```{eval-rst}
.. automodule:: mmirage.config.loading
   :members:
   :undoc-members:
   :show-inheritance:
```

## `mmirage.config.batch_provider`

Provider-agnostic batch submission configuration used by the OpenAI Batch API integration and any future batch providers.

```{eval-rst}
.. automodule:: mmirage.config.batch_provider
   :members:
   :undoc-members:
   :show-inheritance:
```

## `mmirage.config.openai_batch`

OpenAI-specific batch configuration extending {class}`~mmirage.config.batch_provider.BatchProviderConfig`.

```{eval-rst}
.. automodule:: mmirage.config.openai_batch
   :members:
   :undoc-members:
   :show-inheritance:
```

## `mmirage.config.utils`

YAML parsing helpers and configuration loaders.

```{eval-rst}
.. automodule:: mmirage.config.utils
   :members:
   :undoc-members:
   :show-inheritance:
```
