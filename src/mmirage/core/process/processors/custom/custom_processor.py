"""Custom processor implementation for executing user-defined Python functions."""

import concurrent.futures
import logging
import time
from typing import Any, List

from pebble import ProcessPool

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry, TokenCounts
from mmirage.core.process.variables import VariableEnvironment
from mmirage.core.process.processors.custom.config import CustomProcessorConfig, CustomOutputVar
from mmirage.core.process.processors.custom.worker import execute_custom_function, initialize_worker

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("custom", CustomProcessorConfig, CustomOutputVar)
class CustomProcessor(BaseProcessor[CustomOutputVar]):
    """Processor that runs user-provided Python scripts in a persistent, isolated process pool.
    
    Ensures safe execution by using a 'spawn' context and implements strict circuit breaking
    for misbehaving scripts.
    """

    def __init__(self, config: CustomProcessorConfig) -> None:
   
        start_time = time.time()
        super().__init__(config)
        self.config: CustomProcessorConfig = config
        self._error_count = 0
        self._timeout_count = 0
        self._is_broken = False

        #check user script file existence
        if not os.path.exists(self.config.script_path):
            raise FileNotFoundError(
                f"CustomProcessor failed to boot: script_path '{self.config.script_path}' does not exist."
            )

        self._pool = ProcessPool(
            max_workers=self.config.max_workers,
            context="spawn", # use 'spawn' context for safe and memory-efficient execution (but slower than 'fork')
            initializer=initialize_worker,
            initargs=(self.config.script_path, self.config.function_name)
        )
        
        self._load_time = time.time() - start_time
        logger.info(
            f"Initialized CustomProcessor pool with {self.config.max_workers} workers "
            f"(target: {self.config.function_name} in {self.config.script_path})"
        )

    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: CustomOutputVar
    ) -> List[VariableEnvironment]:
        """Process a batch of data rows through the user's custom function.
        
        Implements an order-preserving fan-out/fan-in execution model with soft 
        and hard failure recovery.
        """

        if self._is_broken:
            raise RuntimeError(
                f"CustomProcessor circuit breaker tripped. Max timeouts ({self.config.max_timeouts}) reached."
            )

        #to guarantee strict input ordering
        results: List[Any] = [None] * len(batch)
        
        row_dicts = [env.to_dict() for env in batch]
        
        timeout_seconds = self.config.timeout_ms / 1000.0
        future_to_index = {}
        for index, row_dict in enumerate(row_dicts):
            future = self._pool.schedule(
                execute_custom_function, 
                args=(row_dict,), 
                timeout=timeout_seconds
            )
            future_to_index[future] = index

        for future in concurrent.futures.as_completed(future_to_index.keys()):
            index = future_to_index[future]
            
            try:
                result = future.result()
                results[index] = result
                
            except concurrent.futures.TimeoutError:
                self._timeout_count += 1
                logger.warning(f"Row {index} timed out. Applying fallback value.")
                results[index] = self.config.fallback_value
                
                if self._timeout_count >= self.config.max_timeouts:
                    self._trip_circuit_breaker("Max timeouts reached. Tripping circuit breaker and shutting down pool.")
            except ProcessExpired as e:
                # critical error : process died unexpectedly during its execution
                self._trip_circuit_breaker(f"Worker process crashed fatally : {e}")
            except Exception as e:
                # if user script raises exception 
                logger.error(f"User script raised an exception on row {index}: {e}. Applying fallback value.")
                results[index] = copy.deepcopy(self.config.fallback_value)
                self._error_count += 1
                if self._error_count >= self.config.max_errors:
                    self._trip_circuit_breaker("Max errors reached. Tripping circuit breaker and shutting down pool.")

        for i, env in enumerate(batch):
            env.set_var(output_var.name, results[i])
            
        return batch

    def finalize(self) -> None:
        """shut down the persistent worker pool."""
        if not self._is_broken and hasattr(self, "_pool"):
            self._pool.stop()
            self._pool.join()
    
    def _trip_circuit_breaker(self, reason: str) -> None:
        """Helper to centralize circuit breaker hard-fail logic."""
        self._is_broken = True
        logger.error(f"Tripping circuit breaker: {reason}")
        self._pool.stop()
        self._pool.join()
        raise RuntimeError(f"Custom processor circuit breaker tripped: {reason}")

    def get_token_counts(self) -> TokenCounts:
        """Return zero token counts as this is not an LLM processor."""
        return TokenCounts(input_tokens=0, output_tokens=0)

    def get_load_time(self) -> float:
        """Return the time taken to initialize the processor and its pool."""
        return self._load_time

