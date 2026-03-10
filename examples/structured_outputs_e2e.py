"""End-to-end MVP example for MMIRAGE structured outputs.

Run:
    python examples/structured_outputs_e2e.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from mmirage.core.process.structured_outputs import (
    StructuredOutputBackend,
    StructuredTaskRunner,
    TaskSpecAST,
    compile_task,
    parse_task_spec_yaml,
)


TASK_SPEC_YAML = """
input_schema:
  passage:
    type: string
output_schema:
  summary:
    type: string
  word_count:
    type: integer

dataset_mapping:
  inputs:
    passage: text
  outputs:
    structured.summary: summary
    structured.length: word_count

execution:
  backend: dspy
  instruction: Summarize the passage and count words in the summary.
"""


@dataclass
class DemoProgram:
    pass


class DemoBackend(StructuredOutputBackend):
    """Small deterministic backend for demonstration.

    Swap with `DspyBackend()` in production.
    """

    def compile(self, ast: TaskSpecAST) -> DemoProgram:
        return DemoProgram()

    def infer(self, backend_program: DemoProgram, inputs: Dict[str, Any]) -> Dict[str, Any]:
        words = str(inputs["passage"]).split()
        summary = " ".join(words[:8])
        return {"summary": summary, "word_count": len(summary.split())}


def main() -> None:
    ast = parse_task_spec_yaml(TASK_SPEC_YAML)
    backend = DemoBackend()
    compiled = compile_task(ast, backend)

    runner = StructuredTaskRunner(compiled, backend)
    sample = {"text": "MMIRAGE helps teams process large multimodal datasets with flexible pipelines."}

    result = runner.run_sample(sample)
    print(result)


if __name__ == "__main__":
    main()
