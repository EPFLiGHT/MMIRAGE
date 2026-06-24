import hashlib
import sys
import types
from pathlib import Path

from PIL import Image

_SRC = Path(__file__).resolve().parents[2] / "src"
_PKG = types.ModuleType("mmirage")
_PKG.__path__ = [str(_SRC / "mmirage")]
sys.modules.setdefault("mmirage", _PKG)

_JMESPATH = types.ModuleType("jmespath")
_JMESPATH.compile = lambda key: None
_JMESPATH.search = lambda key, sample: sample.get(key)
_JMESPATH.parser = types.SimpleNamespace(ParsedResult=object)
sys.modules.setdefault("jmespath", _JMESPATH)

from mmirage.core.process.processors.image_gen.config import (  # noqa: E402
    ExternalImageBackendConfig,
    ImageGenConfig,
    ImageGenOutputVar,
    ImageOutputMode,
)
from mmirage.core.process.processors.image_gen.image_gen_processor import (  # noqa: E402
    ImageGenProcessor,
)
from mmirage.core.process.variables import VariableEnvironment  # noqa: E402


class InMemoryBackend:
    def __init__(self, *, fail_batch=False):
        self.fail_batch = fail_batch
        self.batch_calls = []
        self.one_calls = []

    def generate_batch(self, prompts, negative_prompts, params, seeds):
        self.batch_calls.append((prompts, negative_prompts, params, seeds))
        if self.fail_batch:
            raise RuntimeError("batch backend failed")
        return [Image.new("RGB", (1, 1), "red") for _ in prompts]

    def generate_one(self, *, prompt, negative_prompt=None, params=None, seed=None):
        self.one_calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "params": params,
                "seed": seed,
            }
        )
        return Image.new("RGB", (1, 1), "blue")

    def shutdown(self):
        pass


def _processor(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(
        "mmirage.core.process.processors.image_gen.image_gen_processor._create_backend",
        lambda config: backend,
    )
    config = ImageGenConfig(
        type="image_gen",
        backend="external",
        external=ExternalImageBackendConfig(base_url="http://image-server.invalid"),
        output_dir=str(tmp_path),
        parallel_inference=True,
        parallel_chunk_size=4,
    )
    return ImageGenProcessor(config)


def _output_var():
    return ImageGenOutputVar(
        name="generated_image",
        type="image_gen",
        prompt="a {{ subject }}",
        negative_prompt="bad {{ subject }}",
        output_mode=ImageOutputMode.PIL,
        seed=10,
    )


def test_compute_source_hash_uses_explicit_empty_payload():
    env = VariableEnvironment({})

    assert ImageGenProcessor._compute_source_hash(env) == hashlib.sha256(
        b"empty"
    ).hexdigest()[:8]


def test_parallel_batch_failure_falls_back_to_sequential_and_updates_counter(
    monkeypatch,
    tmp_path,
):
    backend = InMemoryBackend(fail_batch=True)
    processor = _processor(monkeypatch, tmp_path, backend)
    batch = [
        VariableEnvironment({"subject": "cat"}),
        VariableEnvironment({"subject": "dog"}),
    ]

    result = processor.batch_process_sample(batch, _output_var())

    assert len(backend.batch_calls) == 1
    assert [call["prompt"] for call in backend.one_calls] == ["a cat", "a dog"]
    assert [call["negative_prompt"] for call in backend.one_calls] == [
        "bad cat",
        "bad dog",
    ]
    assert [call["seed"] for call in backend.one_calls] == [10, 11]
    assert processor._sample_counter == 2
    assert [env.is_image_var("generated_image") for env in result] == [True, True]
    assert all(isinstance(env.get("generated_image"), Image.Image) for env in result)


def test_parallel_batch_success_sets_output_variable_and_updates_counter(
    monkeypatch,
    tmp_path,
):
    backend = InMemoryBackend()
    processor = _processor(monkeypatch, tmp_path, backend)
    batch = [
        VariableEnvironment({"subject": "cat"}),
        VariableEnvironment({"subject": "dog"}),
    ]

    result = processor.batch_process_sample(batch, _output_var())

    assert len(backend.batch_calls) == 1
    assert len(backend.one_calls) == 0
    assert processor._sample_counter == 2
    assert [env.is_image_var("generated_image") for env in result] == [True, True]
    assert all(isinstance(env.get("generated_image"), Image.Image) for env in result)
