from mmirage.core.process.base import ProcessorRegistry
from mmirage.core.process.processors.llm.config import SGLangLLMConfig, SGLangServerArgs


class FakeEngine:
    def __init__(self, **_kwargs):
        return None

    def generate(self, **_kwargs):
        raise AssertionError("Generation should not run in this test")

    def shutdown(self):
        return None


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return ""


def test_llm_processor_initializes_sglang_runtime(monkeypatch):
    monkeypatch.setattr(
        "mmirage.core.process.processors.llm.llm_processor.sgl.Engine",
        FakeEngine,
    )
    monkeypatch.setattr(
        "mmirage.core.process.processors.llm.llm_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "mmirage.core.process.processors.llm.llm_processor.SGLANG_AVAILABLE",
        True,
    )

    config = SGLangLLMConfig(
        type="llm",
        server_args=SGLangServerArgs(model_path="dummy-model"),
        default_sampling_params={"temperature": 0.1},
        chat_template="qwen2-vl",
    )

    processor = ProcessorRegistry.get_processor("llm")(config)

    assert isinstance(processor.llm, FakeEngine)
    assert isinstance(processor.tokenizer, FakeTokenizer)
    assert processor.sampling_params == {"temperature": 0.1}
    assert processor.chat_template == "qwen2-vl"
