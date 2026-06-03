import base64
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

_BACKEND_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/mmirage/core/process/processors/image_gen/backends/sglang_backend.py"
)
_SPEC = importlib.util.spec_from_file_location("sglang_backend", _BACKEND_PATH)
sglang_backend = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(sglang_backend)
SGLangImageBackend = sglang_backend.SGLangImageBackend


class StubResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return self._body


def png_b64(color=(12, 34, 56)):
    image = Image.new("RGBA", (1, 1), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def json_response(payload):
    return StubResponse(json.dumps(payload).encode("utf-8"))


def install_urlopen_stub(monkeypatch, handler):
    monkeypatch.setattr(sglang_backend.urllib.request, "urlopen", handler)


def test_generate_one_posts_expected_payload_and_decodes_image(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return json_response({"data": [{"b64_json": png_b64()}]})

    install_urlopen_stub(monkeypatch, fake_urlopen)

    backend = SGLangImageBackend(
        "http://localhost:30000/v1",
        api_key="test-key",
        timeout_seconds=42,
        request_model="sd-model",
        validate_server=False,
    )

    image = backend.generate_one(
        prompt="a small lighthouse",
        negative_prompt="fog",
        seed=123,
        params={
            "width": 512,
            "height": 768,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "output_quality": 91,
            "sampler": "euler",
            "ignored": None,
        },
    )

    assert image.mode == "RGB"
    assert image.size == (1, 1)

    request, timeout = requests[0]
    assert timeout == 42
    assert request.full_url == "http://localhost:30000/v1/images/generations"
    assert request.get_method() == "POST"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["Content-type"] == "application/json"
    assert json.loads(request.data.decode("utf-8")) == {
        "prompt": "a small lighthouse",
        "n": 1,
        "response_format": "b64_json",
        "model": "sd-model",
        "negative_prompt": "fog",
        "seed": 123,
        "size": "512x768",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "output-quality": 91,
        "sampler": "euler",
    }


def test_generate_batch_sends_prompt_specific_negative_prompts_and_seeds(monkeypatch):
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data.decode("utf-8")))
        return json_response({"data": [{"b64_json": png_b64()}]})

    install_urlopen_stub(monkeypatch, fake_urlopen)

    backend = SGLangImageBackend(
        "http://localhost:30000",
        validate_server=False,
        max_concurrent_requests=1,
    )

    images = backend.generate_batch(
        ["first prompt", "second prompt"],
        negative_prompts=["low quality", None],
        seeds=[11, None],
        params={"size": "256x256", "output-compression": 80},
    )

    assert [image.size for image in images] == [(1, 1), (1, 1)]
    assert payloads == [
        {
            "prompt": "first prompt",
            "n": 1,
            "response_format": "b64_json",
            "negative_prompt": "low quality",
            "seed": 11,
            "size": "256x256",
            "output-compression": 80,
        },
        {
            "prompt": "second prompt",
            "n": 1,
            "response_format": "b64_json",
            "size": "256x256",
            "output-compression": 80,
        },
    ]


def test_generate_batch_rejects_mismatched_request_options():
    backend = SGLangImageBackend("http://localhost:30000", validate_server=False)

    with pytest.raises(ValueError, match="Expected 2 negative prompts"):
        backend.generate_batch(["one", "two"], negative_prompts=["only one"])

    with pytest.raises(ValueError, match="Expected 2 seeds"):
        backend.generate_batch(["one", "two"], seeds=[1])


@pytest.mark.parametrize(
    ("response", "match"),
    [
        ({}, "Unexpected SGLang image response"),
        ({"data": []}, "Unexpected SGLang image response"),
        ({"data": [{"b64_json": "not-base64"}]}, "invalid base64 image data"),
        (
            {
                "data": [
                    {
                        "b64_json": base64.b64encode(b"not an image").decode(
                            "ascii"
                        )
                    }
                ]
            },
            "Could not decode SGLang image response",
        ),
    ],
)
def test_generate_one_reports_bad_image_responses(monkeypatch, response, match):
    def fake_urlopen(request, timeout):
        return json_response(response)

    install_urlopen_stub(monkeypatch, fake_urlopen)
    backend = SGLangImageBackend("http://localhost:30000", validate_server=False)

    with pytest.raises(RuntimeError, match=match):
        backend.generate_one(prompt="a valid prompt")


def test_read_json_reports_http_url_and_json_failures(monkeypatch):
    backend = SGLangImageBackend("http://localhost:30000", validate_server=False)

    def http_error(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "server error",
            hdrs={},
            fp=io.BytesIO(b'{"error":"boom"}'),
        )

    install_urlopen_stub(monkeypatch, http_error)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        backend.generate_one(prompt="a valid prompt")

    def url_error(request, timeout):
        raise urllib.error.URLError("connection refused")

    install_urlopen_stub(monkeypatch, url_error)
    with pytest.raises(RuntimeError, match="Could not reach SGLang server"):
        backend.generate_one(prompt="a valid prompt")

    def non_json(request, timeout):
        return StubResponse(b"not json")

    install_urlopen_stub(monkeypatch, non_json)
    with pytest.raises(RuntimeError, match="non-JSON response"):
        backend.generate_one(prompt="a valid prompt")
