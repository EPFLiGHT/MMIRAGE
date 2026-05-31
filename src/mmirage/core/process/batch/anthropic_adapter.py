"""Concrete Anthropic implementation of batch submission contracts."""

from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import hashlib

from anthropic import Anthropic

from mmirage.config.anthropic_batch import AnthropicBatchConfig
from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult

logger = logging.getLogger(__name__)


class AnthropicBatchAdapter(BatchSubmissionAdapter):
    """Provider adapter for Anthropic Messages Batches API."""

    required_credentials = ("api_key",)

    def build_request(
        self,
        custom_id: str,
        payload: Dict[str, Any],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        anthropic_config = self._check_anthropic_config(config)
        normalized_custom_id = self._normalize_custom_id(custom_id)
        body = copy.deepcopy(payload)
        expected_schema = body.pop("expected_schema", None)

        if expected_schema is not None and (
            not isinstance(expected_schema, list)
            or not all(isinstance(key, str) for key in expected_schema)
        ):
            raise ValueError(
                "expected_schema must be a list of strings, "
                f"got {type(expected_schema).__name__}"
            )

        body.setdefault("model", anthropic_config.model)
        body.setdefault("max_tokens", anthropic_config.max_tokens)
        body.setdefault("temperature", anthropic_config.temperature)
        if anthropic_config.top_p is not None:
            body.setdefault("top_p", anthropic_config.top_p)
        if "temperature" in body and "top_p" in body:
            body.pop("top_p", None)

        messages = body.get("messages")
        if isinstance(messages, list):
            body["messages"] = self._normalize_messages(messages)

        if isinstance(expected_schema, list) and all(isinstance(k, str) for k in expected_schema):
            properties = {key: {"type": "string"} for key in expected_schema}
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": expected_schema,
                        "additionalProperties": False,
                    },
                }
            }
        
        payload_request = {
            "custom_id": normalized_custom_id,
            "params": body,
        }
        #for debug :
        print(f"Built request for custom_id={normalized_custom_id}: {payload_request}")
        return payload_request

    def estimate_request_bytes(self, request: Dict[str, Any]) -> int:
        serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        return len(serialized.encode("utf-8"))

    def submit_chunk(
        self,
        chunk_id: str,
        requests: Sequence[Dict[str, Any]],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        anthropic_config = self._check_anthropic_config(config)
        client = self._create_client(anthropic_config)
        batches_client = self._resolve_batches_client(client)

        metadata = dict(anthropic_config.metadata)
        metadata["chunk_id"] = chunk_id

        try:
            response = batches_client.create(requests=requests, metadata=metadata)
        except TypeError:
            response = batches_client.create(requests=requests)

        return {
            "id": self._attr_or_get(response, "id", ""),
            "status": self._attr_or_get(response, "status", None),
            "chunk_id": chunk_id,
        }

    def parse_submission_result(self, raw_result: Dict[str, Any]) -> BatchSubmissionResult:
        coerced = self._coerce_mapping(raw_result)
        batch_id = str(self._attr_or_get(raw_result, "id", "") or "")
        status = self._attr_or_get(raw_result, "status", None)
        if not status and isinstance(coerced, Mapping):
            status = (
                coerced.get("status")
                or coerced.get("processing_status")
                or coerced.get("state")
                or coerced.get("status_code")
            )
        status = status or "unknown"
        return BatchSubmissionResult(
            provider_batch_id=batch_id,
            status=status,
            raw_response=raw_result,
        )

    def check_batch_status(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> BatchSubmissionResult:
        anthropic_config = self._check_anthropic_config(config)
        client = self._create_client(anthropic_config)
        batches_client = self._resolve_batches_client(client)
        retrieved = batches_client.retrieve(provider_batch_id)
        return self.parse_submission_result(raw_result=retrieved)

    def retrieve_results(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> Sequence[Dict[str, Any]]:
        """Download completed Anthropic batch rows and normalize text into ``generated_text``."""
        anthropic_config = self._check_anthropic_config(config)
        client = self._create_client(anthropic_config)
        batches_client = self._resolve_batches_client(client)

        retrieved = batches_client.retrieve(provider_batch_id)
        status = self._normalize_status(self.parse_submission_result(raw_result=retrieved).status)
        if status not in {"completed", "succeeded"}:
            raise ValueError(
                f"Batch '{provider_batch_id}' is not completed yet (status={status})."
            )

        results_response = batches_client.results(provider_batch_id)
        rows = self._parse_results_response(results_response)
        normalized: List[Dict[str, Any]] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            if not row.get("custom_id"):
                custom_id = self._extract_custom_id(row)
                if custom_id:
                    row["custom_id"] = custom_id
            error_message = self._extract_error_message(row)
            if error_message:
                row.setdefault("status", "error")
                row["error_message"] = error_message
            if "generated_text" not in row:
                generated_text = self._extract_generated_text(row)
                if generated_text:
                    row["generated_text"] = generated_text
            normalized.append(row)

        return normalized

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                content_blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                content_blocks = AnthropicBatchAdapter._normalize_content_blocks(content)
            else:
                content_blocks = []
            normalized.append({"role": role, "content": content_blocks})
        return normalized

    @staticmethod
    def _normalize_content_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for part in blocks:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if isinstance(text, str):
                    normalized.append({"type": "text", "text": text})
                continue
            if part_type == "image_url":
                url = part.get("image_url", {}).get("url")
                if isinstance(url, str):
                    source = AnthropicBatchAdapter._image_source_from_url(url)
                    normalized.append({"type": "image", "source": source})
                continue
            if part_type == "image" and isinstance(part.get("source"), dict):
                normalized.append({"type": "image", "source": dict(part["source"])})
        return normalized

    @staticmethod
    def _normalize_custom_id(custom_id: str) -> str:
        # Anthropic requires custom_id to match ^[a-zA-Z0-9_-]{1,64}$
        raw = str(custom_id).strip()
        if not raw:
            raise ValueError("custom_id must be a non-empty string")

        safe_chars = []
        for ch in raw:
            if ch.isalnum() or ch in {"_", "-"}:
                safe_chars.append(ch)
            else:
                safe_chars.append("_")

        normalized = "".join(safe_chars)
        if normalized == raw and len(normalized) <= 64:
            return normalized

        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        max_base = 64 - (1 + len(digest))
        if max_base < 1:
            return digest[:64]

        base = normalized[:max_base]
        return f"{base}-{digest}"

    @staticmethod
    def _image_source_from_url(url: str) -> Dict[str, str]:
        if url.startswith("data:"):
            media_type, data = AnthropicBatchAdapter._parse_data_uri(url)
        elif url.startswith("http://") or url.startswith("https://"):
            raise ValueError(
                "Anthropic batch requests require local image paths or data URIs; remote URLs are unsupported."
            )
        else:
            if not os.path.exists(url):
                raise ValueError(f"Image path does not exist: {url}")
            media_type = AnthropicBatchAdapter._guess_mime_type(url)
            with open(url, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")

        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    @staticmethod
    def _parse_data_uri(data_uri: str) -> tuple[str, str]:
        try:
            header, encoded = data_uri.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid data URI format") from exc

        if ";base64" not in header:
            raise ValueError("Data URI must be base64 encoded")

        media_type = header.replace("data:", "").split(";", 1)[0] or "application/octet-stream"
        return media_type, encoded

    @staticmethod
    def _guess_mime_type(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        return mime_type or "image/jpeg"

    @staticmethod
    def _extract_generated_text(row: Dict[str, Any]) -> str:
        # Anthropic batch results contain result.message.content blocks.
        try:
            result = row.get("result", {})
            message = result.get("message", {})
            content = message.get("content", [])
            if isinstance(content, list):
                texts = [block.get("text", "") for block in content if block.get("type") == "text"]
                return "".join(texts)
        except Exception:
            pass
        return ""

    @staticmethod
    def _extract_error_message(row: Dict[str, Any]) -> str:
        try:
            result = row.get("result", {})
            result_type = result.get("type")
            if result_type in {"error", "errored", "failed"}:
                error = result.get("error", {})
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str):
                        return message
        except Exception:
            pass
        return ""

    @staticmethod
    def _parse_results_response(response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, list):
            return [AnthropicBatchAdapter._coerce_mapping(item) for item in response]

        coerced = AnthropicBatchAdapter._coerce_mapping(response)
        if isinstance(coerced, dict):
            return [coerced]

        if isinstance(response, (str, bytes)):
            text = response.decode("utf-8") if isinstance(response, bytes) else response
            return AnthropicBatchAdapter._parse_jsonl(text)

        if isinstance(response, Iterable):
            rows: List[Dict[str, Any]] = []
            for item in response:
                coerced_item = AnthropicBatchAdapter._coerce_mapping(item)
                if isinstance(coerced_item, dict):
                    rows.append(coerced_item)
                elif isinstance(item, str):
                    rows.extend(AnthropicBatchAdapter._parse_jsonl(item))
            return rows

        logger.debug("Unhandled Anthropic results response type: %s", type(response))
        return []

    @staticmethod
    def _normalize_status(status: Any) -> str:
        value = str(status or "").strip().lower()
        if value in {"ended", "finished", "complete", "completed", "succeeded", "success"}:
            return "completed"
        if value in {"in_progress", "processing", "running", "queued"}:
            return "in_progress"
        if value in {"failed", "errored", "error"}:
            return "failed"
        return value or "unknown"

    @staticmethod
    def _parse_jsonl(text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows

    @staticmethod
    def _coerce_mapping(item: Any) -> Any:
        if isinstance(item, Mapping):
            return dict(item)
        if hasattr(item, "model_dump"):
            try:
                return dict(item.model_dump())
            except Exception:
                return item
        if hasattr(item, "dict"):
            try:
                return dict(item.dict())
            except Exception:
                return item
        return item

    @staticmethod
    def _extract_custom_id(row: Mapping[str, Any]) -> str:
        try:
            custom_id = row.get("custom_id")
            if isinstance(custom_id, str) and custom_id.strip():
                return custom_id
        except Exception:
            pass

        try:
            request = row.get("request", {})
            custom_id = request.get("custom_id")
            if isinstance(custom_id, str) and custom_id.strip():
                return custom_id
        except Exception:
            pass

        return ""

    @staticmethod
    def _check_anthropic_config(config: BatchProviderConfig) -> AnthropicBatchConfig:
        if isinstance(config, AnthropicBatchConfig):
            return config
        raise TypeError("AnthropicBatchAdapter requires AnthropicBatchConfig")

    @staticmethod
    def _create_client(config: AnthropicBatchConfig) -> Anthropic:
        api_key = config.credentials.get("api_key", "").strip() or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        ).strip()
        if not api_key:
            raise ValueError(
                "Anthropic API key is missing. Provide credentials.api_key or set ANTHROPIC_API_KEY."
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        if config.timeout_seconds is not None:
            client_kwargs["timeout"] = config.timeout_seconds
        return Anthropic(**client_kwargs)

    @staticmethod
    def _resolve_batches_client(client: Anthropic) -> Any:
        if hasattr(client, "messages") and hasattr(client.messages, "batches"):
            return client.messages.batches
        if hasattr(client, "beta") and hasattr(client.beta, "messages"):
            return client.beta.messages.batches
        raise AttributeError("Anthropic client does not expose messages.batches")

    @staticmethod
    def _attr_or_get(obj: Any, attr: str, default: Any = None) -> Any:
        try:
            val = getattr(obj, attr)
        except Exception:
            val = None
        if val is not None:
            return val
        if isinstance(obj, Mapping):
            return obj.get(attr, default)
        return default
