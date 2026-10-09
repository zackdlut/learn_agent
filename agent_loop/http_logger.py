"""Log full HTTP requests and responses to the file named by LOG_FILE.

LOG_FILE comes from the environment (this repo's .env). A relative path is
resolved from the directory that contains .env.

JSON bodies are parsed and re-serialized with indentation and ensure_ascii=False.
Nested JSON strings (e.g. tool call ``arguments``) are expanded so the log stays
readable without escaped inner JSON.

Sync hooks go on httpx.Client. Async hooks go on httpx.AsyncClient.
ChatOpenAI accepts both via http_client and http_async_client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from .env import env_file, ensure_env_loaded

_LOGGER_NAME = "agent_loop.http"
_logger: logging.Logger | None = None


def log_path() -> Path:
    """Return the log file path from LOG_FILE."""
    ensure_env_loaded()
    dotenv = env_file()
    raw = os.getenv("LOG_FILE")
    if not raw:
        raise RuntimeError("LOG_FILE is not set. Add it to .env.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        base = dotenv.parent if dotenv is not None else Path.cwd()
        path = base / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_logger() -> logging.Logger:
    """Return the process-wide logger that writes to LOG_FILE."""
    global _logger
    if _logger is not None:
        return _logger

    path = log_path()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(_file_handler(path))
    _logger = logger
    return logger


def log(message: str) -> None:
    """Write one log record from synchronous code."""
    get_logger().info("%s", message)


def log_section(title: str) -> None:
    """Write a visible section header."""
    bar = "─" * 56
    log(f"\n{bar}\n  {title}\n{bar}")


def clear_log() -> None:
    """Delete LOG_FILE so the next records start a new file."""
    global _logger
    if _logger is not None:
        for handler in list(_logger.handlers):
            handler.close()
            _logger.removeHandler(handler)
        _logger = None
    log_path().unlink(missing_ok=True)


async def alog(message: str) -> None:
    """Write one log record without blocking the event loop on the file write."""
    await asyncio.to_thread(log, message)


def make_sync_client(**kwargs: Any) -> httpx.Client:
    """httpx client whose requests and responses are written to LOG_FILE."""
    return httpx.Client(**_client_kwargs(kwargs, sync_event_hooks()))


def make_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """Async httpx client whose requests and responses are written to LOG_FILE."""
    return httpx.AsyncClient(**_client_kwargs(kwargs, async_event_hooks()))


def sync_event_hooks() -> dict[str, list[Any]]:
    return {"request": [_log_request], "response": [_log_response]}


def async_event_hooks() -> dict[str, list[Any]]:
    return {"request": [_alog_request], "response": [_alog_response]}


def _client_kwargs(
    kwargs: dict[str, Any], hooks: dict[str, list[Any]]
) -> dict[str, Any]:
    extra = kwargs.pop("event_hooks", None)
    kwargs.setdefault("trust_env", True)
    kwargs["event_hooks"] = _merge_hooks(hooks, extra)
    return kwargs


def _merge_hooks(
    base: dict[str, list[Any]], extra: dict[str, list[Any]] | None
) -> dict[str, list[Any]]:
    merged = {key: list(fns) for key, fns in base.items()}
    for key, fns in (extra or {}).items():
        merged.setdefault(key, []).extend(fns)
    return merged


def _file_handler(path: Path) -> logging.FileHandler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_Formatter())
    return handler


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        text = record.getMessage()
        if "\n" not in text:
            return f"[{timestamp}] {text}"
        return f"[{timestamp}]\n{text}\n"


def _log_request(request: httpx.Request) -> None:
    log(_format_request(request))


def _log_response(response: httpx.Response) -> None:
    body = _read_response_body(response)
    log(_format_response(response, body))


async def _alog_request(request: httpx.Request) -> None:
    await alog(_format_request(request))


async def _alog_response(response: httpx.Response) -> None:
    body = await _aread_response_body(response)
    await alog(_format_response(response, body))


def _read_response_body(response: httpx.Response) -> str:
    if _is_event_stream(response):
        return "(streamed body omitted)"
    response.read()
    return response.text


async def _aread_response_body(response: httpx.Response) -> str:
    if _is_event_stream(response):
        return "(streamed body omitted)"
    await response.aread()
    return response.text


def _is_event_stream(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "")
    return content_type.startswith("text/event-stream")


def _format_request(request: httpx.Request) -> str:
    raw = request.content.decode("utf-8", errors="replace")
    body = _pretty_body(raw, request.headers.get("content-type", ""))
    return _format_exchange(f">>> {request.method} {request.url}", request.headers, body)


def _format_response(response: httpx.Response, body: str) -> str:
    if body != "(streamed body omitted)":
        body = _pretty_body(body, response.headers.get("content-type", ""))
    title = f"<<< {response.status_code} {response.request.url}"
    return _format_exchange(title, response.headers, body)


def _format_exchange(title: str, headers: httpx.Headers, body: str) -> str:
    parts = [title, "", _format_headers(headers)]
    if body:
        parts.extend(["", body])
    return "\n".join(parts)


def _format_headers(headers: httpx.Headers) -> str:
    return "\n".join(f"  {name}: {value}" for name, value in headers.items())


def _pretty_body(body: str, content_type: str) -> str:
    text = body.strip()
    if not text:
        return ""
    if not _is_json_content(content_type) and text[0] not in "{[":
        return body
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return body
    expanded = _expand_embedded_json_strings(parsed)
    return json.dumps(expanded, indent=2, ensure_ascii=False)


def _expand_embedded_json_strings(value: Any) -> Any:
    """Turn JSON-encoded strings (e.g. tool arguments) into nested objects."""
    if isinstance(value, dict):
        return {key: _expand_embedded_json_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_embedded_json_strings(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "{[":
            try:
                inner = json.loads(stripped)
            except json.JSONDecodeError:
                return value
            return _expand_embedded_json_strings(inner)
    return value


def _is_json_content(content_type: str) -> bool:
    media = content_type.split(";", 1)[0].strip().lower()
    return media == "application/json" or media.endswith("+json")
