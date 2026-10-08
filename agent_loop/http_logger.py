"""Log full HTTP requests and responses to the file named by LOG_FILE.

LOG_FILE comes from the environment (this repo's .env). A relative path is
resolved from the directory that contains .env.

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
from dotenv import load_dotenv

_LOGGER_NAME = "agent_loop.http"
_logger: logging.Logger | None = None


def log_path() -> Path:
    """Return the log file path from LOG_FILE."""
    env_file = _find_env_file()
    if env_file is not None:
        load_dotenv(env_file, override=False)
    raw = os.getenv("LOG_FILE")
    if not raw:
        raise RuntimeError("LOG_FILE is not set. Add it to .env.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        base = env_file.parent if env_file is not None else Path.cwd()
        path = base / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _find_env_file() -> Path | None:
    start = Path(__file__).resolve().parent
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


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
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _is_json_content(content_type: str) -> bool:
    media = content_type.split(";", 1)[0].strip().lower()
    return media == "application/json" or media.endswith("+json")
