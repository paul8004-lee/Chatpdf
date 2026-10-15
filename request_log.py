"""Append-only request diagnostics; no prompts, credentials, or document bodies."""
import json
import logging
import re
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler


_current = ContextVar("request_log", default=None)
_write_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def current():
    return _current.get()


class RequestLog:
    def __init__(self, path, operation, secrets=()):
        self.path = Path(path)
        self.operation = operation
        self.request_id = uuid.uuid4().hex
        self.secrets = tuple(str(s) for s in secrets if s)
        self.stage = operation
        self.status = "success"
        self.tokens = dict(input_tokens=0, output_tokens=0, total_tokens=0)
        self.usage_complete = True

    def redact(self, value):
        if isinstance(value, dict):
            return {str(k): ("[REDACTED]" if re.search(
                r"api.?key|authorization|password|secret|access.?token", str(k), re.I
            ) else self.redact(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            for secret in sorted(self.secrets, key=len, reverse=True):
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
            return re.sub(r"Bearer\s+[^\s'\"]+", "Bearer [REDACTED]", value, flags=re.I)
        return value

    def emit(self, event, **fields):
        record = self.redact(dict(
            schema_version=1, time=datetime.now(timezone.utc).isoformat(),
            request_id=self.request_id, operation=self.operation,
            stage=self.stage, event=event, **fields,
        ))
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with _write_lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
        except Exception as error:
            # Do not recursively log or expose the failed record / filesystem path.
            _logger.error("Request log write failed (%s), request_id=%s",
                          type(error).__name__, self.request_id)

    def error(self, error, **fields):
        self.emit("error", error_type=type(error).__name__, error=str(error),
                  traceback="".join(traceback.format_exception(
                      type(error), error, error.__traceback__)), **fields)

    def usage(self, model, usage=None, reason=None, stage=None):
        fields = dict(model=model, usage_status="available" if usage else "unavailable")
        if usage:
            counts = dict(input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                          output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)))
            counts["total_tokens"] = usage.get("total_tokens", sum(counts.values()))
            for key, value in counts.items():
                self.tokens[key] += value
            fields.update(counts)
        else:
            self.usage_complete = False
            fields.update(input_tokens=None, output_tokens=None, total_tokens=None, reason=reason)
        previous = self.stage
        self.stage = stage or previous
        try:
            self.emit("token_usage", **fields)
        finally:
            self.stage = previous


@contextmanager
def request(path, operation, secrets=(), activate=True):
    trace = RequestLog(path, operation, secrets)
    token = _current.set(trace) if activate else None
    started = time.perf_counter()
    trace.emit("request_started")
    try:
        yield trace
    except GeneratorExit:
        trace.status = "cancelled"
        raise
    except Exception as error:
        trace.status = "error"
        trace.error(error)
        raise
    finally:
        trace.stage = operation
        trace.emit("request_finished", status=trace.status,
                   seconds=round(time.perf_counter() - started, 4),
                   token_usage=trace.tokens,
                   usage_status="complete" if trace.usage_complete else "partial")
        if activate:
            _current.reset(token)


def iterate(trace, iterator):
    """Gradio may resume a synchronous generator in a different worker context."""
    try:
        while True:
            token = _current.set(trace)
            try:
                value = next(iterator)
            except StopIteration:
                return
            finally:
                _current.reset(token)
            yield value
    finally:
        token = _current.set(trace)
        try:
            iterator.close()
        finally:
            _current.reset(token)


@contextmanager
def stage(name):
    trace = current()
    if trace is None:
        yield
        return
    previous = trace.stage
    trace.stage = name
    started = time.perf_counter()
    try:
        yield
    except Exception as error:
        trace.error(error)
        trace.emit("stage_finished", status="error", seconds=round(time.perf_counter()-started, 4))
        raise
    else:
        trace.emit("stage_finished", status="success", seconds=round(time.perf_counter()-started, 4))
    finally:
        trace.stage = previous


class TokenUsageCallback(BaseCallbackHandler):
    """Read provider usage from completed (including streamed) chat results."""
    def __init__(self, trace, model=None):
        self.trace = trace
        self.model = model
        self.runs = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self.runs[run_id] = (self.trace.stage, (metadata or {}).get("ls_model_name") or self.model)

    def on_llm_end(self, response, *, run_id, **kwargs):
        stage_name, model = self.runs.pop(run_id, (self.trace.stage, self.model))
        output = response.llm_output or {}
        model = output.get("model_name") or model
        usage = output.get("token_usage")
        if not usage:
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None)
                    if usage:
                        model = (getattr(message, "response_metadata", {}) or {}).get("model_name") or model
                        break
                if usage:
                    break
        self.trace.usage(model, usage, "provider_usage_missing" if not usage else None, stage_name)

    def on_llm_error(self, error, *, run_id, **kwargs):
        stage_name, model = self.runs.pop(run_id, (self.trace.stage, self.model))
        self.trace.usage(model, reason="request_failed_usage_unknown", stage=stage_name)
