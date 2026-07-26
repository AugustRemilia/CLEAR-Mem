"""Live LLM adapter primitives with retry, parse, and cost controls."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LiveLLMError(RuntimeError):
    pass


class BudgetExceededError(LiveLLMError):
    pass


class ResponseSchemaError(LiveLLMError):
    pass


class HTTPResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status_code: int
    body: dict[str, Any]


class JSONTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> HTTPResponse:
        ...


class UrllibJSONTransport:
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> HTTPResponse:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                return HTTPResponse(status_code=response.status, body=json.loads(raw))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"error": raw}
            return HTTPResponse(status_code=exc.code, body=body)


class LiveLLMConfig(BaseModel):
    """Configuration for one tiny live smoke run."""

    model_config = ConfigDict(extra="forbid")

    adapter_id: str = "openai_compat"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str = "CLEAR_LLM_API_KEY"
    api_key: str | None = Field(default=None, exclude=True)
    auth_scheme: Literal["bearer", "api_key"] = "bearer"
    completion_token_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    response_format_json: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=1, ge=0)
    retry_backoff_seconds: float = Field(default=0.2, ge=0.0)
    max_calls: int = Field(default=2, ge=0)
    max_retry_calls: int = Field(default=1, ge=0)
    max_requests_per_minute: int | None = Field(default=None, ge=1)
    min_request_interval_seconds: float = Field(default=0.0, ge=0.0)
    max_prompt_tokens_per_call: int = Field(default=2000, ge=1)
    max_completion_tokens: int = Field(default=256, ge=1)
    max_total_tokens: int = Field(default=5000, ge=1)
    max_estimated_cost_usd: float = Field(default=0.25, ge=0.0)
    input_cost_per_million_tokens: float = Field(default=0.14, ge=0.0)
    output_cost_per_million_tokens: float = Field(default=0.28, ge=0.0)

    @model_validator(mode="after")
    def _validate_retry_budget(self) -> "LiveLLMConfig":
        if self.max_retry_calls < self.max_retries:
            raise ValueError("max_retry_calls must be >= max_retries")
        return self

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def resolved_api_key(self) -> str:
        return self.resolved_api_keys()[0]

    def resolved_api_keys(self) -> list[str]:
        if self.api_key:
            return [self.api_key]
        keys = _resolve_api_keys_from_env(self.api_key_env)
        if not keys:
            plural_env = f"{self.api_key_env}S"
            raise LiveLLMError(
                "missing API key; set "
                f"{self.api_key_env}, {plural_env}, {self.api_key_env}_1, "
                f"or config.api_key"
            )
        return keys


class LiveLLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    case_id: str
    attempt: int
    is_retry: bool
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    adapter_id: str
    model: str
    credential_slot: int | None = None
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    prompt_tokens_estimate: int
    max_completion_tokens: int
    prompt_tokens_reported: int | None = None
    completion_tokens_reported: int | None = None
    reserved_cost_usd: float
    input_hash: str
    attempt_hash: str
    request_hash: str
    response_hash: str | None = None

    def to_jsonl(self) -> str:
        return self.model_dump_json() + "\n"


class LiveLLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    content: str
    parsed_json: dict[str, Any]
    call_record: LiveLLMCallRecord


class LiveLLMBudget:
    def __init__(self, config: LiveLLMConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self.calls = 0
        self.retry_calls = 0
        self.reserved_tokens = 0
        self.reserved_cost_usd = 0.0

    def reserve(self, prompt_tokens_estimate: int, max_completion_tokens: int, is_retry: bool) -> float:
        with self._lock:
            if self.calls + 1 > self.config.max_calls:
                raise BudgetExceededError("max_calls exceeded before request")
            if is_retry and self.retry_calls + 1 > self.config.max_retry_calls:
                raise BudgetExceededError("max_retry_calls exceeded before request")
            tokens = prompt_tokens_estimate + max_completion_tokens
            if self.reserved_tokens + tokens > self.config.max_total_tokens:
                raise BudgetExceededError("max_total_tokens exceeded before request")
            cost = _estimate_cost_usd(prompt_tokens_estimate, max_completion_tokens, self.config)
            if self.reserved_cost_usd + cost > self.config.max_estimated_cost_usd:
                raise BudgetExceededError("max_estimated_cost_usd exceeded before request")
            self.calls += 1
            if is_retry:
                self.retry_calls += 1
            self.reserved_tokens += tokens
            self.reserved_cost_usd += cost
        return cost


class _RequestsPerMinuteLimiter:
    def __init__(
        self,
        max_requests_per_minute: int | None,
        min_request_interval_seconds: float,
        sleep_fn: Callable[[float], None],
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.min_request_interval_seconds = min_request_interval_seconds
        self.sleep_fn = sleep_fn
        self.clock_fn = clock_fn
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self.max_requests_per_minute is None and self.min_request_interval_seconds <= 0:
            return
        while True:
            with self._lock:
                now = self.clock_fn()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                interval_wait = max(0.0, self._next_allowed_at - now)
                rpm_wait = 0.0
                if (
                    self.max_requests_per_minute is not None
                    and len(self._timestamps) >= self.max_requests_per_minute
                ):
                    rpm_wait = max(0.0, 60.0 - (now - self._timestamps[0]))
                wait_seconds = max(interval_wait, rpm_wait)
                if wait_seconds <= 0.0:
                    self._timestamps.append(now)
                    self._next_allowed_at = now + self.min_request_interval_seconds
                    return
            self.sleep_fn(wait_seconds)


class OpenAICompatChatAdapter:
    """Minimal OpenAI-compatible chat-completions adapter.

    The adapter expects the model to return a JSON object in the message content.
    It records every attempt, including parse failures and retryable HTTP errors.
    """

    def __init__(
        self,
        config: LiveLLMConfig,
        transport: JSONTransport | None = None,
        sleep_fn=time.sleep,
        record_sink: Callable[[LiveLLMCallRecord], None] | None = None,
    ) -> None:
        self.config = config
        self.api_keys = config.resolved_api_keys()
        self.transport = transport or UrllibJSONTransport()
        self.sleep_fn = sleep_fn
        self.record_sink = record_sink
        self._records_lock = threading.Lock()
        self._credential_lock = threading.Lock()
        self._next_credential_slot = 0
        self.records: list[LiveLLMCallRecord] = []
        self.budget = LiveLLMBudget(config)
        self.request_limiters = [
            _RequestsPerMinuteLimiter(
                config.max_requests_per_minute,
                config.min_request_interval_seconds,
                sleep_fn=sleep_fn,
            )
            for _ in self.api_keys
        ]

    def complete_json(
        self,
        *,
        run_id: str,
        case_id: str,
        messages: list[dict[str, str]],
    ) -> LiveLLMResponse:
        prompt_tokens_estimate = _estimate_prompt_tokens(messages)
        input_hash = _hash_obj({"model": self.config.model, "messages": messages})
        if prompt_tokens_estimate > self.config.max_prompt_tokens_per_call:
            attempt_hash = _hash_obj({"input_hash": input_hash, "attempt": 0})
            record = self._record(
                run_id=run_id,
                case_id=case_id,
                attempt=0,
                is_retry=False,
                status="budget_exceeded",
                prompt_tokens_estimate=prompt_tokens_estimate,
                reserved_cost_usd=0.0,
                input_hash=input_hash,
                attempt_hash=attempt_hash,
                error_type="prompt_too_large",
                error_message="prompt token estimate exceeds max_prompt_tokens_per_call",
            )
            self._append_record(record)
            raise BudgetExceededError(record.error_message or "prompt too large")

        last_error: LiveLLMError | None = None
        for attempt in range(self.config.max_retries + 1):
            is_retry = attempt > 0
            attempt_hash = _hash_obj({"input_hash": input_hash, "attempt": attempt})
            try:
                reserved_cost = self.budget.reserve(
                    prompt_tokens_estimate,
                    self.config.max_completion_tokens,
                    is_retry=is_retry,
                )
            except BudgetExceededError as exc:
                record = self._record(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    is_retry=is_retry,
                    status="budget_exceeded",
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    reserved_cost_usd=0.0,
                    input_hash=input_hash,
                    attempt_hash=attempt_hash,
                    error_type="budget_exceeded",
                    error_message=str(exc),
                )
                self._append_record(record)
                raise

            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": 0,
                self.config.completion_token_field: self.config.max_completion_tokens,
            }
            payload.update(self.config.extra_body)
            if self.config.response_format_json:
                payload["response_format"] = {"type": "json_object"}
            credential_slot, api_key = self._next_credential()
            try:
                self.request_limiters[credential_slot].wait()
                response = self.transport.post_json(
                    self.config.chat_completions_url,
                    headers=self._request_headers(api_key),
                    payload=payload,
                    timeout_seconds=self.config.timeout_seconds,
                )
            except Exception as exc:  # pragma: no cover - defensive around live network stack
                record = self._record(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    is_retry=is_retry,
                    status="transport_error",
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    reserved_cost_usd=reserved_cost,
                    input_hash=input_hash,
                    attempt_hash=attempt_hash,
                    credential_slot=credential_slot,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                self._append_record(record)
                last_error = LiveLLMError(str(exc))
                if attempt < self.config.max_retries:
                    self.sleep_fn(self.config.retry_backoff_seconds)
                    continue
                raise last_error

            usage = response.body.get("usage") if isinstance(response.body, dict) else {}
            prompt_reported = _int_or_none((usage or {}).get("prompt_tokens"))
            completion_reported = _int_or_none((usage or {}).get("completion_tokens"))
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                record = self._record(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    is_retry=is_retry,
                    status="retryable_error",
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    reserved_cost_usd=reserved_cost,
                    input_hash=input_hash,
                    attempt_hash=attempt_hash,
                    credential_slot=credential_slot,
                    status_code=response.status_code,
                    response_hash=_hash_obj(response.body),
                    prompt_tokens_reported=prompt_reported,
                    completion_tokens_reported=completion_reported,
                    error_type="http_retryable",
                    error_message=str(response.body.get("error", response.body)),
                )
                self._append_record(record)
                last_error = LiveLLMError(record.error_message or "retryable error")
                if attempt < self.config.max_retries:
                    self.sleep_fn(self.config.retry_backoff_seconds)
                    continue
                raise last_error
            if response.status_code >= 400:
                record = self._record(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    is_retry=is_retry,
                    status="non_retryable_error",
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    reserved_cost_usd=reserved_cost,
                    input_hash=input_hash,
                    attempt_hash=attempt_hash,
                    credential_slot=credential_slot,
                    status_code=response.status_code,
                    response_hash=_hash_obj(response.body),
                    prompt_tokens_reported=prompt_reported,
                    completion_tokens_reported=completion_reported,
                    error_type="http_non_retryable",
                    error_message=str(response.body.get("error", response.body)),
                )
                self._append_record(record)
                raise LiveLLMError(record.error_message or "non-retryable error")

            content = _extract_message_content(response.body)
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                record = self._record(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    is_retry=is_retry,
                    status="parse_error",
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    reserved_cost_usd=reserved_cost,
                    input_hash=input_hash,
                    attempt_hash=attempt_hash,
                    credential_slot=credential_slot,
                    status_code=response.status_code,
                    response_hash=_hash_obj(response.body),
                    prompt_tokens_reported=prompt_reported,
                    completion_tokens_reported=completion_reported,
                    error_type="json_parse_error",
                    error_message=str(exc),
                )
                self._append_record(record)
                last_error = LiveLLMError(str(exc))
                if attempt < self.config.max_retries:
                    self.sleep_fn(self.config.retry_backoff_seconds)
                    continue
                raise last_error
            if not isinstance(parsed, dict):
                record = self._record(
                    run_id=run_id,
                    case_id=case_id,
                    attempt=attempt,
                    is_retry=is_retry,
                    status="schema_error",
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    reserved_cost_usd=reserved_cost,
                    input_hash=input_hash,
                    attempt_hash=attempt_hash,
                    credential_slot=credential_slot,
                    status_code=response.status_code,
                    response_hash=_hash_obj(response.body),
                    prompt_tokens_reported=prompt_reported,
                    completion_tokens_reported=completion_reported,
                    error_type="json_object_required",
                    error_message="response JSON must be an object",
                )
                self._append_record(record)
                last_error = ResponseSchemaError(record.error_message or "schema error")
                if attempt < self.config.max_retries:
                    self.sleep_fn(self.config.retry_backoff_seconds)
                    continue
                raise last_error

            record = self._record(
                run_id=run_id,
                case_id=case_id,
                attempt=attempt,
                is_retry=is_retry,
                status="success",
                prompt_tokens_estimate=prompt_tokens_estimate,
                reserved_cost_usd=reserved_cost,
                input_hash=input_hash,
                attempt_hash=attempt_hash,
                credential_slot=credential_slot,
                status_code=response.status_code,
                response_hash=_hash_obj(response.body),
                prompt_tokens_reported=prompt_reported,
                completion_tokens_reported=completion_reported,
            )
            self._append_record(record)
            return LiveLLMResponse(
                run_id=run_id,
                case_id=case_id,
                content=content,
                parsed_json=parsed,
                call_record=record,
            )

        raise last_error or LiveLLMError("live LLM call failed")

    def _next_credential(self) -> tuple[int, str]:
        with self._credential_lock:
            slot = self._next_credential_slot
            self._next_credential_slot = (self._next_credential_slot + 1) % len(self.api_keys)
        return slot, self.api_keys[slot]

    def _request_headers(self, api_key: str) -> dict[str, str]:
        if self.config.auth_scheme == "api_key":
            return {
                "api-key": api_key,
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _record(
        self,
        *,
        run_id: str,
        case_id: str,
        attempt: int,
        is_retry: bool,
        status: str,
        prompt_tokens_estimate: int,
        reserved_cost_usd: float,
        input_hash: str,
        attempt_hash: str,
        status_code: int | None = None,
        response_hash: str | None = None,
        credential_slot: int | None = None,
        prompt_tokens_reported: int | None = None,
        completion_tokens_reported: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> LiveLLMCallRecord:
        return LiveLLMCallRecord(
            run_id=run_id,
            case_id=case_id,
            attempt=attempt,
            is_retry=is_retry,
            status=status,
            adapter_id=self.config.adapter_id,
            model=self.config.model,
            credential_slot=credential_slot,
            status_code=status_code,
            error_type=error_type,
            error_message=error_message,
            prompt_tokens_estimate=prompt_tokens_estimate,
            max_completion_tokens=self.config.max_completion_tokens,
            prompt_tokens_reported=prompt_tokens_reported,
            completion_tokens_reported=completion_tokens_reported,
            reserved_cost_usd=reserved_cost_usd,
            input_hash=input_hash,
            attempt_hash=attempt_hash,
            request_hash=attempt_hash,
            response_hash=response_hash,
        )

    def _append_record(self, record: LiveLLMCallRecord) -> None:
        with self._records_lock:
            self.records.append(record)
        if self.record_sink is not None:
            self.record_sink(record)


def _extract_message_content(body: dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LiveLLMError("response missing choices[0].message.content") from exc
    if not isinstance(content, str):
        raise LiveLLMError("response content is not a string")
    return content


def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    raw = json.dumps(messages, ensure_ascii=False)
    return max(1, len(raw) // 4)


def _estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    config: LiveLLMConfig,
) -> float:
    return (
        prompt_tokens * config.input_cost_per_million_tokens
        + completion_tokens * config.output_cost_per_million_tokens
    ) / 1_000_000


def _hash_obj(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _resolve_api_keys_from_env(api_key_env: str) -> list[str]:
    keys: list[str] = []
    for env_name in (f"{api_key_env}S", "CLEAR_LLM_API_KEYS"):
        raw = os.getenv(env_name)
        if raw:
            keys.extend(_split_api_key_list(raw))
    numbered_names = [f"{api_key_env}_{index}" for index in range(1, 17)]
    if api_key_env == "CLEAR_LLM_API_KEY":
        numbered_names.extend(f"CLEAR_LLM_API_KEY_{index}" for index in range(1, 17))
    for env_name in numbered_names:
        raw = os.getenv(env_name)
        if raw:
            keys.extend(_split_api_key_list(raw))
    raw_single = os.getenv(api_key_env)
    if raw_single:
        keys.extend(_split_api_key_list(raw_single))
    return _dedupe_preserve_order(keys)


def _split_api_key_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
