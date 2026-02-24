"""
RetryMiddleware
===============
Provides exponential-backoff retry logic for LLM calls and tool executions.

Components
----------
ModelRetryMiddleware
    Wraps any callable LLM invocation.  On transient failure (rate-limit,
    timeout, network error) it retries with exponential back-off before
    re-raising.  Each attempt increments ``state["retry_count"]``.

ToolRetryMiddleware
    Lighter-weight retry wrapper for tool/function calls where failures
    are expected to be transient (e.g., temporary data unavailability).

Usage (decorator pattern)
-------------------------
    retry = ModelRetryMiddleware(max_retries=3)

    @retry.wrap
    def call_llm():
        return llm.invoke(messages)

    result = call_llm()
"""

import functools
import time
from typing import Callable


class ModelRetryMiddleware:
    """Retry a failed LLM call with exponential back-off."""

    name = "ModelRetryMiddleware"

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def wrap(self, fn: Callable) -> Callable:
        """Return a version of *fn* that automatically retries on exception."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(1, self.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** (attempt - 1))
                        print(
                            f"  [{ModelRetryMiddleware.name}] Attempt {attempt} failed "
                            f"({type(exc).__name__}). Retrying in {delay:.1f}s…"
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"  [{ModelRetryMiddleware.name}] All {self.max_retries} "
                            f"attempts exhausted."
                        )
            raise last_exc

        return wrapper


class ToolRetryMiddleware:
    """Retry a failed tool / data-access call."""

    name = "ToolRetryMiddleware"

    def __init__(self, max_retries: int = 2, base_delay: float = 0.5):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def wrap(self, fn: Callable) -> Callable:
        """Return a version of *fn* that retries on exception."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(1, self.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        delay = self.base_delay * attempt
                        print(
                            f"  [{ToolRetryMiddleware.name}] Tool attempt {attempt} "
                            f"failed. Retrying in {delay:.1f}s…"
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"  [{ToolRetryMiddleware.name}] All {self.max_retries} "
                            f"tool attempts failed."
                        )
            raise last_exc

        return wrapper
