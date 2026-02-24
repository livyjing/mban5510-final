"""
LimitMiddleware
===============
Enforces per-run hard caps on LLM and tool invocations to prevent runaway
execution loops and uncontrolled API cost.

Components
----------
ModelCallLimitMiddleware
    Raises ``RuntimeError`` if ``state["model_call_count"]`` reaches the
    configured ceiling before a new LLM call is attempted.  Each successful
    call should be followed by ``increment()`` to update the counter.

ToolCallLimitMiddleware
    Same semantics for ``state["tool_call_count"]``.

Design note
-----------
The limit checks are deliberately placed at the *start* of each node that
makes an LLM or tool call, not after.  This ensures the cap is enforced
before any latency or cost is incurred.
"""


class ModelCallLimitMiddleware:
    """Prevent excessive LLM invocations within a single workflow run."""

    name = "ModelCallLimitMiddleware"

    def __init__(self, max_calls: int = 10):
        self.max_calls = max_calls

    def check(self, state: dict) -> None:
        """Raise RuntimeError if the model call limit has been reached."""
        count = state.get("model_call_count", 0)
        if count >= self.max_calls:
            raise RuntimeError(
                f"[{self.name}] Model call limit ({self.max_calls}) reached "
                f"(current count: {count}). Workflow halted."
            )

    def increment(self, state: dict) -> dict:
        """Return a partial state update that increments the counter by 1."""
        return {"model_call_count": state.get("model_call_count", 0) + 1}


class ToolCallLimitMiddleware:
    """Prevent excessive tool / data-function invocations within a single run."""

    name = "ToolCallLimitMiddleware"

    def __init__(self, max_calls: int = 20):
        self.max_calls = max_calls

    def check(self, state: dict) -> None:
        """Raise RuntimeError if the tool call limit has been reached."""
        count = state.get("tool_call_count", 0)
        if count >= self.max_calls:
            raise RuntimeError(
                f"[{self.name}] Tool call limit ({self.max_calls}) reached "
                f"(current count: {count}). Workflow halted."
            )

    def increment(self, state: dict) -> dict:
        """Return a partial state update that increments the counter by 1."""
        return {"tool_call_count": state.get("tool_call_count", 0) + 1}
