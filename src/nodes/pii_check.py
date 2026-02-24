"""
PII Check Node
==============
First node in the LangGraph workflow.  Delegates entirely to
``PIIMiddleware`` to scan and mask the patient's raw message.

Why first?
----------
Running PII masking before any LLM call or log write guarantees that
sensitive identifiers (phone numbers, SSNs, dates of birth, etc.) are
never passed to external APIs unmasked or written to trace output.
"""

from src.middleware.pii import PIIMiddleware

_pii = PIIMiddleware()


def pii_check_node(state: dict) -> dict:
    """LangGraph node: run PIIMiddleware on the incoming patient message."""
    return _pii.run_on_state(state)
