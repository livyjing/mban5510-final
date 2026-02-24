"""
PIIMiddleware
=============
Detects and masks Personally Identifiable Information (PII) in patient
messages before any data touches logs or downstream nodes.

Role in the workflow
--------------------
Runs as the *first* node in the LangGraph graph so that every subsequent
node works only with the sanitised ``user_message_masked`` field in logs.
The original ``user_message`` is preserved in state for LLM inference but
is never written to trace output.

State fields populated
----------------------
- ``user_message_masked``  – safe version suitable for logging
- ``pii_detected``         – bool flag
- ``pii_types``            – categories of PII found (e.g. ["PHONE", "EMAIL"])
- ``route_path``           – appends "pii_check"
"""

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Detection patterns  (pattern, replacement_label)
# ---------------------------------------------------------------------------
_PATTERNS: List[Tuple[str, str]] = [
    # North-American phone numbers:  (xxx) xxx-xxxx  |  xxx-xxx-xxxx
    (r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}", "[PHONE]"),
    # E-mail addresses
    (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "[EMAIL]"),
    # Social Security Numbers
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    # Date-of-birth  MM/DD/YYYY  or  MM-DD-YYYY
    (r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b", "[DOB]"),
    # Generic ID / health-card numbers (6-12 consecutive digits not already matched)
    (r"\b\d{6,12}\b", "[ID_NUMBER]"),
]

_LABEL_TO_CATEGORY = {
    "[PHONE]":     "PHONE",
    "[EMAIL]":     "EMAIL",
    "[SSN]":       "SSN",
    "[DOB]":       "DOB",
    "[ID_NUMBER]": "ID_NUMBER",
}


class PIIMiddleware:
    """
    Scans patient messages for PII and replaces matches with safe placeholders.

    Design note
    -----------
    Implemented as a *middleware-style node* inside the LangGraph workflow
    rather than a transparent HTTP interceptor, because LangGraph's state
    machine model makes node-level middleware the natural integration point.
    The node runs before any LLM or tool call so that no PII leaks into
    model prompts or log files.
    """

    name = "PIIMiddleware"

    # ------------------------------------------------------------------
    def mask(self, text: str) -> Tuple[str, bool, List[str]]:
        """
        Apply all PII patterns to *text*.

        Returns
        -------
        masked_text : str
            Text with PII replaced by category labels.
        detected : bool
            True if at least one PII pattern matched.
        categories : list[str]
            De-duplicated list of matched PII category names.
        """
        masked = text
        found: List[str] = []

        for pattern, label in _PATTERNS:
            if re.search(pattern, masked):
                category = _LABEL_TO_CATEGORY[label]
                if category not in found:
                    found.append(category)
                masked = re.sub(pattern, label, masked)

        return masked, bool(found), found

    # ------------------------------------------------------------------
    def run_on_state(self, state: dict) -> dict:
        """
        Entry point called by the LangGraph node function.

        Reads ``state["user_message"]``, applies masking, and returns the
        state update dict consumed by LangGraph's reducer.
        """
        masked, detected, types = self.mask(state["user_message"])

        if detected:
            print(f"  [{self.name}] PII detected and masked → {types}")
        else:
            print(f"  [{self.name}] No PII detected.")

        return {
            "user_message_masked": masked,
            "pii_detected": detected,
            "pii_types": types,
            "route_path": [self.name],
        }
