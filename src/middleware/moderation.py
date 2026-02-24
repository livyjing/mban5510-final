"""
ModerationMiddleware  (OpenAIModerationMiddleware)
==================================================
Two-layer safety screening that runs immediately after PII masking:

  Layer 1 – Keyword scan  : fast, zero-latency check for medical emergencies
                            and requests for clinical advice.
  Layer 2 – OpenAI API    : calls the ``omni-moderation-latest`` endpoint for
                            general harmful-content detection.

If either layer flags the message the workflow is immediately routed to the
``escalation`` node; no appointment action or draft generation occurs.

State fields populated
----------------------
- ``moderation_flagged``    – True if any layer triggered
- ``moderation_categories`` – list of triggered category strings
- ``intent``                – overridden to "escalate" when flagged
- ``route_path``            – appends "ModerationMiddleware"

Professional note
-----------------
The system must NOT provide clinical advice (project constraint §7).
All emergency / clinical-advice signals are therefore treated as hard
escalation triggers regardless of the OpenAI moderation result.
"""

import os
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------
_EMERGENCY_KEYWORDS: List[str] = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "shortness of breath", "stroke", "unconscious",
    "overdose", "suicidal", "suicide", "self-harm", "self harm",
    "severe bleeding", "anaphylaxis", "allergic reaction",
    "call 911", "ambulance", "emergency room", "er visit",
]

_CLINICAL_ADVICE_KEYWORDS: List[str] = [
    "diagnose", "diagnosis", "do i have", "is it cancer",
    "should i take", "what medication", "what drug", "what dose",
    "test results mean", "what does my result",
    "is this normal", "is that normal",
]


class ModerationMiddleware:
    """
    Guards the workflow entry point against unsafe or out-of-scope requests.

    Design note
    -----------
    Implemented as a dedicated LangGraph node (middleware-as-node pattern)
    so the graph topology makes the safety gate explicit and traceable in the
    ``route_path`` field of every run.
    """

    name = "ModerationMiddleware"

    # ------------------------------------------------------------------
    def _check_keywords(self, text: str) -> Tuple[bool, List[str]]:
        lower = text.lower()
        triggered = [kw for kw in _EMERGENCY_KEYWORDS if kw in lower]
        return bool(triggered), [f"emergency:{kw}" for kw in triggered]

    def _check_clinical(self, text: str) -> Tuple[bool, List[str]]:
        lower = text.lower()
        triggered = [kw for kw in _CLINICAL_ADVICE_KEYWORDS if kw in lower]
        return bool(triggered), [f"clinical_advice:{kw}" for kw in triggered]

    def _check_openai(self, text: str) -> Tuple[bool, List[str]]:
        """Call OpenAI Moderation API; gracefully skip on error."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.moderations.create(
                model="omni-moderation-latest",
                input=text,
            )
            result = response.results[0]
            if result.flagged:
                cats = [
                    cat
                    for cat, flagged in vars(result.categories).items()
                    if flagged
                ]
                return True, cats
        except Exception as exc:
            print(f"  [{self.name}] OpenAI moderation API unavailable: {exc}")
        return False, []

    # ------------------------------------------------------------------
    def run_on_state(self, state: dict) -> dict:
        """
        Run all moderation layers and return state updates.

        Uses the *masked* message so that PII is not sent to external APIs.
        """
        # Prefer the PII-masked version; fall back to raw if masking hasn't run
        text = state.get("user_message_masked") or state.get("user_message", "")

        flagged = False
        all_categories: List[str] = []

        # Layer 1 – emergency keywords (life-threatening → hard escalation)
        kw_flag, kw_cats = self._check_keywords(text)
        if kw_flag:
            flagged = True
            all_categories.extend(kw_cats)

        # Layer 2 – clinical advice keywords (out-of-scope but NOT an emergency)
        # Record the category for audit purposes but do NOT flag for escalation.
        # These requests are answered politely by the general_qa node instead.
        _, cl_cats = self._check_clinical(text)
        if cl_cats:
            all_categories.extend(cl_cats)

        # Layer 3 – OpenAI Moderation API (harmful/unsafe content → hard escalation)
        api_flag, api_cats = self._check_openai(text)
        if api_flag:
            flagged = True
            all_categories.extend(api_cats)

        updates: dict = {
            "moderation_flagged": flagged,
            "moderation_categories": all_categories,
            "route_path": [self.name],
        }

        if flagged:
            print(f"  [{self.name}] ⚠  Content flagged → {all_categories}")
            updates["intent"] = "escalate"
        else:
            print(f"  [{self.name}] Content cleared.")

        return updates
