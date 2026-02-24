"""
Draft Generation Node
=====================
Generates a professional, patient-facing draft response using the LLM,
based on the intent and the structured result stored in ``action_result``.

The draft is intentionally labelled as *unreviewed* until it passes through
the Human-in-the-Loop (HITL) node that follows.

Middleware applied
------------------
* ModelCallLimitMiddleware – enforces LLM call budget
* ModelRetryMiddleware     – handles transient failures
* ModelFallbackMiddleware  – falls back to secondary model if needed

State fields updated
--------------------
draft_response, model_call_count, route_path, messages
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.middleware.fallback import ModelFallbackMiddleware
from src.middleware.limits import ModelCallLimitMiddleware
from src.middleware.retry import ModelRetryMiddleware

_limiter = ModelCallLimitMiddleware(max_calls=10)
_retry   = ModelRetryMiddleware(max_retries=3, base_delay=1.0)
_llm     = ModelFallbackMiddleware()

_SYSTEM_PROMPT = """\
You are a professional patient communication specialist at Halifax Medical Centre.
Write a clear, empathetic, and concise response to a patient based on the
appointment action that was just completed.

Guidelines:
- Use a warm but professional tone.
- Address the patient by name if available.
- Confirm the key details of the action taken (date, type, doctor).
- For prep instructions: present them as a numbered list.
- Do NOT offer clinical advice or medical opinions.
- Keep the response under 150 words.
- End with an invitation to contact the scheduling office if they have questions.
"""


def _build_user_prompt(state: dict) -> str:
    """Construct the LLM user-turn prompt from workflow state."""
    intent = state.get("intent", "unknown")
    result = state.get("action_result") or {}
    patient = state.get("patient_name") or result.get("patient_name", "")

    if intent == "reschedule":
        return (
            f"Patient name: {patient or 'the patient'}\n"
            f"Action: Reschedule appointment\n"
            f"Details: {json.dumps(result, indent=2)}\n\n"
            "Write a confirmation response for the reschedule."
        )
    elif intent == "cancel":
        return (
            f"Patient name: {patient or 'the patient'}\n"
            f"Action: Cancel appointment\n"
            f"Details: {json.dumps(result, indent=2)}\n\n"
            "Write a confirmation response for the cancellation."
        )
    elif intent == "prep_instructions":
        instructions = result.get("instructions", [])
        apt_type = result.get("appointment_type", state.get("appointment_type", "your appointment"))
        return (
            f"Patient name: {patient or 'the patient'}\n"
            f"Action: Provide preparation instructions\n"
            f"Procedure: {apt_type}\n"
            f"Instructions: {json.dumps(instructions, indent=2)}\n\n"
            "Write a patient-friendly message presenting these instructions."
        )
    else:
        return (
            f"Patient name: {patient or 'the patient'}\n"
            f"Intent: {intent}\n"
            f"Details: {json.dumps(result, indent=2)}\n\n"
            "Write an appropriate response for this appointment request."
        )


def draft_generation_node(state: dict) -> dict:
    """LangGraph node: generate a draft patient response using the LLM."""
    _limiter.check(state)

    user_prompt = _build_user_prompt(state)

    @_retry.wrap
    def call_llm():
        return _llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

    response = call_llm()
    draft = response.content.strip()

    print(f"  [DraftGeneration] Draft generated ({len(draft)} chars).")

    return {
        "draft_response": draft,
        "route_path":     ["draft_generation"],
        **_limiter.increment(state),
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": f"[DRAFT] {draft}"},
        ],
    }
