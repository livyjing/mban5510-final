"""
Intent Detection Node
=====================
Classifies the patient's message into one of five intents and extracts
relevant appointment entities using an LLM (structured JSON output).

Middleware applied
------------------
* ModelCallLimitMiddleware – pre-flight check before every LLM call
* ModelRetryMiddleware     – wraps the LLM call with exponential back-off
* ModelFallbackMiddleware  – switches to a secondary model on persistent failure

State fields updated
--------------------
intent, appointment_id, appointment_type, patient_name, preferred_date,
model_call_count, route_path, messages
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.middleware.fallback import ModelFallbackMiddleware
from src.middleware.limits import ModelCallLimitMiddleware
from src.middleware.retry import ModelRetryMiddleware

# ---------------------------------------------------------------------------
# Middleware instances (shared across calls within the same process)
# ---------------------------------------------------------------------------
_limiter = ModelCallLimitMiddleware(max_calls=10)
_retry   = ModelRetryMiddleware(max_retries=3, base_delay=1.0)
_llm     = ModelFallbackMiddleware()        # gpt-4o-mini → gpt-3.5-turbo fallback

# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an intent classifier for a medical appointment assistance system.

━━━ STEP 1: CLASSIFY INTENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classify the patient's message into EXACTLY ONE of these intents:

  reschedule         – patient wants to change the date/time of an existing appointment
                       (use this even if they do not specify which appointment or new date)
  cancel             – patient wants to cancel an existing appointment
                       (use this even if they do not specify which appointment)
  prep_instructions  – patient wants preparation instructions for an upcoming procedure
  escalate           – message indicates a MEDICAL EMERGENCY (e.g. chest pain, difficulty
                       breathing, stroke, suicidal ideation, severe injury, overdose).
                       Do NOT use this for clinical advice questions — those are "unknown".
  unknown            – any message that does not fit the above, including:
                       • general questions about the clinic (hours, location, parking)
                       • requests for clinical advice, diagnoses, or medication guidance
                       • questions about test results or symptoms (non-emergency)
                       • anything else outside appointment management scope
                       Missing details (appointment ID, date) are NOT a reason for unknown.

INTENT EXAMPLES:
  "I want to change my appointment."         → reschedule  (details missing, but intent clear)
  "Can I move my appointment?"               → reschedule
  "Cancel my appointment please."            → cancel      (details missing, but intent clear)
  "What do I need to prepare for tomorrow?"  → prep_instructions
  "I have chest pain."                       → escalate
  "My friend was asking about hospitals."    → unknown     (no appointment action implied)

━━━ STEP 2: EXTRACT ENTITIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extract the following fields ONLY when explicitly stated in the message:
  appointment_id    – appointment group code (e.g. APT-001, APT-002)
  item_id           – specific item code within an appointment (e.g. APT-001-A, APT-002-B)
  appointment_type  – procedure type (e.g. MRI, CT scan, blood work, ultrasound)
  patient_name      – patient's name if mentioned
  preferred_date    – preferred new date for a reschedule request

ENTITY EXTRACTION RULES (strictly enforced):
  - Return null for any field not explicitly stated — do NOT infer or guess.
  - Do NOT set appointment_type to generic words like "appointment" or "general".
  - Do NOT set preferred_date to vague terms like "TBD", "soon", "ASAP",
    "at your convenience". Only extract a real date/time reference
    (e.g., "March 25", "next Tuesday", "early April").
  - "I want to change my appointment" → all entity fields are null (no details given).
    The system will ask the patient for the missing details downstream.

Notes on appointment_id vs item_id:
  - "Cancel my MRI"     → appointment_type="MRI"   (procedure name, NOT an ID code)
  - "Cancel APT-001-A"  → item_id="APT-001-A"      (explicit item code)
  - "Cancel APT-001"    → appointment_id="APT-001"  (explicit group code)
  - appointment_id and item_id are NEVER inferred from the patient's name or the
    [Patient: ...] prefix. They MUST appear as explicit APT-XXX codes in the message.

ESCALATE only when the message contains a MEDICAL EMERGENCY:
  • Chest pain, difficulty breathing, stroke symptoms
  • Suicidal ideation or self-harm intent
  • Severe injury, overdose, anaphylaxis, or unconsciousness
  • Any situation requiring immediate emergency services

Do NOT escalate for:
  • Requests for clinical advice or diagnoses ("do I have…", "what medication…")
  • Questions about test results or what symptoms mean
  → These are "unknown" intent and answered by the general assistant.

Respond with VALID JSON ONLY – no prose, no markdown fences:
{
  "intent": "<intent>",
  "appointment_id": "<id or null>",
  "item_id": "<item id or null>",
  "appointment_type": "<type or null>",
  "patient_name": "<name or null>",
  "preferred_date": "<date or null>",
  "confidence": "high | medium | low",
  "reasoning": "<one-sentence explanation>"
}
"""


def _parse_json(raw: str) -> dict:
    """Extract a JSON object from an LLM response (handles markdown fences)."""
    text = raw.strip()
    # Strip optional ```json … ``` wrapper
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening ``` line and closing ``` line
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def intent_detection_node(state: dict) -> dict:
    """
    LangGraph node: classify patient intent via LLM and extract entities.

    Uses the PII-masked message for the LLM call to avoid leaking raw PII
    to the model or its logs. Falls back to raw message if masking hasn't run.
    """
    # Guard: enforce call limit before incurring any cost
    _limiter.check(state)

    # Prefer PII-masked version; fall back to raw
    message = state.get("user_message_masked") or state.get("user_message", "")

    # Short-circuit: moderation middleware may have already set intent=escalate
    if state.get("intent") == "escalate":
        print("  [IntentDetection] Moderation already flagged escalation – skipping LLM.")
        return {
            "route_path": ["intent_detect_skip"],
            **_limiter.increment(state),
        }

    @_retry.wrap
    def call_llm():
        return _llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ])

    response = call_llm()

    try:
        result = _parse_json(response.content)
    except (json.JSONDecodeError, ValueError):
        result = {
            "intent": "unknown",
            "appointment_id": None,
            "item_id": None,
            "appointment_type": None,
            "patient_name": None,
            "preferred_date": None,
        }

    intent = result.get("intent", "unknown")
    print(
        f"  [IntentDetection] intent={intent!r}  "
        f"type={result.get('appointment_type')!r}  "
        f"id={result.get('appointment_id')!r}  "
        f"item_id={result.get('item_id')!r}  "
        f"confidence={result.get('confidence', '?')}"
    )

    return {
        "intent":           intent,
        "appointment_id":   result.get("appointment_id"),
        "item_id":          result.get("item_id"),
        "appointment_type": result.get("appointment_type"),
        "patient_name":     result.get("patient_name"),
        "preferred_date":   result.get("preferred_date"),
        "route_path":       ["intent_detect"],
        **_limiter.increment(state),
        "messages": state.get("messages", []) + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": f"Intent: {intent}"},
        ],
    }
