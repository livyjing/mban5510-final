"""
Escalation Node
===============
Handles all risk-flagged messages by issuing an immediate, non-clinical
escalation response.

Triggered when
--------------
* ``ModerationMiddleware`` flags the message (sets intent="escalate"), OR
* ``IntentDetectionNode`` classifies intent as "escalate"

Behaviour
---------
* Sets ``status = "ESCALATE"``
* Generates a canned safety response directing the patient to emergency services
* Does NOT call the LLM (safety responses must be deterministic)
* Does NOT enter the HITL loop (risk messages must not be delayed by review)
* Appends trace labels to ``route_path``

Professional note (§7 of project spec)
---------------------------------------
"The system must not provide clinical advice; emergency handling should
direct the user to seek immediate care and/or contact appropriate services."
"""

_EMERGENCY_RESPONSE = """\
⚠  This system is not able to provide medical advice or emergency assistance.

If you are experiencing a medical emergency, please:
  • Call 911 immediately, or
  • Go to your nearest Emergency Room.

For urgent but non-emergency medical concerns, please contact:
  • Halifax Medical Centre: (902) 555-0000
  • Your family physician or a walk-in clinic.

Your safety is the priority. Please seek appropriate professional care now.\
"""

_CLINICAL_ADVICE_RESPONSE = """\
Thank you for reaching out to Halifax Medical Centre.

This automated system is designed to assist with appointment scheduling only
and is not able to provide clinical advice, interpretations of test results,
or medical guidance.

For medical questions, please:
  • Contact your referring physician directly.
  • Call our nursing advice line at (902) 555-0000 ext. 2.
  • Visit a walk-in clinic for non-emergency concerns.

We are happy to help you with appointment scheduling. Please describe your
scheduling need and we will assist you.\
"""

_STAFF_EMERGENCY_ALERT = """\
⚠️ PATIENT IN DANGER — IMMEDIATE ACTION REQUIRED ⚠️

A patient has submitted a message that may indicate a MEDICAL EMERGENCY or
serious safety concern. The automated safety response has already been sent
to the patient.

Required actions:
  1. Contact the patient immediately via phone on record.
  2. Dispatch emergency services (911 / ambulance) if warranted.
  3. Notify the on-call physician or charge nurse without delay.
  4. Document this incident in the patient's file.

Approve below to confirm that emergency protocol has been initiated,
or edit this record before confirming.\
"""

_STAFF_CLINICAL_ALERT = """\
⚠️ CLINICAL ADVICE REQUEST — STAFF REVIEW REQUIRED ⚠️

A patient has submitted a request for clinical advice or medical guidance,
which falls outside the scope of this automated scheduling system.
The automated out-of-scope response has already been sent to the patient.

Required actions:
  1. Review the patient's message in the conversation log.
  2. Forward to the appropriate clinical staff member if follow-up is needed.
  3. Confirm that the request has been routed correctly.

Approve below to acknowledge receipt, or edit this record before confirming.\
"""


def _build_patient_response(state: dict) -> str:
    """Patient-facing canned message — sent immediately, no HITL required."""
    cats = state.get("moderation_categories", [])
    cat_str = " ".join(cats).lower()
    if any(k in cat_str for k in ("clinical_advice",)):
        return _CLINICAL_ADVICE_RESPONSE
    return _EMERGENCY_RESPONSE


def _build_staff_alert(state: dict) -> str:
    """Staff-facing HITL draft — requires acknowledgement before closing the case."""
    cats = state.get("moderation_categories", [])
    cat_str = " ".join(cats).lower()
    if any(k in cat_str for k in ("clinical_advice",)):
        return _STAFF_CLINICAL_ALERT
    return _STAFF_EMERGENCY_ALERT


def escalation_node(state: dict) -> dict:
    """LangGraph node: send canned patient response immediately; alert staff via HITL."""
    patient_response = _build_patient_response(state)
    staff_alert      = _build_staff_alert(state)
    reason = ", ".join(state.get("moderation_categories", ["risk content detected"]))
    print(f"  [Escalation] Escalating run. Reason: {reason}")

    return {
        "status":         "ESCALATE",
        "final_response": patient_response,  # auto-sent to patient, no approval needed
        "draft_response": staff_alert,       # shown to staff in HITL panel for acknowledgement
        "route_path":     ["escalation"],
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"[ESCALATE] {patient_response}"},
        ],
    }
