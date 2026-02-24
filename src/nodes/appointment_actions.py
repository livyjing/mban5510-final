"""
Appointment Action Nodes
========================
Three separate LangGraph nodes that perform the core appointment operations:

  reschedule_node       – looks up the appointment and processes a reschedule
  cancel_node           – looks up the appointment and processes a cancellation
  prep_instructions_node – retrieves preparation instructions for a procedure

Each node:
  1. Enforces the tool call limit (ToolCallLimitMiddleware)
  2. Wraps the data call in retry logic (ToolRetryMiddleware)
  3. Stores the structured result in ``state["action_result"]`` for the
     downstream draft_generation node
  4. Appends a trace label to ``route_path``

State fields updated
--------------------
action_result, appointment_id, appointment_type, patient_name,
tool_call_count, route_path, messages
"""

from src.middleware.limits import ToolCallLimitMiddleware
from src.middleware.retry import ToolRetryMiddleware
from src.tools.appointment_tools import (
    cancel_appointment,
    get_prep_instructions,
    look_up_appointment,
    reschedule_appointment,
)

_tool_limiter = ToolCallLimitMiddleware(max_calls=20)
_tool_retry   = ToolRetryMiddleware(max_retries=2, base_delay=0.5)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve_appointment(state: dict) -> dict:
    """
    Call look_up_appointment using the best available context from state.
    item_id takes priority when present (targets a specific procedure).
    """
    return look_up_appointment(
        appointment_id=state.get("appointment_id") or "",
        patient_name=state.get("patient_name") or "",
        appointment_type=state.get("appointment_type") or "",
        item_id=state.get("item_id") or "",
    )


# ---------------------------------------------------------------------------
# Node: reschedule
# ---------------------------------------------------------------------------

def reschedule_node(state: dict) -> dict:
    """LangGraph node: process a reschedule request."""
    _tool_limiter.check(state)

    @_tool_retry.wrap
    def execute():
        lookup = _resolve_appointment(state)
        if not lookup["found"]:
            return {"success": False, "error": lookup.get("error", "Appointment not found.")}
        apt          = lookup["appointment"]
        matched_item = lookup.get("matched_item")
        # When the user identified by item_id (not appointment_type), derive the type
        # from the matched item so reschedule_appointment can narrow to that single item.
        apt_type = state.get("appointment_type") or (
            matched_item["type"] if matched_item else None
        )
        return reschedule_appointment(
            apt["id"],
            new_date=state.get("preferred_date"),
            appointment_type=apt_type,
        )

    result = execute()
    print(f"  [RescheduleNode] {result.get('message', result.get('error'))}")

    return {
        "action_result":    result,
        "appointment_id":   result.get("appointment_id",   state.get("appointment_id")),
        "appointment_type": result.get("appointment_type", state.get("appointment_type")),
        "patient_name":     result.get("patient_name",     state.get("patient_name")),
        "route_path":       ["reschedule"],
        "tool_call_count":  state.get("tool_call_count", 0) + 1,
        "messages": state.get("messages", []) + [
            {"role": "tool", "name": "reschedule_appointment", "content": str(result)},
        ],
    }


# ---------------------------------------------------------------------------
# Node: cancel
# ---------------------------------------------------------------------------

def cancel_node(state: dict) -> dict:
    """LangGraph node: process a cancellation request."""
    _tool_limiter.check(state)

    @_tool_retry.wrap
    def execute():
        lookup = _resolve_appointment(state)
        if not lookup["found"]:
            return {"success": False, "error": lookup.get("error", "Appointment not found.")}
        apt          = lookup["appointment"]
        matched_item = lookup.get("matched_item")
        # When the user identified by item_id (not appointment_type), derive the type
        # from the matched item so cancel_appointment can narrow to that single item.
        apt_type = state.get("appointment_type") or (
            matched_item["type"] if matched_item else None
        )
        return cancel_appointment(
            apt["id"],
            appointment_type=apt_type,
        )

    result = execute()
    print(f"  [CancelNode] {result.get('message', result.get('error'))}")

    return {
        "action_result":    result,
        "appointment_id":   result.get("appointment_id",   state.get("appointment_id")),
        "appointment_type": result.get("appointment_type", state.get("appointment_type")),
        "patient_name":     result.get("patient_name",     state.get("patient_name")),
        "route_path":       ["cancel"],
        "tool_call_count":  state.get("tool_call_count", 0) + 1,
        "messages": state.get("messages", []) + [
            {"role": "tool", "name": "cancel_appointment", "content": str(result)},
        ],
    }


# ---------------------------------------------------------------------------
# Node: prep_instructions
# ---------------------------------------------------------------------------

def prep_instructions_node(state: dict) -> dict:
    """LangGraph node: retrieve preparation instructions for a procedure."""
    _tool_limiter.check(state)

    @_tool_retry.wrap
    def execute():
        apt_type = state.get("appointment_type") or ""

        # If no type was extracted, infer from raw message keywords
        if not apt_type:
            msg = state.get("user_message", "").lower()
            if "mri" in msg:
                apt_type = "MRI"
            elif "ct" in msg:
                apt_type = "CT Scan"
            elif "blood" in msg:
                apt_type = "Blood Work"
            elif "ultrasound" in msg:
                apt_type = "Ultrasound"
            else:
                apt_type = "General"

        return get_prep_instructions(apt_type)

    result = execute()
    resolved_type = result.get("appointment_type", "General")
    print(f"  [PrepInstructionsNode] Instructions retrieved for: {resolved_type!r}")

    return {
        "action_result":    result,
        "appointment_type": resolved_type,
        "route_path":       ["prep_instructions"],
        "tool_call_count":  state.get("tool_call_count", 0) + 1,
        "messages": state.get("messages", []) + [
            {"role": "tool", "name": "get_prep_instructions", "content": str(result)},
        ],
    }
