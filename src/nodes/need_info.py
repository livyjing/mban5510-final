"""
Need-Info Node
==============
Handles messages where intent could not be determined (intent = "unknown").

Instead of proceeding to an appointment action (which would require a clear
intent), the system responds with a polite clarification request and sets
``status = "NEED_INFO"``.

Design decision
---------------
The NEED_INFO path does NOT enter the HITL loop.  Clarification requests
are deterministic enough that human review adds minimal value and would
slow down an already uncertain interaction.  This is documented in the README
architecture section.
"""

_CLARIFICATION_RESPONSE = """\
Thank you for contacting Halifax Medical Centre's appointment assistance line.

I wasn't able to determine exactly what you need help with from your message.
This system can assist you with the following:

  1. Rescheduling an existing appointment
  2. Cancelling an existing appointment
  3. Getting preparation instructions for an upcoming procedure
     (e.g., MRI, CT scan, blood work, ultrasound)

Could you please clarify which of the above you need, or rephrase your request?
You can also reach our scheduling team directly at (902) 555-0000.\
"""


def need_info_node(state: dict) -> dict:
    """LangGraph node: request clarification when intent is unknown."""
    print("  [NeedInfo] Intent unclear – requesting clarification.")
    return {
        "status":         "NEED_INFO",
        "final_response": _CLARIFICATION_RESPONSE,
        "draft_response": _CLARIFICATION_RESPONSE,
        "hitl_approved":  False,
        "route_path":     ["need_info", "END"],
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": f"[NEED_INFO] {_CLARIFICATION_RESPONSE}"},
        ],
    }
