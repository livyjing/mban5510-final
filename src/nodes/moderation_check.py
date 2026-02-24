"""
Moderation Check Node
=====================
Second node in the workflow.  Delegates to ``ModerationMiddleware`` to
screen the (already PII-masked) patient message for:

  * Medical emergencies requiring immediate escalation
  * Requests for clinical advice (out-of-scope for this system)
  * General harmful content via the OpenAI Moderation API

If the message is flagged, the node sets ``intent = "escalate"`` in the
state, which causes the conditional router after intent_detect to send
the run directly to the escalation node.
"""

from src.middleware.moderation import ModerationMiddleware

_mod = ModerationMiddleware()


def moderation_check_node(state: dict) -> dict:
    """LangGraph node: run ModerationMiddleware safety screening."""
    return _mod.run_on_state(state)
