"""
Finalize Node
=============
Terminal node for the normal (non-escalation) workflow path.

Responsibilities
----------------
* Sets ``status = "READY"`` (the final approved response is ready to send).
* Appends the "finalize" label to ``route_path``.
* Appends "END" to ``route_path`` to mark workflow completion.

No LLM or tool calls are made here; this is a lightweight bookkeeping node.
"""


def finalize_node(state: dict) -> dict:
    """LangGraph node: mark the run as READY and close the route trace."""
    print("  [Finalize] Workflow complete. Status → READY")
    # When HITL was bypassed (e.g. prep_instructions), final_response is not yet
    # set; promote draft_response so callers always find the text in final_response.
    final = state.get("final_response") or state.get("draft_response", "")
    return {
        "status":         "READY",
        "final_response": final,
        "route_path":     ["finalize", "END"],
    }
