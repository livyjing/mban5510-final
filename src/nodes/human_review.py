"""
Human Review Node  (HumanInTheLoopMiddleware)
=============================================
The HITL gate in the workflow.  Uses LangGraph's ``interrupt()`` primitive to
*pause* graph execution, hand control back to the caller (CLI or UI), and
wait for a human reviewer to either approve the draft or supply an edited
version.

How it works
------------
1. ``interrupt(payload)`` is called with the draft response as context.
   LangGraph serialises the current state to the MemorySaver checkpoint and
   raises ``NodeInterrupt``, which causes the stream to end gracefully.

2. The caller (``main.py``) detects the interrupted state by checking
   ``graph.get_state(config).next``, shows the draft to the reviewer, and
   collects a decision via ``input()``.

3. The caller resumes the graph with
   ``graph.stream(Command(resume=decision), config)``, where ``decision``
   is a dict ``{"action": "approve"}`` or ``{"action": "edit", "edited_response": "..."}``.

4. Execution continues from this node; ``interrupt()`` returns the resume
   value and the node writes the final response to state.

State fields updated
--------------------
final_response, hitl_approved, hitl_action, route_path, messages
"""

from langgraph.types import interrupt


def human_review_node(state: dict) -> dict:
    """
    LangGraph node: pause for human review via interrupt/resume.

    The interrupt payload exposes the draft and intent so that the host
    application can present them meaningfully to the reviewer without
    having to re-read the full state.
    """
    draft = state.get("draft_response", "")

    # ── Pause here; resume value is whatever the caller passes to Command(resume=…)
    decision: dict = interrupt({
        "type":   "human_review",
        "intent": state.get("intent"),
        "draft":  draft,
    })

    action   = decision.get("action", "approve")
    edited   = decision.get("edited_response", "")
    final    = edited.strip() if (action == "edit" and edited.strip()) else draft

    if action == "edit":
        print("  [HumanReview] Reviewer edited the draft.")
    else:
        print("  [HumanReview] Reviewer approved the draft.")

    return {
        "final_response": final,
        "hitl_approved":  True,
        "hitl_action":    action,
        "route_path":     [f"hitl_{action}"],
        "messages": state.get("messages", []) + [
            {"role": "human_reviewer", "content": f"[{action.upper()}] {final}"},
        ],
    }
