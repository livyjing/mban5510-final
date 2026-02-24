"""
LangGraph Workflow Assembly
===========================
Constructs and compiles the stateful appointment-assistance workflow.

Architecture
------------
The graph implements a *middleware-style orchestration* pattern where each
node either IS a middleware component or delegates directly to one:

  START
    │
    ▼
  pii_check           ← PIIMiddleware            (PII detection & masking)
    │
    ▼
  moderation_check    ← ModerationMiddleware      (safety screening)
    │
    ├─[flagged]──► escalation ───────────────────────────────────► END
    │
    ▼
  intent_detect       ← LLM + ModelRetryMiddleware + ModelFallbackMiddleware
    │                   + ModelCallLimitMiddleware
    │
    ├─[escalate]──► escalation ──────────────────────────────────► END
    │
    ├─[unknown] ──► general_qa ──────────────────────────────────► END
    │               (AI answers low-risk general questions)
    │
    ├─[cancel]──────┐
    ├─[reschedule]──┼──► validate_info  (checks minimum required entities)
    └─[prep]────────┘       │
                            ├─[missing fields]──────────────────► END  (NEED_INFO)
                            │
                            ├─[ok + cancel/reschedule]──► identity_gate
                            │                               ├─[LOGIN_REQUIRED]─► END
                            │                               ├─[not verified]───► END
                            │                               ├─[reschedule ok]──► reschedule ─┐
                            │                               └─[cancel ok]──────► cancel ──────┤
                            │                                                                 │
                            └─[ok + prep]──────────────────────► prep ───────────────────────┤
                                                                                              ▼
                                                                                    draft_generation
                                                                                              │
                                                                                              ▼
                                                                                       human_review
                                                                                    (interrupt/resume)
                                                                                              │
                                                                                              ▼
                                                                                          finalize
                                                                                              │
                                                                                              ▼
                                                                                            END

Routing logic
-------------
After ``moderation_check``: flagged → escalation; clean → intent_detect
After ``intent_detect``:
  cancel / reschedule / prep_instructions → validate_info
  escalate                                → escalation
  unknown                                 → general_qa
After ``validate_info``:    missing fields → END (NEED_INFO already set)
                            ok + cancel/reschedule → identity_gate
                            ok + prep              → prep_instructions
After ``identity_gate``:    LOGIN_REQUIRED → END; not verified → END
                            verified → reschedule or cancel (by state["intent"])

Checkpointing
-------------
A ``MemorySaver`` checkpointer is attached at compile time.  This is
*required* for ``interrupt()`` / ``Command(resume=…)`` to function correctly.
The checkpointer stores the full graph state between the interrupt and resume.

Returns
-------
``build_workflow()`` returns the compiled ``CompiledGraph`` ready for streaming.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes.appointment_actions import (
    cancel_node,
    prep_instructions_node,
    reschedule_node,
)
from src.nodes.draft_generator import draft_generation_node
from src.nodes.escalation import escalation_node
from src.nodes.finalize import finalize_node
from src.nodes.general_qa import general_qa_node
from src.nodes.human_review import human_review_node
from src.nodes.identity_gate import identity_gate_node
from src.nodes.intent_detection import intent_detection_node
from src.nodes.moderation_check import moderation_check_node
from src.nodes.need_info import need_info_node
from src.nodes.pii_check import pii_check_node
from src.nodes.validate_info import validate_info_node
from src.state import AppointmentState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_moderation(state: AppointmentState) -> str:
    """
    After moderation_check: if the message was flagged, go straight to
    escalation.  Otherwise proceed to intent detection.
    """
    if state.get("moderation_flagged") or state.get("intent") == "escalate":
        return "escalation"
    return "intent_detect"


def _route_after_intent(state: AppointmentState) -> str:
    """
    Route to the appropriate node based on classified intent.

    Cancel / reschedule go to identity_gate FIRST (login before validation),
    so the patient profile panel is visible when the NEED_INFO prompt appears.
    Prep_instructions go to validate_info first (no login needed).
    """
    intent = state.get("intent", "unknown")

    routing_table = {
        "reschedule":        "identity_gate",   # login first, then validate
        "cancel":            "identity_gate",   # login first, then validate
        "prep_instructions": "validate_info",   # validate first, no login needed
        "escalate":          "escalation",
        "unknown":           "general_qa",
    }
    destination = routing_table.get(intent, "general_qa")
    print(f"  [Router] intent={intent!r} → node={destination!r}")
    return destination


def _route_after_identity(state: AppointmentState) -> str:
    """
    After identity_gate:
    - LOGIN_REQUIRED (no [Patient:] prefix) → END  (app shows login form)
    - Not verified (name not in DB)         → END  (rejection message set)
    - Verified                              → validate_info (check required fields
                                              while profile panel is already visible)
    """
    if state.get("status") == "LOGIN_REQUIRED":
        return END
    if not state.get("identity_verified", False):
        return END
    return "validate_info"   # always → validate_info after successful auth


def _route_after_draft(state: AppointmentState) -> str:
    """
    After draft_generation:
    - prep_instructions intent → finalize directly (no HITL; factual info, not an action)
    - reschedule / cancel     → human_review (HITL required for irreversible actions)
    """
    if state.get("intent") == "prep_instructions":
        return "finalize"
    return "human_review"


def _route_after_validate(state: AppointmentState) -> str:
    """
    After validate_info:
    - NEED_INFO set (fields missing) → END  (profile panel already visible; user
                                       can read their appointment IDs and re-submit)
    - All fields present             → action node (reschedule or cancel or prep)
    """
    if state.get("status") == "NEED_INFO":
        return END
    intent = state.get("intent", "")
    if intent == "reschedule":
        return "reschedule"
    if intent == "cancel":
        return "cancel"
    return "prep_instructions"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_workflow():
    """
    Build and compile the appointment-assistance LangGraph workflow.

    Returns
    -------
    CompiledGraph
        The compiled graph with MemorySaver checkpointing enabled.
    """
    builder = StateGraph(AppointmentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("pii_check",          pii_check_node)
    builder.add_node("moderation_check",   moderation_check_node)
    builder.add_node("intent_detect",      intent_detection_node)
    builder.add_node("validate_info",      validate_info_node)
    builder.add_node("identity_gate",      identity_gate_node)
    builder.add_node("reschedule",         reschedule_node)
    builder.add_node("cancel",             cancel_node)
    builder.add_node("prep_instructions",  prep_instructions_node)
    builder.add_node("draft_generation",   draft_generation_node)
    builder.add_node("human_review",       human_review_node)
    builder.add_node("finalize",           finalize_node)
    builder.add_node("escalation",         escalation_node)
    builder.add_node("need_info",          need_info_node)   # retained for CLI use
    builder.add_node("general_qa",         general_qa_node)

    # ── Entry edge ────────────────────────────────────────────────────────────
    builder.add_edge(START, "pii_check")

    # ── Sequential edges ──────────────────────────────────────────────────────
    builder.add_edge("pii_check", "moderation_check")

    # ── Conditional: after moderation ─────────────────────────────────────────
    builder.add_conditional_edges(
        "moderation_check",
        _route_after_moderation,
        {
            "escalation":    "escalation",
            "intent_detect": "intent_detect",
        },
    )

    # ── Conditional: after intent detection ───────────────────────────────────
    # cancel/reschedule → identity_gate first (login before validation)
    # prep_instructions → validate_info first (no login needed)
    builder.add_conditional_edges(
        "intent_detect",
        _route_after_intent,
        {
            "identity_gate": "identity_gate",
            "validate_info": "validate_info",
            "escalation":    "escalation",
            "general_qa":    "general_qa",
        },
    )

    # ── Conditional: after identity gate ──────────────────────────────────────
    # Verified → validate_info (profile panel now visible; NEED_INFO prompt makes sense)
    builder.add_conditional_edges(
        "identity_gate",
        _route_after_identity,
        {
            "validate_info": "validate_info",
            END:             END,
        },
    )

    # ── Conditional: after info validation ────────────────────────────────────
    builder.add_conditional_edges(
        "validate_info",
        _route_after_validate,
        {
            "reschedule":        "reschedule",
            "cancel":            "cancel",
            "prep_instructions": "prep_instructions",
            END:                 END,
        },
    )

    # ── Action nodes → draft generation ──────────────────────────────────────
    builder.add_edge("reschedule",        "draft_generation")
    builder.add_edge("cancel",            "draft_generation")
    builder.add_edge("prep_instructions", "draft_generation")

    # ── Draft → HITL → Finalize ───────────────────────────────────────────────
    # prep_instructions skips HITL (factual info, not an irreversible action)
    builder.add_conditional_edges(
        "draft_generation",
        _route_after_draft,
        {
            "human_review": "human_review",
            "finalize":     "finalize",
        },
    )
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize",         END)

    # ── Terminal paths ────────────────────────────────────────────────────────
    # escalation → human_review: staff must review & send the canned response
    builder.add_edge("escalation", "human_review")
    builder.add_edge("need_info",  END)
    builder.add_edge("general_qa", END)

    # ── Compile with MemorySaver (required for interrupt/resume) ──────────────
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
