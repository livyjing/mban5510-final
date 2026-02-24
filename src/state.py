"""
Workflow state definition for the Appointment Assistance System.

The AppointmentState TypedDict is the single source of truth flowing through
every LangGraph node.  All fields are explicitly typed so that routing logic,
middleware checks, and trace output can reference them unambiguously.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict


class AppointmentState(TypedDict):
    """
    Shared state object passed between every node in the LangGraph workflow.

    Sections
    --------
    Traceability    – run identification and route auditing
    Patient Input   – raw and PII-sanitised message
    Intent          – classification result and extracted entities
    Middleware      – counters and flags populated by middleware components
    Response        – draft → HITL → final pipeline
    Status          – terminal status emitted at workflow end
    History         – running message log for context
    """

    # ── Traceability ──────────────────────────────────────────────────────────
    run_id: str                                     # Unique execution identifier
    timestamp: str                                  # ISO-8601 start timestamp
    # Append-only list: each node appends its label so the full route is visible
    route_path: Annotated[List[str], operator.add]

    # ── Patient Input ─────────────────────────────────────────────────────────
    user_message: str                               # Raw patient message
    user_message_masked: str                        # PII-redacted version (logs only)

    # ── Intent & Extracted Entities ───────────────────────────────────────────
    intent: Optional[
        Literal["reschedule", "cancel", "prep_instructions", "escalate", "unknown"]
    ]
    appointment_id: Optional[str]
    item_id: Optional[str]          # Specific item within an appointment (e.g. APT-001-A)
    appointment_type: Optional[str]
    patient_name: Optional[str]
    preferred_date: Optional[str]

    # ── Identity Verification ─────────────────────────────────────────────────
    identity_verified: bool                         # Set by identity_gate node

    # ── Middleware Tracking ───────────────────────────────────────────────────
    moderation_flagged: bool                        # OpenAI Moderation / keyword result
    moderation_categories: List[str]                # Flagged categories
    pii_detected: bool                              # Whether PII was found in input
    pii_types: List[str]                            # Types of PII detected
    tool_call_count: int                            # ToolCallLimitMiddleware counter
    model_call_count: int                           # ModelCallLimitMiddleware counter
    retry_count: int                                # ModelRetryMiddleware attempts

    # ── Action Result (populated by appointment action nodes) ─────────────────
    action_result: Optional[Dict]

    # ── Response Pipeline ─────────────────────────────────────────────────────
    draft_response: Optional[str]                   # AI-generated draft
    final_response: Optional[str]                   # Post-HITL final output

    # ── Human-in-the-Loop ────────────────────────────────────────────────────
    hitl_approved: bool                             # HITL step completed
    hitl_action: Optional[Literal["approve", "edit"]]  # Human's decision

    # ── Terminal Status ───────────────────────────────────────────────────────
    status: Optional[Literal["READY", "NEED_INFO", "ESCALATE", "LOGIN_REQUIRED"]]

    # ── Conversation History ──────────────────────────────────────────────────
    messages: List[dict]


def initial_state(run_id: str, timestamp: str, user_message: str) -> AppointmentState:
    """Return a fully-initialised state dict for a new workflow run."""
    return AppointmentState(
        run_id=run_id,
        timestamp=timestamp,
        route_path=["START"],
        user_message=user_message,
        user_message_masked="",
        intent=None,
        appointment_id=None,
        item_id=None,
        appointment_type=None,
        patient_name=None,
        preferred_date=None,
        identity_verified=False,
        moderation_flagged=False,
        moderation_categories=[],
        pii_detected=False,
        pii_types=[],
        tool_call_count=0,
        model_call_count=0,
        retry_count=0,
        action_result=None,
        draft_response=None,
        final_response=None,
        hitl_approved=False,
        hitl_action=None,
        status=None,
        messages=[],
    )
