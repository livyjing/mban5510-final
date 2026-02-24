#!/usr/bin/env python3
"""
MBAN 5510 Final Project — Appointment Assistance System
========================================================
CLI Entry Point

Usage
-----
Interactive mode (default):
    python main.py

Single message via flag:
    python main.py --message "I need to reschedule my MRI"
    python main.py -m "Cancel my appointment APT-002"
    python main.py -m "What do I need to do before my CT scan?"

Non-interactive HITL (auto-approve, useful for testing):
    python main.py -m "Cancel my appointment" --auto-approve

Environment variables (see .env.example):
    OPENAI_API_KEY  – required
"""

import argparse
import sys
import textwrap
import uuid
from datetime import datetime

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()  # Load OPENAI_API_KEY and any other vars from .env

from src.state import initial_state
from src.workflow import build_workflow


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 62

def _banner(title: str) -> str:
    return f"\n{_SEP}\n  {title}\n{_SEP}"


def _print_trace(state: dict) -> None:
    """Print the verifiable execution evidence required by the spec."""
    route = " → ".join(state.get("route_path", ["?"]))
    pii_info = (
        f"PII masked ({', '.join(state.get('pii_types', []))})"
        if state.get("pii_detected")
        else "No PII detected"
    )

    print(_banner("EXECUTION TRACE"))
    print(f"  Run ID    : {state.get('run_id', 'N/A')}")
    print(f"  Timestamp : {state.get('timestamp', 'N/A')}")
    print(f"  Status    : {state.get('status', 'N/A')}")
    print(f"  Intent    : {state.get('intent', 'N/A')}")
    print(f"  PII       : {pii_info}")
    print(f"  HITL      : {state.get('hitl_action', 'N/A')}")
    print(f"  Route     : {route}")
    print(_SEP)


def _print_final_response(state: dict) -> None:
    """Print the final client-facing response."""
    print(_banner("FINAL RESPONSE"))
    response = state.get("final_response") or "(no response generated)"
    # Wrap long lines for clean terminal output
    for line in response.split("\n"):
        wrapped = textwrap.fill(line, width=60, subsequent_indent="  ") if len(line) > 60 else line
        print(f"  {wrapped}")
    print(_SEP)


def _print_hitl_prompt(draft: str) -> None:
    """Display the draft response for the human reviewer."""
    print(_banner("HUMAN REVIEW REQUIRED  [HumanInTheLoopMiddleware]"))
    print("  The AI has generated the following draft response:")
    print()
    for line in draft.split("\n"):
        print(f"    {line}")
    print()
    print("  Options:")
    print("    [A] Approve – send the draft as-is")
    print("    [E] Edit   – provide your own response")
    print(_SEP)


# ---------------------------------------------------------------------------
# HITL interaction
# ---------------------------------------------------------------------------

def _collect_human_decision(draft: str, auto_approve: bool) -> dict:
    """
    Present the draft to the human reviewer and collect a decision.

    Parameters
    ----------
    draft       : str   – the AI-generated draft response
    auto_approve: bool  – skip interactive input and approve automatically

    Returns
    -------
    dict  {"action": "approve"} or {"action": "edit", "edited_response": "..."}
    """
    if auto_approve:
        print("  [HITL] Auto-approve mode – approving draft without review.")
        return {"action": "approve"}

    _print_hitl_prompt(draft)

    while True:
        choice = input("  Your choice [A/E]: ").strip().upper()
        if choice in ("A", ""):
            return {"action": "approve"}
        elif choice == "E":
            print("  Enter your edited response.")
            print("  (Type your message; press Enter twice when done.)")
            lines = []
            while True:
                line = input("  > ")
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            # Remove trailing blank line
            while lines and lines[-1] == "":
                lines.pop()
            edited = "\n".join(lines).strip()
            if edited:
                return {"action": "edit", "edited_response": edited}
            else:
                print("  (Empty edit – falling back to approve)")
                return {"action": "approve"}
        else:
            print("  Please enter A to approve or E to edit.")


# ---------------------------------------------------------------------------
# Core workflow runner
# ---------------------------------------------------------------------------

def run_workflow(
    message: str,
    auto_approve: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Execute the appointment-assistance workflow end-to-end.

    Parameters
    ----------
    message     : str  – patient's natural-language request
    auto_approve: bool – bypass interactive HITL (auto-approve)
    verbose     : bool – print per-node progress messages

    Returns
    -------
    The final ``AppointmentState`` dict after workflow completion.
    """
    run_id    = f"run_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat(timespec="seconds")

    print(f"\n{'─'*62}")
    print(f"  Appointment Assistance System  |  Run: {run_id}")
    print(f"  {timestamp}")
    print(f"{'─'*62}")
    print(f"  Patient message: {message!r}")
    print(f"{'─'*62}\n")

    # Build a fresh workflow graph (includes fresh MemorySaver)
    workflow = build_workflow()
    config   = {"configurable": {"thread_id": run_id}}
    state_0  = initial_state(run_id, timestamp, message)

    # ── First execution pass ──────────────────────────────────────────────────
    last_state: dict = {}
    for snapshot in workflow.stream(state_0, config, stream_mode="values"):
        last_state = snapshot

    # ── Check for HITL interrupt ──────────────────────────────────────────────
    graph_snapshot = workflow.get_state(config)

    if graph_snapshot.next:
        # The human_review node used interrupt() – we are now paused
        draft = last_state.get("draft_response", "")
        decision = _collect_human_decision(draft, auto_approve)

        # ── Resume with human decision ─────────────────────────────────────────
        for snapshot in workflow.stream(
            Command(resume=decision),
            config,
            stream_mode="values",
        ):
            last_state = snapshot

    # ── Output ────────────────────────────────────────────────────────────────
    _print_trace(last_state)
    _print_final_response(last_state)

    return last_state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "MBAN 5510 Final Project — Appointment Assistance System\n"
            "A LangGraph-based middleware-driven appointment workflow with HITL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py -m "I need to reschedule my MRI"
              python main.py -m "Cancel appointment APT-002"
              python main.py -m "What should I do before my CT scan?"
              python main.py -m "I have chest pain"          # escalation demo
              python main.py -m "reschedule" --auto-approve  # non-interactive
        """),
    )
    parser.add_argument(
        "-m", "--message",
        type=str,
        help="Patient message to process (omit for interactive prompt)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve the HITL draft without interactive review",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.message:
        message = args.message.strip()
    else:
        print("\nAppointment Assistance System (MBAN 5510 Final Project)")
        print("─" * 62)
        print("Supported requests:")
        print("  • Reschedule an appointment")
        print("  • Cancel an appointment")
        print("  • Request preparation instructions")
        print("  • Type 'quit' to exit")
        print()

        while True:
            try:
                message = input("Patient message > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                sys.exit(0)

            if not message:
                continue
            if message.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                sys.exit(0)
            break

    run_workflow(message, auto_approve=args.auto_approve)


if __name__ == "__main__":
    main()
