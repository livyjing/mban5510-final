"""
MBAN 5510 Final Project — Appointment Assistance System
Web UI (Gradio)
===============
Optional web interface for demonstration purposes.

Run:
    python app.py
    # opens http://127.0.0.1:7860  in your browser

Three-panel layout
------------------
LEFT   – Message input + NEED_INFO prompt (shown inline) + Login panel + Patient profile
CENTRE – Execution Trace (live node-by-node progress)
RIGHT  – HITL Review panel (approve / edit draft) → Final Response

Authentication flow
-------------------
1. User types a message and submits — no pre-selection required.
2. The workflow classifies intent first.
   - If login is needed (cancel/reschedule) → login panel appears.
3. User enters:  username = patient name without spaces (e.g. AlexMorgan)
                 password = 123 (same for everyone)
4. On success the workflow re-runs with [Patient: name] prepended.
   - If required info is missing → NEED_INFO prompt appears above the input box.
   - If info is complete → HITL review or direct response.
   Patient profile panel shows name, phone, and full active-appointment table.
5. Logged-in state persists for the session tab.  Logout button clears state.

Prep-instruction and general-question requests are answered immediately
without requiring login.
"""

import re as _re_dt
import shutil
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path
import sys

import gradio as gr
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

from src.state import initial_state
from src.workflow import build_workflow
from src.tools.appointment_tools import set_active_file

# ---------------------------------------------------------------------------
# Session database  (master is never modified; session resets on each launch)
# ---------------------------------------------------------------------------
_MASTER_PATH  = Path(__file__).parent / "data" / "appointments.json"
_SESSION_PATH = Path(__file__).parent / "data" / "appointments_session.json"

shutil.copy(_MASTER_PATH, _SESSION_PATH)
set_active_file(_SESSION_PATH)

# ---------------------------------------------------------------------------
# In-memory run registry  (maps thread_id → workflow + last_state)
# ---------------------------------------------------------------------------
_registry: dict = {}


# ---------------------------------------------------------------------------
# Log-capturing helper
# ---------------------------------------------------------------------------

class _Capture:
    """Redirect stdout prints to a string buffer during workflow execution."""
    def __init__(self):
        self._buf = StringIO()
        self._old = None

    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = self._buf
        return self

    def __exit__(self, *_):
        sys.stdout = self._old

    def value(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Credential & patient helpers
# ---------------------------------------------------------------------------

def _build_credentials() -> dict:
    """
    Return {username_lower: patient_name} from the session file.
    Username is the patient's full name with spaces removed, lowercased.
    e.g. "Alex Morgan" → "alexmorgan": "Alex Morgan"
    """
    import json
    with open(_SESSION_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {
        apt["patient_name"].replace(" ", "").lower(): apt["patient_name"]
        for apt in data["appointments"]
    }


def _validate_login(username: str, password: str):
    """
    Validate login credentials.  Returns patient_name string on success,
    None on failure.
    """
    if password.strip() != "123":
        return None
    key = username.strip().lower().replace(" ", "")
    return _build_credentials().get(key)


def _patient_profile(patient_name: str) -> str:
    """
    Return a rich Markdown profile panel for the logged-in patient.
    Shows patient name, phone, and a full active-appointment table
    (item ID, procedure, date & time, doctor, location).
    Reads from session file so cancelled/rescheduled items are reflected live.
    """
    import json
    if not patient_name:
        return ""

    with open(_SESSION_PATH, encoding="utf-8") as f:
        data = json.load(f)

    apts = [a for a in data["appointments"]
            if patient_name.lower() in a["patient_name"].lower()]
    if not apts:
        return f"_No appointments found for {patient_name}._"

    rec = apts[0]   # primary record for basic patient info
    rows = []
    for apt in apts:
        if apt.get("status") == "cancelled":
            continue
        for item in apt.get("items", []):
            if item.get("status") == "cancelled":
                continue
            datetime_display = _fmt_appointment_dt(
                str(item.get("date", apt["date"])),
                str(item.get("time", "")),
            )
            rows.append(
                f"| `{item['item_id']}` | **{item['type']}** "
                f"| {datetime_display} "
                f"| {item['doctor']} "
                f"| {item['location']} |"
            )

    if not rows:
        active_block = "_No active appointments._"
    else:
        header = (
            "| Item ID | Procedure | Date & Time | Doctor | Location |\n"
            "|---------|-----------|-------------|--------|----------|"
        )
        active_block = header + "\n" + "\n".join(rows)

    return (
        f"### 👤 {rec['patient_name']}\n"
        f"📞 {rec['patient_phone']}\n\n"
        f"**Active Appointments**\n\n"
        f"{active_block}"
    )


def _fmt_appointment_dt(date_str: str, time_str: str) -> str:
    """
    Normalise a stored appointment date + time to a single consistent display string.

    Target format : "Apr 15, 2026 · 9:00 AM"
    Fallback      : original strings joined with ' · ' if parsing fails

    Handles:
      - ISO dates       "2026-04-15"
      - Long/short name "March 15, 2026" / "Mar 15, 2026"
      - No-year dates   "April 15"  (year inferred as current year)
      - Embedded time   "2026-04-15 9:00am"  (time extracted from date_str)
      - Relative dates  "next Tuesday"  (returned as-is, cannot parse)
    """
    # Extract time embedded inside date_str (e.g. "2026-04-15 9:00am")
    tm = _re_dt.search(
        r'(\d{1,2}:\d{2}\s*(?:am|pm)|\b\d{1,2}\s*(?:am|pm)\b)',
        date_str, _re_dt.IGNORECASE,
    )
    if tm:
        if not time_str:
            time_str = tm.group(1).strip()
        date_str = date_str[:tm.start()].strip().rstrip(',').strip()

    # Parse date — year-bearing formats first, then inject current year for no-year formats
    date_obj = None
    cur_year = datetime.now().year
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            date_obj = datetime.strptime(date_str.strip(), fmt)
            break
        except ValueError:
            pass
    if date_obj is None:
        for fmt in ("%B %d", "%b %d", "%m/%d"):
            try:
                d = datetime.strptime(f"{date_str.strip()} {cur_year}", fmt + " %Y")
                date_obj = d
                break
            except ValueError:
                pass

    # Parse time
    time_obj = None
    if time_str:
        for fmt in ("%I:%M %p", "%I:%M%p", "%I %p", "%I%p", "%H:%M"):
            try:
                time_obj = datetime.strptime(time_str.strip().upper(), fmt)
                break
            except ValueError:
                pass

    # Format — platform-safe (avoid %-d / %#d)
    if date_obj:
        date_display = f"{date_obj.strftime('%b')} {date_obj.day}, {date_obj.year}"
    else:
        date_display = date_str  # unparseable (e.g. "next Tuesday") — keep as-is

    if time_obj:
        h = time_obj.hour % 12 or 12
        time_display = f"{h}:{time_obj.strftime('%M')} {'AM' if time_obj.hour < 12 else 'PM'}"
    elif time_str:
        time_display = time_str  # unparseable time — keep as-is
    else:
        time_display = ""

    return f"{date_display} · {time_display}" if time_display else date_display


def _split_date_time(raw: str) -> "tuple[str, str | None]":
    """
    Split a combined date-time string into (date_part, time_part).

    Examples
    --------
    "2026-04-15 9:00am"  → ("2026-04-15", "9:00 AM")
    "April 15 at 2pm"    → ("April 15 at", "2:00 PM")   ← trailing words stripped
    "April 15"           → ("April 15", None)
    """
    m = _re_dt.search(
        r'(\d{1,2}:\d{2}\s*(?:am|pm)|\b\d{1,2}\s*(?:am|pm)\b)',
        raw, _re_dt.IGNORECASE,
    )
    if not m:
        return raw, None
    time_raw  = m.group(1).strip()
    date_part = _re_dt.sub(r',?\s+at\s*$', '', raw[:m.start()].strip(), count=1).strip() or raw
    # Normalise: "9:00am" → "9:00 AM",  "2pm" → "2:00 PM"
    time_norm = _re_dt.sub(
        r'\s*(am|pm)$',
        lambda x: ' ' + x.group(1).upper(),
        time_raw,
        flags=_re_dt.IGNORECASE,
    )
    if ':' not in time_norm:
        time_norm = time_norm.replace(' AM', ':00 AM').replace(' PM', ':00 PM')
    return date_part, time_norm


# ---------------------------------------------------------------------------
# Workflow helpers
# ---------------------------------------------------------------------------

def _run_first_pass(message: str, patient_name: str = "") -> tuple[str, str, str, bool, str]:
    """
    Execute the workflow until it either completes or hits the HITL interrupt.

    Returns
    -------
    run_id       : str
    trace_log    : str   – captured stdout from all nodes
    draft        : str   – draft response (empty if no HITL needed)
    needs_hitl   : bool
    direct_resp  : str   – final response if no HITL
    """
    run_id    = f"run_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat(timespec="seconds")

    full_message = f"[Patient: {patient_name}] {message}" if patient_name else message

    wf     = build_workflow()
    config = {"configurable": {"thread_id": run_id}}
    s0     = initial_state(run_id, timestamp, full_message)

    with _Capture() as cap:
        last = {}
        for snap in wf.stream(s0, config, stream_mode="values"):
            last = snap

    gstate = wf.get_state(config)
    _registry[run_id] = {"wf": wf, "config": config, "last": last}

    trace = cap.value()
    route = " → ".join(last.get("route_path", []))
    needs_hitl = bool(gstate.next)
    display_status = "PENDING_REVIEW" if needs_hitl else (last.get("status") or "?")

    identity_ok = last.get("identity_verified", True)
    identity_str = "verified ✓" if identity_ok else ((last.get("status") or "FAILED") + " ✗")

    trace_display = (
        f"Run ID    : {run_id}\n"
        f"Timestamp : {timestamp}\n"
        f"Status    : {display_status}\n"
        f"Intent    : {last.get('intent', '?')}\n"
        f"Identity  : {identity_str}\n"
        f"PII       : {'detected – masked' if last.get('pii_detected') else 'none detected'}\n"
        f"Route     : {route}\n"
        f"\n── Node logs ──────────────────────────────\n"
        f"{trace}"
    )
    draft       = last.get("draft_response", "") if needs_hitl else ""
    direct_resp = last.get("final_response", "") if not needs_hitl else ""

    return run_id, trace_display, draft, needs_hitl, direct_resp


def _resume(run_id: str, action: str, edited: str) -> tuple[str, str]:
    """Resume a paused workflow after human review."""
    entry = _registry.get(run_id)
    if not entry:
        return "ERROR: run not found in registry.", ""

    wf, config = entry["wf"], entry["config"]
    decision = {"action": action}
    if action == "edit" and edited.strip():
        decision["edited_response"] = edited.strip()

    with _Capture() as cap:
        last = {}
        for snap in wf.stream(Command(resume=decision), config, stream_mode="values"):
            last = snap

    _registry[run_id]["last"] = last
    route = " → ".join(last.get("route_path", []))
    trace_update = (
        f"\n── HITL Resume ────────────────────────────\n"
        f"Action    : {action.upper()}\n"
        f"Status    : {last.get('status', '?')}\n"
        f"Route     : {route}\n"
        f"{cap.value()}"
    )
    return trace_update, last.get("final_response", "")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
#trace_box textarea { font-family: 'Courier New', monospace; font-size: 13px; }
#final_box textarea { font-size: 15px; line-height: 1.6; }
#draft_box textarea { font-size: 14px; background: #fffbf0; }
#need_info_box { background: #f0f7ff; border-left: 4px solid #3b82f6;
                 border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; }
#escalation_alert { background: #fff0f0; border-left: 4px solid #ef4444;
                    border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; }
.status-badge { font-weight: bold; }
"""


def build_ui():
    with gr.Blocks(title="Appointment Assistance System — MBAN 5510") as demo:

        # ── Persistent state ──────────────────────────────────────────────
        run_id_state            = gr.State("")
        trace_state             = gr.State("")
        logged_in_patient_state = gr.State("")   # "" = not logged in
        pending_msg_state       = gr.State("")   # message stored while awaiting login

        # ── Header ────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🏥 Appointment Assistance System
            **MBAN 5510 Final Project** | Middleware-Driven LangGraph Workflow with Human-in-the-Loop

            > Supports: **Reschedule** · **Cancel** · **Preparation Instructions** · **Escalation** · **General Q&A**
            """
        )

        with gr.Row():
            # ── LEFT COLUMN: Input + NEED_INFO + Login + Profile ───────────
            with gr.Column(scale=4):

                # Logged-in banner (hidden until login succeeds)
                logged_in_banner = gr.Markdown("", visible=False)

                gr.Markdown("### 📝 Your Request")

                # NEED_INFO inline prompt — shown above the input box when
                # the system needs more information from the patient.
                need_info_md = gr.Markdown(
                    "",
                    visible=False,
                    elem_id="need_info_box",
                )

                msg_input = gr.Textbox(
                    label="Enter your request",
                    placeholder='e.g. "Cancel my MRI" · "Reschedule APT-001 to April 5" · "What are your hours?"',
                    lines=3,
                )
                submit_btn = gr.Button("▶  Submit Request", variant="primary", size="lg")

                # Login panel — appears when LOGIN_REQUIRED is returned
                with gr.Group(visible=False) as login_panel:
                    gr.Markdown("### 🔐 Login Required")
                    gr.Markdown(
                        "To cancel or reschedule you must log in.\n\n"
                        "**Username:** your full name without spaces  \n"
                        "*(e.g. Alex Morgan → `AlexMorgan`)*\n\n"
                        "**Password:** `123`"
                    )
                    username_input = gr.Textbox(
                        label="Username",
                        placeholder="e.g. AlexMorgan",
                    )
                    password_input = gr.Textbox(
                        label="Password",
                        type="password",
                        placeholder="Password",
                    )
                    login_btn      = gr.Button("🔑  Log In", variant="primary")
                    login_error_md = gr.Markdown("")

                # Patient profile panel — shown after successful login
                with gr.Group(visible=False) as patient_profile_panel:
                    patient_profile_md = gr.Markdown("")
                    logout_btn = gr.Button("🚪  Log Out", variant="secondary", size="sm")

                gr.Markdown(
                    """
                    ---
                    **Status Legend**
                    | Status | Meaning |
                    |---|---|
                    | `READY` | Completed successfully |
                    | `NEED_INFO` | Missing required information |
                    | `ESCALATE` | Safety risk — directed to clinical staff |
                    | `LOGIN_REQUIRED` | Authentication needed |
                    """
                )

            # ── CENTRE COLUMN: Execution Trace ────────────────────────
            with gr.Column(scale=5):
                gr.Markdown("### 🔍 Execution Trace")
                trace_box = gr.Textbox(
                    label="Run log (route · middleware · nodes)",
                    lines=22,
                    max_lines=30,
                    interactive=False,
                    elem_id="trace_box",
                )

            # ── RIGHT COLUMN: HITL + Final Response ───────────────────
            with gr.Column(scale=5):
                gr.Markdown("### 👤 Human Review (HITL)")

                with gr.Group(visible=False) as hitl_panel:
                    escalation_alert_md = gr.Markdown(
                        "", visible=False, elem_id="escalation_alert"
                    )
                    gr.Markdown(
                        "**AI Draft generated.** Please review and approve or edit before sending."
                    )
                    draft_box = gr.Textbox(
                        label="AI Draft Response",
                        lines=8,
                        interactive=False,
                        elem_id="draft_box",
                    )
                    with gr.Row():
                        approve_btn = gr.Button("✅  Approve Draft", variant="primary")
                        edit_btn    = gr.Button("✏️  Edit Draft", variant="secondary")

                with gr.Group(visible=False) as edit_panel:
                    edited_box = gr.Textbox(
                        label="Your edited response",
                        lines=8,
                        placeholder="Type your revised response here…",
                    )
                    confirm_edit_btn = gr.Button("📤  Confirm Edit & Send", variant="primary")

                gr.Markdown("### 📨 Final Response")
                final_box = gr.Textbox(
                    label="Client-facing final response (post-review)",
                    lines=10,
                    interactive=False,
                    elem_id="final_box",
                )
                status_display = gr.Markdown("")

        # ── Event handlers ────────────────────────────────────────────────

        def on_submit(message: str, logged_in_patient: str):
            """First pass: run workflow until interrupt or completion."""
            if not message.strip():
                return (
                    gr.update(), gr.update(),
                    "Please enter a message.",
                    gr.update(visible=False), gr.update(),
                    gr.update(visible=False), gr.update(visible=False),
                    "", "", "",
                    gr.update(value="", visible=False),   # need_info_md
                    gr.update(value="", visible=False),   # escalation_alert_md
                )

            run_id, trace, draft, needs_hitl, direct_resp = _run_first_pass(
                message, patient_name=logged_in_patient
            )
            last   = _registry.get(run_id, {}).get("last", {})
            status = last.get("status", "")
            intent = last.get("intent", "")

            if status == "LOGIN_REQUIRED":
                return (
                    run_id, trace,
                    "",                              # trace_box hidden until after login
                    gr.update(visible=True),         # login_panel
                    message,                         # pending_msg_state
                    gr.update(visible=False),        # hitl_panel
                    gr.update(visible=False),        # edit_panel
                    "", "",
                    "🔐 **Login required to cancel or reschedule.**",
                    gr.update(value="", visible=False),   # need_info_md
                    gr.update(value="", visible=False),   # escalation_alert_md
                )

            if needs_hitl and status == "ESCALATE":
                # Patient response is already in final_response (auto-sent, no approval needed).
                # Staff sees a separate action checklist in the HITL panel for acknowledgement.
                patient_msg = last.get("final_response", "")
                alert_text = (
                    "🚨 **PATIENT IN DANGER — IMMEDIATE ACTION REQUIRED**\n\n"
                    "The automated safety response has been sent to the patient. "
                    "Review the action checklist below and **Approve** to confirm that "
                    "emergency protocol has been initiated."
                )
                return (
                    run_id, trace, trace,
                    gr.update(visible=False),        # login_panel
                    "",                              # pending_msg_state
                    gr.update(visible=True),         # hitl_panel
                    gr.update(visible=False),        # edit_panel
                    draft, "",                       # draft = staff alert checklist
                    "🚨 **ESCALATION — Awaiting staff acknowledgement**",
                    gr.update(value=patient_msg, visible=True),   # need_info_md: patient-facing msg
                    gr.update(value=alert_text, visible=True),    # escalation_alert_md: staff banner
                )

            if needs_hitl:
                return (
                    run_id, trace, trace,
                    gr.update(visible=False),        # login_panel
                    "",                              # pending_msg_state
                    gr.update(visible=True),         # hitl_panel
                    gr.update(visible=False),        # edit_panel
                    draft, "",
                    "⏳ **Waiting for human review…**",
                    gr.update(value="", visible=False),   # need_info_md
                    gr.update(value="", visible=False),   # escalation_alert_md
                )

            emoji = {"ESCALATE": "🚨", "NEED_INFO": "❓", "READY": "✅"}.get(status, "ℹ️")

            if status == "NEED_INFO":
                # Show the NEED_INFO prompt inline above the input box
                return (
                    run_id, trace, trace,
                    gr.update(visible=False),        # login_panel
                    "",                              # pending_msg_state
                    gr.update(visible=False),        # hitl_panel
                    gr.update(visible=False),        # edit_panel
                    "", "",                          # draft_box, final_box empty
                    f"❓ **Status: NEED_INFO**",
                    gr.update(value=direct_resp, visible=True),   # need_info_md shown
                    gr.update(value="", visible=False),           # escalation_alert_md
                )

            if intent == "prep_instructions":
                # Prep instructions: no HITL, show inline above the input box
                return (
                    run_id, trace, trace,
                    gr.update(visible=False),        # login_panel
                    "",                              # pending_msg_state
                    gr.update(visible=False),        # hitl_panel
                    gr.update(visible=False),        # edit_panel
                    "", "",                          # draft_box, final_box empty
                    f"ℹ️ **Preparation Instructions**",
                    gr.update(value=direct_resp, visible=True),   # need_info_md shown
                    gr.update(value="", visible=False),           # escalation_alert_md
                )

            if intent == "unknown":
                # General Q&A (health info / clinic questions): no HITL,
                # show answer inline in the patient panel — not in the hospital HITL panel.
                return (
                    run_id, trace, trace,
                    gr.update(visible=False),        # login_panel
                    "",                              # pending_msg_state
                    gr.update(visible=False),        # hitl_panel
                    gr.update(visible=False),        # edit_panel
                    "", "",                          # draft_box, final_box empty
                    f"ℹ️ **General Q&A**",
                    gr.update(value=direct_resp, visible=True),   # need_info_md shown
                    gr.update(value="", visible=False),           # escalation_alert_md
                )

            return (
                run_id, trace, trace,
                gr.update(visible=False),            # login_panel
                "",                                  # pending_msg_state
                gr.update(visible=False),            # hitl_panel
                gr.update(visible=False),            # edit_panel
                "", direct_resp,
                f"{emoji} **Status: {status}**",
                gr.update(value="", visible=False),  # need_info_md hidden
                gr.update(value="", visible=False),  # escalation_alert_md
            )

        def on_login(username: str, password: str,
                     pending_msg: str, run_id: str, trace: str,
                     logged_in_patient: str):
            """Validate credentials then re-run the workflow with the patient prefix."""
            patient_name = _validate_login(username, password)

            if not patient_name:
                return (
                    logged_in_patient, run_id, trace,
                    gr.update(visible=True),   # login_panel stays open
                    gr.update(value="❌ Invalid credentials. "
                                    "Try your name without spaces (e.g. AlexMorgan) "
                                    "and password 123."),
                    gr.update(visible=False),  # logged_in_banner
                    gr.update(visible=False),  # patient_profile_panel
                    gr.update(),               # patient_profile_md
                    trace,                     # trace_box
                    gr.update(visible=False), gr.update(visible=False),
                    "", "",
                    "🔐 **Login failed — please try again.**",
                    gr.update(value="", visible=False),   # need_info_md
                    gr.update(value="", visible=False),   # escalation_alert_md
                )

            # Valid — re-run workflow with patient prefix
            run_id2, trace2, draft, needs_hitl, direct_resp = _run_first_pass(
                pending_msg, patient_name=patient_name
            )
            status  = _registry.get(run_id2, {}).get("last", {}).get("status", "")
            profile = _patient_profile(patient_name)
            banner  = f"✅ Logged in as: **{patient_name}**"

            if needs_hitl:
                return (
                    patient_name, run_id2, trace2,
                    gr.update(visible=False),                       # login_panel
                    gr.update(value=""),                            # login_error_md
                    gr.update(value=banner, visible=True),          # logged_in_banner
                    gr.update(visible=True),                        # patient_profile_panel
                    gr.update(value=profile),                       # patient_profile_md
                    trace2,                                         # trace_box
                    gr.update(visible=True), gr.update(visible=False),
                    draft, "",
                    "⏳ **Waiting for human review…**",
                    gr.update(value="", visible=False),             # need_info_md
                    gr.update(value="", visible=False),             # escalation_alert_md
                )

            emoji = {"ESCALATE": "🚨", "NEED_INFO": "❓", "READY": "✅"}.get(status, "ℹ️")

            if status == "NEED_INFO":
                # Show the NEED_INFO prompt inline above the input box
                return (
                    patient_name, run_id2, trace2,
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(value=banner, visible=True),
                    gr.update(visible=True),
                    gr.update(value=profile),
                    trace2,
                    gr.update(visible=False), gr.update(visible=False),
                    "", "",                                         # draft_box, final_box empty
                    f"❓ **Status: NEED_INFO**",
                    gr.update(value=direct_resp, visible=True),    # need_info_md shown
                    gr.update(value="", visible=False),            # escalation_alert_md
                )

            return (
                patient_name, run_id2, trace2,
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=banner, visible=True),
                gr.update(visible=True),
                gr.update(value=profile),
                trace2,
                gr.update(visible=False), gr.update(visible=False),
                "", direct_resp,
                f"{emoji} **Status: {status}**",
                gr.update(value="", visible=False),                # need_info_md hidden
                gr.update(value="", visible=False),                # escalation_alert_md
            )

        def on_logout():
            """Clear all session state and return to the fresh logged-out view."""
            return (
                "",                                   # logged_in_patient_state
                "",                                   # pending_msg_state
                "",                                   # run_id_state
                "",                                   # trace_state
                gr.update(value="", visible=False),   # logged_in_banner
                gr.update(visible=False),             # patient_profile_panel
                gr.update(visible=False),             # login_panel
                gr.update(value=""),                  # login_error_md
                "",                                   # trace_box
                gr.update(visible=False),             # hitl_panel
                gr.update(visible=False),             # edit_panel
                "",                                   # draft_box
                "",                                   # final_box
                "",                                   # status_display
                gr.update(value="", visible=False),   # need_info_md
                gr.update(value="", visible=False),   # escalation_alert_md
            )

        def on_approve(run_id: str, trace: str, logged_in_patient: str):
            # Read status BEFORE _resume overwrites registry["last"] with finalize state
            is_escalation = _registry.get(run_id, {}).get("last", {}).get("status") == "ESCALATE"
            trace_update, final = _resume(run_id, "approve", "")
            full_trace = trace + trace_update
            _apply_to_session(run_id)
            profile = _patient_profile(logged_in_patient)
            status_msg = ("🚨 **Status: ESCALATE** — Emergency protocol acknowledged."
                          if is_escalation else
                          "✅ **Status: READY** — Draft approved.")
            # For cancel/reschedule: show the confirmed response in the patient panel
            patient_panel = (gr.update()   # escalation: keep existing patient message
                             if is_escalation else
                             gr.update(value=final, visible=True))
            return (
                full_trace, full_trace,
                gr.update(visible=False), gr.update(visible=False),
                "", final,
                status_msg,
                gr.update(value=profile),                # patient_profile_md
                gr.update(value="", visible=False),      # escalation_alert_md (clear)
                patient_panel,                           # need_info_md
            )

        def on_edit_click():
            return gr.update(visible=True)

        def on_confirm_edit(run_id: str, trace: str, edited: str, logged_in_patient: str):
            # Read status BEFORE _resume overwrites registry["last"] with finalize state
            is_escalation = _registry.get(run_id, {}).get("last", {}).get("status") == "ESCALATE"
            trace_update, final = _resume(run_id, "edit", edited)
            full_trace = trace + trace_update
            _apply_to_session(run_id)
            profile = _patient_profile(logged_in_patient)
            status_msg = ("🚨 **Status: ESCALATE** — Emergency record updated and acknowledged."
                          if is_escalation else
                          "✅ **Status: READY** — Edited response sent.")
            # For cancel/reschedule: show the edited response in the patient panel
            patient_panel = (gr.update()   # escalation: keep existing patient message
                             if is_escalation else
                             gr.update(value=final, visible=True))
            return (
                full_trace, full_trace,
                gr.update(visible=False), gr.update(visible=False),
                "", final,
                status_msg,
                gr.update(value=profile),                # patient_profile_md
                gr.update(value="", visible=False),      # escalation_alert_md (clear)
                patient_panel,                           # need_info_md
            )

        def _apply_to_session(run_id: str) -> None:
            """Persist cancel/reschedule to the session file after HITL approval."""
            from src.tools.appointment_tools import _load, _save
            entry = _registry.get(run_id, {})
            last  = entry.get("last", {})
            intent        = last.get("intent")
            action_result = last.get("action_result") or {}
            if not action_result.get("success"):
                return

            scope   = action_result.get("scope")
            apt_id  = action_result.get("appointment_id")
            item_id = action_result.get("item_id")
            data    = _load()

            if intent == "cancel":
                for apt in data["appointments"]:
                    if apt["id"] != apt_id:
                        continue
                    if scope == "single_item" and item_id:
                        apt["items"] = [it for it in apt.get("items", [])
                                        if it["item_id"] != item_id]
                        if not apt["items"]:
                            apt["status"] = "cancelled"
                    else:
                        apt["status"] = "cancelled"
                        apt["items"] = []
                    break

            elif intent == "reschedule":
                new_date = str(action_result.get("new_date", "")).strip()
                if new_date and new_date.lower() not in ("", "a date of your choice"):
                    date_part, time_part = _split_date_time(new_date)
                    for apt in data["appointments"]:
                        if apt["id"] != apt_id:
                            continue
                        if scope == "single_item" and item_id:
                            for it in apt.get("items", []):
                                if it["item_id"] == item_id:
                                    it["date"] = date_part
                                    if time_part:
                                        it["time"] = time_part
                                    break
                        else:
                            apt["date"] = date_part
                            for it in apt.get("items", []):
                                it["date"] = date_part
                                if time_part:
                                    it["time"] = time_part
                        break

            _save(data)

        # ── Wire events ───────────────────────────────────────────────────

        _submit_outputs = [
            run_id_state, trace_state, trace_box,
            login_panel, pending_msg_state,
            hitl_panel, edit_panel,
            draft_box, final_box, status_display,
            need_info_md, escalation_alert_md,
        ]

        submit_btn.click(
            fn=on_submit,
            inputs=[msg_input, logged_in_patient_state],
            outputs=_submit_outputs,
        )
        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, logged_in_patient_state],
            outputs=_submit_outputs,
        )

        _login_outputs = [
            logged_in_patient_state, run_id_state, trace_state,
            login_panel, login_error_md,
            logged_in_banner, patient_profile_panel, patient_profile_md,
            trace_box,
            hitl_panel, edit_panel,
            draft_box, final_box, status_display,
            need_info_md, escalation_alert_md,
        ]

        login_btn.click(
            fn=on_login,
            inputs=[username_input, password_input,
                    pending_msg_state, run_id_state, trace_state,
                    logged_in_patient_state],
            outputs=_login_outputs,
        )
        password_input.submit(
            fn=on_login,
            inputs=[username_input, password_input,
                    pending_msg_state, run_id_state, trace_state,
                    logged_in_patient_state],
            outputs=_login_outputs,
        )

        logout_btn.click(
            fn=on_logout,
            inputs=[],
            outputs=[
                logged_in_patient_state, pending_msg_state,
                run_id_state, trace_state,
                logged_in_banner, patient_profile_panel, login_panel, login_error_md,
                trace_box, hitl_panel, edit_panel,
                draft_box, final_box, status_display,
                need_info_md, escalation_alert_md,
            ],
        )

        _hitl_outputs = [
            trace_state, trace_box,
            hitl_panel, edit_panel,
            draft_box, final_box, status_display,
            patient_profile_md, escalation_alert_md, need_info_md,
        ]

        approve_btn.click(
            fn=on_approve,
            inputs=[run_id_state, trace_state, logged_in_patient_state],
            outputs=_hitl_outputs,
        )

        edit_btn.click(
            fn=on_edit_click,
            inputs=[],
            outputs=[edit_panel],
        )

        confirm_edit_btn.click(
            fn=on_confirm_edit,
            inputs=[run_id_state, trace_state, edited_box, logged_in_patient_state],
            outputs=_hitl_outputs,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Appointment Assistance System Web UI…")
    print("Open: http://127.0.0.1:7860\n")
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=_CSS,
    )
