"""
Identity Gate Node
==================
Verifies that the requesting patient exists in the database.

Now positioned AFTER intent detection — reached only on cancel/reschedule paths.

Behaviour
---------
* Extracts patient name from the ``[Patient: name]`` prefix injected by
  app.py after a successful login.
* If the prefix is absent: the user is not logged in → returns LOGIN_REQUIRED
  so the app can display the login form.
* If the name is found in the database: sets ``identity_verified = True``
  and routes to the appropriate action node (reschedule or cancel).
* If the name is **not** found: sets ``identity_verified = False``, writes
  a client-facing rejection message, and ends the workflow immediately.

State fields updated
--------------------
identity_verified, patient_name (on success), status, final_response,
draft_response, hitl_approved (on failure/login-required), route_path
"""

import re

_REJECTION_RESPONSE = (
    "We were unable to verify your identity in our system. "
    "To protect patient privacy, appointment information and changes "
    "can only be accessed by verified patients.\n\n"
    "Please contact Halifax Medical Centre directly:\n"
    "📞 (902) 555-0000  |  ✉️ scheduling@halifaxmedical.example.com\n"
    "Or visit us in person at 1234 University Avenue, Halifax, NS, "
    "with a valid photo ID."
)


def identity_gate_node(state: dict) -> dict:
    """
    LangGraph node: verify patient identity before proceeding.
    """
    from src.tools.appointment_tools import look_up_appointment

    user_message = state.get("user_message", "")

    # Extract [Patient: <name>] prefix added by app.py
    match = re.match(r"\[Patient:\s*(.+?)\]", user_message)
    patient_name = match.group(1).strip() if match else ""

    if not patient_name:
        # No patient prefix → user is not logged in.
        # Identity gate is only reached for cancel/reschedule, so login is required.
        print("  [IdentityGate] No patient prefix — login required.")
        return {
            "identity_verified": False,
            "status":            "LOGIN_REQUIRED",
            "final_response":    "Please log in to manage your appointments.",
            "draft_response":    "Please log in to manage your appointments.",
            "hitl_approved":     False,
            "route_path":        ["identity_gate", "END"],
        }

    # Look up patient in the database
    result = look_up_appointment(patient_name=patient_name)

    if result.get("found"):
        print(f"  [IdentityGate] Identity verified ✓  patient={patient_name!r}")
        return {
            "identity_verified": True,
            "patient_name":      patient_name,
            "route_path":        ["identity_gate"],
        }
    else:
        print(f"  [IdentityGate] Identity FAILED ✗  patient={patient_name!r} not found in DB.")
        return {
            "identity_verified": False,
            "status":            "NEED_INFO",
            "final_response":    _REJECTION_RESPONSE,
            "draft_response":    _REJECTION_RESPONSE,
            "hitl_approved":     False,
            "route_path":        ["identity_gate", "END"],
        }
