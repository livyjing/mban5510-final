"""
Validate Info Node
==================
Guards every actionable intent (cancel, reschedule, prep_instructions) by
checking that the minimum required entities were extracted before any
appointment operation or identity check is attempted.

This prevents silent "default" behaviour such as selecting the first
scheduled appointment when no identification was provided.

Minimum required fields
-----------------------
reschedule       : (appointment_id | item_id | appointment_type)  AND  preferred_date
cancel           : (appointment_id | item_id | appointment_type)
prep_instructions: appointment_type

If any required field is absent
    → status = "NEED_INFO", final_response = targeted request for the missing info
    → route_path ends with "END" (workflow terminates; no login or action triggered)

If all required fields are present
    → pass-through: returns only {"route_path": ["validate_info"]}
    → downstream routing sends cancel/reschedule to identity_gate, prep to action node

State fields updated (failure path only)
-----------------------------------------
status, final_response, draft_response, hitl_approved, route_path
"""


# ---------------------------------------------------------------------------
# Validation helpers — guard against LLM-hallucinated entity values
# ---------------------------------------------------------------------------

import re as _re_id

_APT_ID_RE  = _re_id.compile(r'^APT-\d+$',         _re_id.IGNORECASE)
_ITEM_ID_RE = _re_id.compile(r'^APT-\d+-[A-Z]+$',  _re_id.IGNORECASE)


def _clean_apt_id(value) -> "str | None":
    """Accept only properly-formatted appointment group IDs (e.g. APT-001)."""
    if not value:
        return None
    return value if _APT_ID_RE.match(str(value).strip()) else None


def _clean_item_id(value) -> "str | None":
    """Accept only properly-formatted item IDs (e.g. APT-001-A)."""
    if not value:
        return None
    return value if _ITEM_ID_RE.match(str(value).strip()) else None


_VALID_APT_TYPES = frozenset({
    "mri", "ct scan", "ct", "blood work", "blood test", "blood",
    "ultrasound", "x-ray", "xray", "x ray",
    "colonoscopy", "echocardiogram", "mammogram",
})

# Words that signal a real date/time reference
_DATE_HINT_WORDS = frozenset({
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "tomorrow", "next", "today", "morning", "afternoon", "evening",
    "early", "late", "week", "month",
})


def _clean_apt_type(value) -> "str | None":
    """Accept only recognised medical procedure types; reject vague/hallucinated strings."""
    if not value:
        return None
    return value if value.lower() in _VALID_APT_TYPES else None


def _clean_pref_date(value) -> "str | None":
    """Accept only strings that look like a real date/time reference."""
    if not value:
        return None
    v = value.lower()
    if any(c.isdigit() for c in v):         # contains a number → real date
        return value
    words = set(v.replace(",", " ").replace("/", " ").split())
    if words & _DATE_HINT_WORDS:             # contains a date keyword → real date
        return value
    return None                             # vague string like "TBD", "soon", etc.


# ---------------------------------------------------------------------------


def validate_info_node(state: dict) -> dict:
    """LangGraph node: enforce minimum required information before proceeding."""
    intent    = state.get("intent", "")

    # Normalise: reject hallucinated / generic LLM-generated values
    apt_id    = _clean_apt_id(state.get("appointment_id"))
    item_id   = _clean_item_id(state.get("item_id"))
    apt_type  = _clean_apt_type(state.get("appointment_type"))
    pref_date = _clean_pref_date(state.get("preferred_date"))

    # True when the user has given some way to identify which appointment
    has_id = bool(apt_id or item_id or apt_type)
    missing = []

    if intent == "reschedule":
        if not has_id:
            missing.append(
                "**Which appointment to reschedule** — please provide:\n"
                "    - An appointment ID (e.g., `APT-001`) or item ID (e.g., `APT-001-A`), or\n"
                "    - The procedure type (e.g., *MRI*, *blood work*, *CT scan*)"
            )
        if not pref_date:
            missing.append(
                "**Your preferred new date or time** — "
                "e.g., *March 25*, *next Tuesday*, *early April*"
            )

    elif intent == "cancel":
        if not has_id:
            missing.append(
                "**Which appointment to cancel** — please provide:\n"
                "    - An appointment ID (e.g., `APT-001`) or item ID (e.g., `APT-001-A`), or\n"
                "    - The procedure type (e.g., *MRI*, *blood work*, *CT scan*)"
            )

    elif intent == "prep_instructions":
        if not apt_type:
            missing.append(
                "**The procedure type** — e.g., *MRI*, *CT scan*, *blood work*, "
                "*ultrasound*, *X-ray*, *colonoscopy*, *echocardiogram*, *mammogram*"
            )

    # ── All fields present → pass through ──────────────────────────────────
    if not missing:
        print("  [ValidateInfo] All required fields present — passing through.")
        # Write back normalised values so downstream nodes see clean data
        result: dict = {"route_path": ["validate_info"]}
        if apt_id    is not None: result["appointment_id"]   = apt_id
        if item_id   is not None: result["item_id"]          = item_id
        if apt_type  is not None: result["appointment_type"] = apt_type
        if pref_date is not None: result["preferred_date"]   = pref_date
        return result

    # ── Missing fields → build a targeted NEED_INFO response ─────────────
    items_md = "\n\n".join(f"  • {m}" for m in missing)

    if intent == "reschedule":
        intro = "I'd be happy to reschedule your appointment! To proceed, I need:"
    elif intent == "cancel":
        intro = "I can cancel your appointment. To proceed, I need:"
    else:
        intro = "I'd be happy to provide preparation instructions! Please specify:"

    # Adjust tip based on whether the user is already logged in
    import re as _re
    is_logged_in = bool(_re.match(r"\[Patient:", state.get("user_message", "")))
    if is_logged_in:
        tip = ("💡 *Your active appointments are shown in the panel on the left — "
               "use the Item ID (e.g., `APT-001-A`) to specify which one.*")
    else:
        tip = ("💡 *Tip: Log in first — your active appointments will be displayed "
               "in the panel on the left so you can find the correct ID.*")

    response = (
        f"{intro}\n\n"
        f"{items_md}\n\n"
        f"{tip}"
    )

    print(f"  [ValidateInfo] NEED_INFO — missing fields for intent={intent!r}: "
          f"{len(missing)} item(s)")
    return {
        "status":         "NEED_INFO",
        "final_response": response,
        "draft_response": response,
        "hitl_approved":  False,
        "route_path":     ["validate_info", "END"],
    }
