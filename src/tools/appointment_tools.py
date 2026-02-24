"""
Appointment Tools
=================
Data-access and action functions that represent the system's "tool layer."

New data structure (multi-item appointments)
--------------------------------------------
Each appointment (APT-XXX) groups one or more exam/procedure items:

    {
      "id": "APT-001",
      "patient_name": "Alex Morgan",
      "date": "2026-03-15",
      "items": [
        { "item_id": "APT-001-A", "type": "MRI",        "time": "9:00 AM",  ... },
        { "item_id": "APT-001-B", "type": "Blood Work", "time": "11:30 AM", ... }
      ]
    }

Lookup priority
---------------
1. item_id        – exact item  (e.g. "APT-001-A")
2. appointment_id – whole appointment (e.g. "APT-001")
3. patient_name   – first appointment matching name
4. appointment_type – first item across all appointments matching type
5. default        – first scheduled appointment
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DATA_FILE   = Path(__file__).resolve().parents[2] / "data" / "appointments.json"
_ACTIVE_FILE: Path = _DATA_FILE   # overridden to session file by app.py


def set_active_file(path: Path) -> None:
    """Point all reads/writes at a session copy instead of the master file."""
    global _ACTIVE_FILE
    _ACTIVE_FILE = Path(path)


def _load() -> dict:
    with open(_ACTIVE_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(data: dict) -> None:
    with open(_ACTIVE_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _items_summary(items: List[dict]) -> str:
    """Human-readable list, e.g. 'MRI (9:00 AM), Blood Work (11:30 AM)'"""
    return ", ".join(f"{it['type']} ({it['time']})" for it in items)


def _find_item_by_type(
    appointments: List[dict], apt_type: str
) -> Tuple[Optional[dict], Optional[dict]]:
    """Return (appointment, item) for the first item matching apt_type."""
    for apt in appointments:
        for item in apt.get("items", []):
            if apt_type.lower() in item["type"].lower() or \
               apt_type.lower() in item["description"].lower():
                return apt, item
    return None, None


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def look_up_appointment(
    appointment_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    appointment_type: Optional[str] = None,
    item_id: Optional[str] = None,
) -> Dict:
    """
    Locate an appointment (and optionally a specific item within it).

    Returns
    -------
    {
      "found": bool,
      "appointment":  { ...apt fields..., "items": [...] },
      "matched_item": { ...item fields... } | None,
      "error": str    # only when found=False
    }
    """
    data = _load()
    appointments: List[dict] = data.get("appointments", [])

    # 1. Exact item_id  (e.g. "APT-001-A")
    if item_id:
        for apt in appointments:
            for item in apt.get("items", []):
                if item["item_id"].upper() == item_id.upper():
                    return {"found": True, "appointment": apt, "matched_item": item}

    # 2. Appointment by ID — if type also given, narrow down to that item
    if appointment_id:
        for apt in appointments:
            if apt["id"].upper() == appointment_id.upper():
                if appointment_type:
                    # Try to match a specific item within this appointment
                    for item in apt.get("items", []):
                        if appointment_type.lower() in item["type"].lower() or \
                           appointment_type.lower() in item["description"].lower():
                            return {"found": True, "appointment": apt, "matched_item": item}
                # No type given (or type not matched) → whole appointment
                return {"found": True, "appointment": apt, "matched_item": None}

    # 3. Patient name
    if patient_name:
        matches = [a for a in appointments
                   if patient_name.lower() in a["patient_name"].lower()]
        if matches:
            return {"found": True, "appointment": matches[0], "matched_item": None}

    # 4. Appointment type — find first item with matching type
    if appointment_type:
        apt, item = _find_item_by_type(appointments, appointment_type)
        if apt:
            return {"found": True, "appointment": apt, "matched_item": item}

    # 5. Default: first scheduled appointment
    scheduled = [a for a in appointments if a["status"] == "scheduled"]
    if scheduled:
        return {"found": True, "appointment": scheduled[0], "matched_item": None}

    return {"found": False, "error": "No appointment found matching the provided criteria."}


def reschedule_appointment(
    appointment_id: str,
    new_date: Optional[str] = None,
    appointment_type: Optional[str] = None,
) -> Dict:
    """
    Simulate rescheduling.

    If appointment_type is given, reschedule only that item.
    Otherwise reschedule the entire appointment (all items).
    """
    result = look_up_appointment(
        appointment_id=appointment_id,
        appointment_type=appointment_type,
    )
    if not result["found"]:
        return {"success": False, "error": result.get("error", "Appointment not found.")}

    apt      = result["appointment"]
    item     = result["matched_item"]
    proposed = new_date or "a date of your choice"

    if item:
        # Single-item reschedule — use item-level date if available
        item_date = item.get("date", apt["date"])
        return {
            "success":          True,
            "appointment_id":   apt["id"],
            "patient_name":     apt["patient_name"],
            "scope":            "single_item",
            "item_id":          item["item_id"],
            "appointment_type": item["type"],
            "description":      item["description"],
            "old_date":         item_date,
            "old_time":         item["time"],
            "new_date":         proposed,
            "doctor":           item["doctor"],
            "location":         item["location"],
            "message": (
                f"The {item['type']} ({item['description']}) for {apt['patient_name']} "
                f"on {item_date} at {item['time']} has been rescheduled to {proposed}."
            ),
        }
    else:
        # Whole-appointment reschedule
        items_str = _items_summary(apt.get("items", []))
        return {
            "success":        True,
            "appointment_id": apt["id"],
            "patient_name":   apt["patient_name"],
            "scope":          "whole_appointment",
            "items_summary":  items_str,
            "old_date":       apt["date"],
            "new_date":       proposed,
            "message": (
                f"All items for appointment {apt['id']} ({items_str}) "
                f"for {apt['patient_name']} have been rescheduled from "
                f"{apt['date']} to {proposed}."
            ),
        }


def cancel_appointment(
    appointment_id: str,
    appointment_type: Optional[str] = None,
) -> Dict:
    """
    Simulate cancellation.

    If appointment_type is given, cancel only that item.
    Otherwise cancel the entire appointment (all items).
    """
    result = look_up_appointment(
        appointment_id=appointment_id,
        appointment_type=appointment_type,
    )
    if not result["found"]:
        return {"success": False, "error": result.get("error", "Appointment not found.")}

    apt  = result["appointment"]
    item = result["matched_item"]

    if item:
        return {
            "success":          True,
            "appointment_id":   apt["id"],
            "patient_name":     apt["patient_name"],
            "scope":            "single_item",
            "item_id":          item["item_id"],
            "appointment_type": item["type"],
            "description":      item["description"],
            "date":             apt["date"],
            "time":             item["time"],
            "doctor":           item["doctor"],
            "message": (
                f"The {item['type']} ({item['description']}) for {apt['patient_name']} "
                f"on {apt['date']} at {item['time']} with {item['doctor']} "
                f"has been cancelled."
            ),
        }
    else:
        items_str = _items_summary(apt.get("items", []))
        return {
            "success":        True,
            "appointment_id": apt["id"],
            "patient_name":   apt["patient_name"],
            "scope":          "whole_appointment",
            "items_summary":  items_str,
            "date":           apt["date"],
            "message": (
                f"All items for appointment {apt['id']} ({items_str}) "
                f"for {apt['patient_name']} on {apt['date']} have been cancelled."
            ),
        }


def get_prep_instructions(appointment_type: str) -> Dict:
    """Retrieve preparation instructions for a given procedure type."""
    data = _load()
    prep: dict = data.get("prep_instructions", {})

    for key, instructions in prep.items():
        if key.lower() in appointment_type.lower() or appointment_type.lower() in key.lower():
            return {"found": True, "appointment_type": key, "instructions": instructions}

    general = prep.get("General", [
        "Please contact our scheduling team for specific preparation instructions.",
        "Arrive 15 minutes before your scheduled appointment time.",
        "Bring your provincial health card and a valid photo ID.",
    ])
    return {"found": False, "appointment_type": "General", "instructions": general}


def get_clinic_info() -> Dict:
    return _load().get("clinic_info", {})
