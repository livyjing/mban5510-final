"""
Unit tests for middleware components.

Run with:  pytest tests/test_middleware.py -v
"""

import pytest


# ---------------------------------------------------------------------------
# PIIMiddleware tests
# ---------------------------------------------------------------------------

class TestPIIMiddleware:
    @pytest.fixture
    def pii(self):
        from src.middleware.pii import PIIMiddleware
        return PIIMiddleware()

    def test_masks_phone_number(self, pii):
        masked, detected, types = pii.mask("Call me at (902) 555-0101 please")
        assert "[PHONE]" in masked
        assert detected is True
        assert "PHONE" in types

    def test_masks_email(self, pii):
        masked, detected, types = pii.mask("Email me at patient@example.com")
        assert "[EMAIL]" in masked
        assert "EMAIL" in types

    def test_masks_ssn(self, pii):
        masked, detected, types = pii.mask("My SSN is 123-45-6789")
        assert "[SSN]" in masked
        assert "SSN" in types

    def test_no_pii_clean_message(self, pii):
        masked, detected, types = pii.mask("I need to reschedule my MRI")
        assert detected is False
        assert types == []
        assert masked == "I need to reschedule my MRI"

    def test_run_on_state(self, pii):
        state = {"user_message": "Call (902) 555-1234 for patient John"}
        result = pii.run_on_state(state)
        assert "user_message_masked" in result
        assert "[PHONE]" in result["user_message_masked"]
        assert result["pii_detected"] is True
        assert "PIIMiddleware" in result["route_path"]


# ---------------------------------------------------------------------------
# ModerationMiddleware tests
# ---------------------------------------------------------------------------

class TestModerationMiddleware:
    @pytest.fixture
    def mod(self):
        from src.middleware.moderation import ModerationMiddleware
        return ModerationMiddleware()

    def test_detects_emergency_chest_pain(self, mod):
        flagged, cats = mod._check_keywords("I have chest pain and can't breathe")
        assert flagged is True
        assert any("emergency" in c for c in cats)

    def test_detects_clinical_advice_request(self, mod):
        flagged, cats = mod._check_clinical("Can you diagnose what I have?")
        assert flagged is True

    def test_clean_scheduling_message(self, mod):
        kw_flag, _ = mod._check_keywords("I need to reschedule my blood work")
        cl_flag, _ = mod._check_clinical("I need to reschedule my blood work")
        assert kw_flag is False
        assert cl_flag is False

    def test_run_on_state_sets_escalate(self, mod):
        state = {
            "user_message": "I have chest pain",
            "user_message_masked": "I have chest pain",
        }
        result = mod.run_on_state(state)
        assert result["moderation_flagged"] is True
        assert result["intent"] == "escalate"


# ---------------------------------------------------------------------------
# LimitMiddleware tests
# ---------------------------------------------------------------------------

class TestLimitMiddleware:
    def test_model_limit_not_exceeded(self):
        from src.middleware.limits import ModelCallLimitMiddleware
        lim = ModelCallLimitMiddleware(max_calls=5)
        state = {"model_call_count": 3}
        lim.check(state)  # Should not raise

    def test_model_limit_exceeded(self):
        from src.middleware.limits import ModelCallLimitMiddleware
        lim = ModelCallLimitMiddleware(max_calls=5)
        state = {"model_call_count": 5}
        with pytest.raises(RuntimeError, match="Model call limit"):
            lim.check(state)

    def test_tool_limit_exceeded(self):
        from src.middleware.limits import ToolCallLimitMiddleware
        lim = ToolCallLimitMiddleware(max_calls=3)
        state = {"tool_call_count": 3}
        with pytest.raises(RuntimeError, match="Tool call limit"):
            lim.check(state)

    def test_increment(self):
        from src.middleware.limits import ModelCallLimitMiddleware
        lim = ModelCallLimitMiddleware()
        state = {"model_call_count": 2}
        update = lim.increment(state)
        assert update["model_call_count"] == 3


# ---------------------------------------------------------------------------
# RetryMiddleware tests
# ---------------------------------------------------------------------------

class TestRetryMiddleware:
    def test_succeeds_on_first_attempt(self):
        from src.middleware.retry import ModelRetryMiddleware
        retry = ModelRetryMiddleware(max_retries=3)
        call_count = {"n": 0}

        @retry.wrap
        def succeed():
            call_count["n"] += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count["n"] == 1

    def test_retries_on_failure_then_succeeds(self):
        from src.middleware.retry import ModelRetryMiddleware
        retry = ModelRetryMiddleware(max_retries=3, base_delay=0.0)
        call_count = {"n": 0}

        @retry.wrap
        def fail_twice():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("transient error")
            return "recovered"

        result = fail_twice()
        assert result == "recovered"
        assert call_count["n"] == 3

    def test_raises_after_max_retries(self):
        from src.middleware.retry import ModelRetryMiddleware
        retry = ModelRetryMiddleware(max_retries=2, base_delay=0.0)

        @retry.wrap
        def always_fail():
            raise RuntimeError("permanent error")

        with pytest.raises(RuntimeError, match="permanent error"):
            always_fail()


# ---------------------------------------------------------------------------
# AppointmentTools tests
# ---------------------------------------------------------------------------

class TestAppointmentTools:
    def test_look_up_by_id(self):
        from src.tools.appointment_tools import look_up_appointment
        result = look_up_appointment(appointment_id="APT-001")
        assert result["found"] is True
        assert result["appointment"]["id"] == "APT-001"

    def test_look_up_default_returns_scheduled(self):
        from src.tools.appointment_tools import look_up_appointment
        result = look_up_appointment()
        assert result["found"] is True

    def test_reschedule_success(self):
        from src.tools.appointment_tools import reschedule_appointment
        result = reschedule_appointment("APT-001", "2026-04-10")
        assert result["success"] is True
        assert result["new_date"] == "2026-04-10"

    def test_cancel_success(self):
        from src.tools.appointment_tools import cancel_appointment
        result = cancel_appointment("APT-002")
        assert result["success"] is True
        assert result["appointment_id"] == "APT-002"

    def test_prep_instructions_mri(self):
        from src.tools.appointment_tools import get_prep_instructions
        result = get_prep_instructions("MRI")
        assert result["found"] is True
        assert len(result["instructions"]) > 0

    def test_prep_instructions_fallback(self):
        from src.tools.appointment_tools import get_prep_instructions
        result = get_prep_instructions("unknown_procedure_xyz")
        assert "instructions" in result
        assert len(result["instructions"]) > 0
