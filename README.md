# Appointment Assistance System
### MBAN 5510 Final Project — Middleware-Driven Workflow with Human-in-the-Loop

> **Deadline:** February 25, 2026 | **Instructor:** Michael Zhang | **Course:** MBAN 5510

---

## Overview

This project implements a **stateful, middleware-driven appointment assistance system** using **LangGraph**. It is explicitly *not* a chatbot — it is an enterprise-grade workflow orchestration system that:

- Classifies patient intent via LLM
- Routes requests through a typed state machine
- Screens all input with safety/PII middleware before any LLM call
- Generates AI draft responses
- Pauses for **Human-in-the-Loop (HITL)** review before sending
- Emits a verifiable execution trace for every run

---

## Demo Video

> **LinkedIn Demo:** *(link to be added after recording)*

Scenarios demonstrated:
1. Normal flow — cancel appointment (HITL approve)
2. Normal flow — reschedule appointment (HITL edit)
3. Normal flow — preparation instructions
4. Escalation — emergency keyword detected
5. HITL behaviour — reviewer edits draft before sending

---

## Supported Request Types

| Intent | Example Patient Message |
|---|---|
| **Reschedule** | "I need to reschedule my MRI appointment" |
| **Cancel** | "Please cancel my appointment APT-002" |
| **Prep Instructions** | "What should I do before my CT scan?" |
| **Escalate** | "I have chest pain" → emergency response, no LLM draft |
| **Unknown** | Unclear message → clarification request, no HITL |

---

## System Architecture

```
START
  │
  ▼
pii_check           ← PIIMiddleware            [masks phone/email/SSN/DOB]
  │
  ▼
moderation_check    ← ModerationMiddleware      [keyword + OpenAI API safety]
  │
  ├─[flagged]─────► escalation ──────────────────────────────────────► END
  │
  ▼
intent_detect       ← LLM (gpt-4o-mini)         [classify + extract entities]
  │                   + ModelRetryMiddleware
  │                   + ModelFallbackMiddleware
  │                   + ModelCallLimitMiddleware
  │
  ├─[escalate]────► escalation ──────────────────────────────────────► END
  ├─[unknown]─────► need_info ───────────────────────────────────────► END
  │
  ├─[reschedule]──► reschedule ─┐
  ├─[cancel]──────► cancel ─────┤  ← ToolRetryMiddleware
  └─[prep]────────► prep ───────┘    + ToolCallLimitMiddleware
                               │
                               ▼
                        draft_generation   ← LLM + Retry + Fallback + Limit
                               │
                               ▼
                        human_review       ← HumanInTheLoopMiddleware
                        (interrupt/resume)     via LangGraph interrupt()
                               │
                               ▼
                           finalize         sets status = READY
                               │
                               ▼
                             END
```

### State Machine

Every node reads from and writes to a single `AppointmentState` TypedDict. Key fields:

| Field | Purpose |
|---|---|
| `run_id` | Unique execution identifier |
| `timestamp` | ISO-8601 start time |
| `route_path` | Append-only execution trace (all visited nodes) |
| `user_message_masked` | PII-redacted version for logging |
| `intent` | Classified intent |
| `moderation_flagged` | Safety screening result |
| `draft_response` | AI-generated draft |
| `final_response` | Post-HITL final output |
| `status` | Terminal status: `READY` / `NEED_INFO` / `ESCALATE` |
| `hitl_action` | Human decision: `approve` / `edit` |

---

## Middleware Components

All middleware is explicitly instantiated and invoked within the workflow. The table below maps each component to its role and integration point.

| Middleware | Role | Integration Point |
|---|---|---|
| **PIIMiddleware** | Detects and masks phone numbers, emails, SSNs, DOBs in patient messages before any log or LLM call | `pii_check` node (first node) |
| **ModerationMiddleware** | Two-layer safety screen: keyword emergency detection + OpenAI Moderation API; sets `intent=escalate` on flag | `moderation_check` node (second node) |
| **HumanInTheLoopMiddleware** | Pauses the graph via LangGraph `interrupt()`; waits for reviewer to approve or edit the draft before finalising | `human_review` node |
| **ModelRetryMiddleware** | Wraps LLM calls with exponential back-off (3 retries, 1s/2s/4s delays) | `intent_detect`, `draft_generation` nodes |
| **ModelFallbackMiddleware** | Falls back from `gpt-4o-mini` to `gpt-3.5-turbo` on primary model failure | `intent_detect`, `draft_generation` nodes |
| **ModelCallLimitMiddleware** | Enforces a 10-call-per-run ceiling on LLM invocations; raises `RuntimeError` if exceeded | Pre-flight check in `intent_detect`, `draft_generation` nodes |
| **ToolCallLimitMiddleware** | Enforces a 20-call-per-run ceiling on tool invocations | Pre-flight check in all action nodes |
| **ToolRetryMiddleware** | Wraps data-access tool calls with 2-retry logic | All appointment action nodes |

### Middleware–LangGraph State Integration

Each middleware component interacts with `AppointmentState` through clearly defined fields:

- **PIIMiddleware** → writes `user_message_masked`, `pii_detected`, `pii_types`
- **ModerationMiddleware** → writes `moderation_flagged`, `moderation_categories`, may override `intent`
- **Limit middlewares** → read `model_call_count` / `tool_call_count`; increment via state update
- **Retry middlewares** → write `retry_count` (tracked in state for observability)
- **HITL middleware** → reads `draft_response`; writes `final_response`, `hitl_approved`, `hitl_action`

The `route_path` field is an **append-only** list (LangGraph reducer: `operator.add`). Every node appends its label, producing a complete execution trace automatically.

---

## Terminal Status Definitions

| Status | Meaning |
|---|---|
| `READY` | Workflow completed normally; final response is ready to send |
| `NEED_INFO` | Intent could not be determined; patient asked to clarify |
| `ESCALATE` | Safety risk detected; patient directed to emergency services or clinical staff |

---

## Environment Setup

### Prerequisites

- Python **3.11** or higher
- An **OpenAI API key** (gpt-4o-mini access required)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Final2

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### `.env` configuration

```
OPENAI_API_KEY=sk-your-api-key-here
```

> **Security note:** The `.gitignore` excludes `.env`. Never commit your API key.

---

## How to Run the CLI

### Interactive mode (recommended for demos)

```bash
python main.py
```

You will be prompted for a patient message. After the AI generates a draft, you will be asked to **approve** or **edit** it before the final response is printed.

### Single-message mode

```bash
# Reschedule
python main.py -m "I need to reschedule my MRI appointment"

# Cancel
python main.py -m "Please cancel my appointment APT-002"

# Preparation instructions
python main.py -m "What should I do to prepare for a CT scan?"

# Escalation (emergency)
python main.py -m "I have severe chest pain and can't breathe"

# Non-interactive (auto-approve HITL, useful for testing)
python main.py -m "Cancel my appointment" --auto-approve
```

### Expected output structure

```
──────────────────────────────────────────────────────────────
  Appointment Assistance System  |  Run: run_a1b2c3d4
  2026-02-22T14:30:00
──────────────────────────────────────────────────────────────
  Patient message: 'Cancel my MRI appointment'
──────────────────────────────────────────────────────────────

  [PIIMiddleware] No PII detected.
  [ModerationMiddleware] Content cleared.
  [IntentDetection] intent='cancel'  type='MRI'  id=None  confidence=high
  [Router] intent='cancel' → node='cancel'
  [CancelNode] Appointment APT-001 (MRI - Brain) ... has been cancelled.
  [DraftGeneration] Draft generated (142 chars).

==============================================================
  HUMAN REVIEW REQUIRED  [HumanInTheLoopMiddleware]
==============================================================
    Dear Alex, your MRI appointment on March 15 ...
  Options:
    [A] Approve
    [E] Edit

  [HumanReview] Reviewer approved the draft.

==============================================================
  EXECUTION TRACE
==============================================================
  Run ID    : run_a1b2c3d4
  Timestamp : 2026-02-22T14:30:00
  Status    : READY
  Intent    : cancel
  PII       : No PII detected
  HITL      : approve
  Route     : START → PIIMiddleware → ModerationMiddleware →
              intent_detect → cancel → draft_generation →
              human_review → hitl_approve → finalize → END
==============================================================

==============================================================
  FINAL RESPONSE
==============================================================
  Dear Alex, your MRI appointment (APT-001) ...
==============================================================
```

---

## HITL Workflow Explanation

Human-in-the-Loop is implemented using **LangGraph's `interrupt()` / `Command(resume=…)` mechanism**:

### Step-by-step

1. **Draft generation** — the `draft_generation_node` calls the LLM and stores the result in `state["draft_response"]`.

2. **Interrupt** — the `human_review_node` calls `interrupt({"type": "human_review", "draft": draft})`. LangGraph serialises the full workflow state to the `MemorySaver` checkpoint and raises `NodeInterrupt`, which ends the stream gracefully.

3. **Human review** — `main.py` detects the interrupted state by checking `workflow.get_state(config).next`. It then:
   - Displays the draft to the reviewer
   - Waits for `[A]pprove` or `[E]dit` input
   - Collects an edited response if chosen

4. **Resume** — `main.py` calls `workflow.stream(Command(resume=decision), config)`. Execution resumes inside `human_review_node`; `interrupt()` returns the `decision` dict.

5. **Finalise** — the node writes `final_response` (either the original draft or the edited version), sets `hitl_action`, and the graph proceeds to `finalize` → `END`.

### Key guarantee

> *"Final response must reflect human review input."* — the `final_response` field is **only ever set by the HITL node**, never by draft generation directly. The draft never becomes the final response without passing through human review.

### Escalation bypass

Risk-flagged messages skip the HITL loop entirely. The `escalation_node` writes a fixed safety response directly to `final_response`. This prevents human reviewers from inadvertently approving an inappropriate response to an emergency.

---

## Design Decisions

### 1. Middleware as dedicated graph nodes vs. transparent wrappers

**Decision:** PII and Moderation middleware run as the first two nodes in the graph. Retry, Limit, and Fallback middleware run *inside* nodes as function wrappers.

**Rationale:** Safety gates (PII masking, moderation) must be visible in the `route_path` trace and must execute before *any* LLM call. Making them explicit nodes makes this guarantee architectural rather than conventional. Retry/Limit/Fallback are operational concerns best expressed as decorators around individual LLM/tool calls.

### 2. `route_path` as an append-only reducer

**Decision:** `route_path: Annotated[List[str], operator.add]` uses LangGraph's reducer mechanism.

**Rationale:** Each node only needs to return `["its_label"]`; LangGraph automatically appends it to the existing list. This prevents nodes from needing to read and rewrite the full trace, and ensures the trace is accurate even when the graph is resumed after an interrupt.

### 3. No HITL for escalation or NEED_INFO paths

**Decision:** `escalation_node` and `need_info_node` write directly to `final_response` without triggering `interrupt()`.

**Rationale:** Emergency responses must be deterministic and immediate — introducing a review delay for a safety escalation would be counterproductive. Clarification requests are also sufficiently templated that human review adds minimal value.

### 4. PII masking uses masked message for LLM calls

**Decision:** The intent detection and draft generation LLM calls use `user_message_masked`, not `user_message`.

**Rationale:** Ensures PII is never sent to external APIs. The LLM still receives all scheduling-relevant information (appointment types, dates, action verbs) because those are not masked.

### 5. Mock data over live database

**Decision:** `data/appointments.json` serves as a simulated appointment backend.

**Rationale:** The project evaluates workflow and middleware design, not database integration. A JSON backend keeps the system self-contained and runnable without external dependencies.

---

## Project Structure

```
Final2/
├── main.py                          # CLI entry point
├── requirements.txt
├── .env.example                     # Environment variable template
├── .gitignore
├── README.md
├── data/
│   └── appointments.json            # Mock appointment + prep-instructions data
└── src/
    ├── state.py                     # AppointmentState TypedDict
    ├── workflow.py                  # LangGraph graph assembly & routing
    ├── middleware/
    │   ├── pii.py                   # PIIMiddleware
    │   ├── moderation.py            # ModerationMiddleware
    │   ├── retry.py                 # ModelRetryMiddleware, ToolRetryMiddleware
    │   ├── limits.py                # ModelCallLimitMiddleware, ToolCallLimitMiddleware
    │   └── fallback.py              # ModelFallbackMiddleware
    ├── nodes/
    │   ├── pii_check.py             # Node: PIIMiddleware gate
    │   ├── moderation_check.py      # Node: ModerationMiddleware gate
    │   ├── intent_detection.py      # Node: LLM intent classification
    │   ├── appointment_actions.py   # Nodes: reschedule / cancel / prep
    │   ├── draft_generator.py       # Node: LLM draft generation
    │   ├── human_review.py          # Node: HumanInTheLoopMiddleware (interrupt)
    │   ├── finalize.py              # Node: set READY status
    │   ├── escalation.py            # Node: safety escalation response
    │   └── need_info.py             # Node: clarification request
    └── tools/
        └── appointment_tools.py     # Data-access functions (lookup/reschedule/cancel/prep)
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific scenario
pytest tests/test_workflow.py -v -k "test_cancel"
```

*(Tests directory is included in the repository.)*

---

## Assumptions

1. The system operates in a **single-patient context** per run. There is no session management across multiple turns.
2. Appointment data is **mocked** (`data/appointments.json`). In production, the tool layer would call a real scheduling API.
3. PII masking is **regex-based**. A production system would use a dedicated NER service (e.g., AWS Comprehend Medical).
4. The OpenAI Moderation API call in `ModerationMiddleware` may be skipped gracefully if the API is unavailable; the keyword layer still runs.
5. The `preferred_date` for reschedule is passed as a free-text string from the LLM extraction. In production, date parsing and availability checking would be implemented.

---

## Security Notes

- API keys are managed via `.env` (never committed).
- PII is masked **before** any LLM call or log write.
- Log output uses `user_message_masked`, not the raw message.
- No clinical advice is generated; all medical questions trigger escalation.

---

*Submitted for MBAN 5510 — Advanced Business Analytics, Saint Mary's University, Sobey School of Business.*
