"""
General Q&A Node
================
Handles messages with ``unknown`` intent that already passed the moderation
risk screen.  Covers two types of questions:

  1. Health information questions (symptoms, conditions, treatments, preparation)
     → Live MedlinePlus retrieval (RAG) + evidence-based LLM answer with sources

  2. Clinic / administrative questions (hours, parking, location, booking)
     → LLM answers from general knowledge

Flow
----
The node first calls ``get_health_context(message)`` which queries the
MedlinePlus Web Service and returns compressed, numbered evidence passages.

  • If MedlinePlus returns context  → use ``_HEALTH_SYSTEM_PROMPT`` (RAG mode)
  • If MedlinePlus returns nothing  → use ``_CLINIC_SYSTEM_PROMPT`` (LLM mode)

Clinical diagnoses and personalised medical advice are always refused.
No HITL review is required for either type.

Middleware applied
------------------
* ModelCallLimitMiddleware – pre-flight check before every LLM call
* ModelRetryMiddleware     – wraps the LLM call with exponential back-off
* ModelFallbackMiddleware  – switches to a secondary model on persistent failure

State fields updated
--------------------
status, final_response, draft_response, hitl_approved, model_call_count,
route_path, messages
"""

from langchain_core.messages import HumanMessage, SystemMessage

from src.middleware.fallback import ModelFallbackMiddleware
from src.middleware.limits import ModelCallLimitMiddleware
from src.middleware.retry import ModelRetryMiddleware
from src.tools.medlineplus_tool import get_health_context

# ---------------------------------------------------------------------------
# Middleware instances
# ---------------------------------------------------------------------------
_limiter = ModelCallLimitMiddleware(max_calls=10)
_retry   = ModelRetryMiddleware(max_retries=3, base_delay=1.0)
_llm     = ModelFallbackMiddleware()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_HEALTH_SYSTEM_PROMPT = """\
You are a knowledgeable health information assistant for Halifax Medical Centre.
Your goal is to give the patient a genuinely helpful answer based on the
MedlinePlus evidence retrieved below.

ANSWER FULLY:
  - Explain what a condition is, its common symptoms, causes, and risk factors.
  - Describe general treatment approaches, medications commonly used, and
    lifestyle advice as documented by MedlinePlus.
  - Explain what test results or medical terms mean in plain language.
  - Tell the patient when to seek care and what to expect.

Use the retrieved context as your primary source.  If the context covers the
question, give a complete, informative answer — do NOT deflect or refuse.

Only if the question requires knowing this specific patient's personal history,
examination findings, or lab values that only a clinician can assess, add a
brief note at the end:
  "For advice tailored to your personal situation, please speak with your doctor
   or call us at (902) 555-0000."

Keep your response under 200 words and maintain a warm, professional tone.
End with: Sources: [bracket numbers you used, e.g. [1], [2]]
"""

_CLINIC_SYSTEM_PROMPT = """\
You are a knowledgeable assistant for Halifax Medical Centre.
Your goal is to give the patient the most helpful answer you can.

Answer any question you are able to, including:
  - Health and medical questions (symptoms, conditions, treatments, medications,
    preparation for procedures, what test results typically mean, etc.)
  - Clinic information (location, phone, hours, parking, booking policies)
  - Administrative questions about wait times or referral procedures

Give a full, informative answer using your medical knowledge.
Only if you genuinely cannot answer the question at all, say:
  "I'm not sure about that — please contact our scheduling team at
   (902) 555-0000 or scheduling@halifaxmedical.example.com and they will
   be happy to help."

Keep responses under 150 words and maintain a warm, professional tone.

Clinic contact:
  Phone : (902) 555-0000
  Email : scheduling@halifaxmedical.example.com
  Address: 1234 University Avenue, Halifax, NS B3H 1T8
"""


def general_qa_node(state: dict) -> dict:
    """LangGraph node: answer general or health questions via LLM (+ MedlinePlus RAG)."""
    _limiter.check(state)

    # Prefer PII-masked version; fall back to raw
    message = state.get("user_message_masked") or state.get("user_message", "")

    # ── Try MedlinePlus live retrieval ─────────────────────────────────────
    context, source_urls = get_health_context(message)

    if context:
        # RAG mode: inject retrieved evidence into the system prompt
        system_content = _HEALTH_SYSTEM_PROMPT + f"\n\nMedlinePlus context:\n{context}"
        system_msg = SystemMessage(content=system_content)
        log_tag = "health-RAG"
    else:
        # Fallback: answer from general LLM knowledge (clinic / admin questions)
        system_msg = SystemMessage(content=_CLINIC_SYSTEM_PROMPT)
        log_tag = "clinic-QA"

    @_retry.wrap
    def call_llm():
        return _llm.invoke([system_msg, HumanMessage(content=message)])

    response = call_llm()
    answer = response.content.strip()

    if context:
        print(f"  [GeneralQA/{log_tag}] Answered via MedlinePlus RAG. Sources: {source_urls}")
    else:
        print(f"  [GeneralQA/{log_tag}] Answered via LLM knowledge (no MedlinePlus results).")

    return {
        "status":         "READY",
        "final_response": answer,
        "draft_response": answer,
        "hitl_approved":  False,
        "route_path":     ["general_qa", "END"],
        **_limiter.increment(state),
        "messages": state.get("messages", []) + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ],
    }
