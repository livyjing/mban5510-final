"""
ModelFallbackMiddleware
=======================
Wraps every LLM invocation with a transparent fallback chain so that a
transient failure of the primary model does not break the workflow.

Behaviour
---------
1. Invoke the *primary* model (default: ``gpt-4o-mini``).
2. On any exception, log the error and invoke the *fallback* model
   (default: ``gpt-3.5-turbo``).
3. If the fallback also fails the exception propagates (caught by
   ModelRetryMiddleware upstream if composed).

Design note
-----------
Both models are instantiated at construction time so that configuration
(temperature, API key) is resolved once rather than on every call.
The ``invoke`` method mirrors ``ChatOpenAI.invoke`` so callers treat this
class as a drop-in replacement.
"""

import os
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI


class ModelFallbackMiddleware:
    """
    Transparent LLM wrapper that falls back to a secondary model on failure.

    Parameters
    ----------
    primary_model   : ChatOpenAI, optional
        Primary model instance. Defaults to gpt-4o-mini.
    fallback_model  : ChatOpenAI, optional
        Fallback model instance. Defaults to gpt-3.5-turbo.
    """

    name = "ModelFallbackMiddleware"

    def __init__(
        self,
        primary_model: Optional[ChatOpenAI] = None,
        fallback_model: Optional[ChatOpenAI] = None,
    ):
        self.primary: ChatOpenAI = primary_model or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.fallback: ChatOpenAI = fallback_model or ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ------------------------------------------------------------------
    def invoke(self, messages: List[Any], **kwargs: Any) -> Any:
        """
        Invoke primary model; on failure fall back to the secondary model.

        Parameters
        ----------
        messages : list
            LangChain message objects (SystemMessage, HumanMessage, …).
        **kwargs
            Forwarded to the underlying ``ChatOpenAI.invoke`` call.
        """
        try:
            return self.primary.invoke(messages, **kwargs)
        except Exception as primary_err:
            print(
                f"  [{self.name}] Primary model "
                f"({self.primary.model_name}) failed: {primary_err}\n"
                f"  [{self.name}] Switching to fallback "
                f"({self.fallback.model_name})…"
            )
            return self.fallback.invoke(messages, **kwargs)
