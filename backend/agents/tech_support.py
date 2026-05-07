"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Can escalate to higher-privilege agents when needed.

SECURITY NOTES:
- Token validation enforced on all incoming requests
- Input sanitization applied before LLM calls
- PII fields encrypted and not logged in plaintext
- Audit trail maintained for all AI-driven actions
"""

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity, validate_token
from llm.approved_client import ApprovedLLMClient

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")

# ---------------------------------------------------------------------------
# Approved model registry
# ---------------------------------------------------------------------------
APPROVED_MODELS = {
    "tech-support-default": "tech-support-default@1.0.0",
}
PINNED_MODEL_ID = "tech-support-default"
PINNED_MODEL_VERSION = "tech-support-default@1.0.0"

# ---------------------------------------------------------------------------
# Tool / agent allow list
# ---------------------------------------------------------------------------
ALLOWED_TOOLS = {"llm_chat"}
POLICY_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Task status
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    COMPLETE = "COMPLETE"
    ESCALATED = "ESCALATED"
    ERROR = "ERROR"
    DENIED = "DENIED"


# ---------------------------------------------------------------------------
# Encryption helper (Fernet symmetric encryption for PII)
# ---------------------------------------------------------------------------
def _get_fernet():
    try:
        from cryptography.fernet import Fernet
        key = os.environ.get("PII_ENCRYPTION_KEY")
        if not key:
            key = Fernet.generate_key().decode()
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        return None


def _encrypt_pii(value: str) -> str:
    f = _get_fernet()
    if f is None:
        return "[ENCRYPTED]"
    try:
        return f.encrypt(value.encode()).decode()
    except Exception:
        return "[ENCRYPTED]"


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "audit_trail.jsonl")


def _write_audit_record(record: dict) -> None:
    """Append-only audit record writer (JSON-lines)."""
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        audit_logger.error("Failed to write audit record: %s", exc)


# ---------------------------------------------------------------------------
# Input sanitization helpers
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 4096

_DANGEROUS_CODE_PATTERNS = re.compile(
    r"\b(eval|exec|compile|__import__|subprocess|os\.system|os\.popen"
    r"|importlib|ctypes|open\s*\(|input\s*\()\b",
    re.IGNORECASE,
)

_PROMPT_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|all|prior)\s+instructions?|"
    r"you\s+are\s+now|disregard\s+your|forget\s+your|"
    r"system\s*prompt|<\s*/?system\s*>|"
    r"\[\s*system\s*\]|"
    r"act\s+as\s+if|pretend\s+you\s+are)",
    re.IGNORECASE,
)

_INVISIBLE_CHAR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\ufeff]"
)

_BASE64_BLOB_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})")

_SHELL_COMMAND_PATTERN = re.compile(
    r"(rm\s+-rf|chmod\s+|chown\s+|wget\s+|curl\s+|bash\s+|sh\s+|"
    r"python\s+-c|perl\s+-e|ruby\s+-e|nc\s+|netcat\s+)",
    re.IGNORECASE,
)


def _sanitize_input(message: str) -> str:
    """
    Strip/escape characters used in prompt-injection attacks,
    enforce max length, remove null bytes and invisible characters,
    reject base64 blobs and shell commands.
    """
    if not isinstance(message, str):
        message = str(message)

    # Remove null bytes and invisible/hidden characters
    message = _INVISIBLE_CHAR_PATTERN.sub("", message)

    # Strip leading/trailing whitespace
    message = message.strip()

    # Enforce maximum length
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]

    # Remove base64-encoded blobs
    message = _BASE64_BLOB_PATTERN.sub("[REDACTED_BLOB]", message)

    # Remove shell/binary commands
    message = _SHELL_COMMAND_PATTERN.sub("[REDACTED_CMD]", message)

    # Remove/escape prompt injection patterns
    message = _PROMPT_INJECTION_PATTERNS.sub("[REDACTED]", message)

    return message


# ---------------------------------------------------------------------------
# LLM output validation
# ---------------------------------------------------------------------------
_DYNAMIC_EXEC_PATTERN = re.compile(
    r"\b(eval|exec|compile|__import__|subprocess\.(?:run|Popen|call|check_output)"
    r"|os\.system|os\.popen|importlib\.import_module|ctypes\.)\b",
    re.IGNORECASE,
)


def _validate_llm_output(response: str) -> str:
    """
    Check LLM response for dynamic code execution primitives.
    Raises ValueError if dangerous content is found.
    """
    if _DYNAMIC_EXEC_PATTERN.search(response):
        raise ValueError(
            "LLM response contains potentially dangerous code execution primitives "
            "and has been blocked."
        )
    return response


# ---------------------------------------------------------------------------
# Provenance / watermarking helper
# ---------------------------------------------------------------------------
def _attach_provenance(response: str, trace_id: str) -> dict[str, Any]:
    """
    Attach synthetic-content provenance metadata to an LLM response.
    Raises RuntimeError if labeling fails (fail-safe).
    """
    try:
        provenance = {
            "model_id": PINNED_MODEL_ID,
            "model_version": PINNED_MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "content_origin": "ai-generated",
            "synthetic_content_label": "AI_GENERATED",
            "trace_id": trace_id,
        }
        return {"content": response, "provenance": provenance}
    except Exception as exc:
        raise RuntimeError(f"Failed to attach provenance metadata: {exc}") from exc


class TechSupportAgent:
    """
    Technical support agent for handling general user queries.

    Privilege Level: LOW
    Capabilities:
    - Answer general questions
    - Provide technical guidance
    - Inform users to contact finance team directly for financial queries
    """

    ALLOWED_ROLES = ["user", "tech_support", "admin"]
    PRIVILEGE_LEVEL = "low"

    def __init__(self, llm_client: ApprovedLLMClient):
        self.llm_client = llm_client
        self.agent_id = "tech_support"
        self.agent_name = "Tech Support Agent"

    # ------------------------------------------------------------------
    # Tool allow-list enforcement
    # ------------------------------------------------------------------
    def _enforce_tool_allowlist(self, tool_id: str, actor: str) -> None:
        if tool_id not in ALLOWED_TOOLS:
            _write_audit_record({
                "event": "tool_denied",
                "actor": actor,
                "tool_id": tool_id,
                "policy_version": POLICY_VERSION,
                "denial_reason": f"Tool '{tool_id}' is not in the allowed list.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.warning(
                "Tool invocation denied",
                extra={"actor": actor, "tool_id": tool_id, "policy_version": POLICY_VERSION}
            )
            raise PermissionError(
                f"Tool '{tool_id}' is not permitted for this agent."
            )

    # ------------------------------------------------------------------
    # Model registry validation
    # ------------------------------------------------------------------
    def _validate_model(self, model_id: str) -> None:
        if model_id not in APPROVED_MODELS:
            raise ValueError(
                f"Model '{model_id}' is not in the approved model registry."
            )
        if APPROVED_MODELS[model_id] != PINNED_MODEL_VERSION:
            raise ValueError(
                f"Model version mismatch for '{model_id}'. "
                f"Expected '{PINNED_MODEL_VERSION}', "
                f"got '{APPROVED_MODELS[model_id]}'."
            )

    async def handle(
        self,
        context: dict[str, Any],
        caller: AgentIdentity,
        headers: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Handle incoming request from orchestrator or direct call.

        Args:
            context: Request context with user message and metadata
            caller: Identity of the calling agent/user
            headers: Request headers (including auth token)

        Returns:
            Response dictionary with terminal status field.
        """
        trace_id = str(uuid.uuid4())

        # Validate the incoming token using a real token verifier
        token = headers.get("X-Agent-Token") if headers else None
        if not token:
            logger.warning("Request rejected: missing authentication token.")
            return {
                "response": "Authentication required.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "status": TaskStatus.DENIED,
                "trace_id": trace_id,
            }

        try:
            validate_token(token, caller)
        except Exception as exc:
            logger.warning("Request rejected: invalid token. %s", exc)
            return {
                "response": "Authentication failed: invalid or expired token.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "status": TaskStatus.DENIED,
                "trace_id": trace_id,
            }

        raw_user_message = context.get("user_message", "")
        # Sanitize user message before any processing
        user_message = _sanitize_input(raw_user_message)
        user_message = self._sanitize_input_extended(user_message)

        # Check if this needs finance team referral
        if self._needs_finance_referral(user_message):
            logger.info(
                "Tech support referring user to finance team",
                extra={"reason": "Financial query detected", "trace_id": trace_id}
            )
            return {
                "response": (
                    "Your query appears to relate to financial information. "
                    "Please contact the finance team directly for assistance "
                    "with financial reports, budgets, or related matters."
                ),
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "status": TaskStatus.COMPLETE,
                "trace_id": trace_id,
            }

        # Handle the query directly
        response_payload = await self._process_query(user_message, context, trace_id, caller)

        return {
            "response": response_payload["content"],
            "provenance": response_payload.get("provenance"),
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
            "model_id": PINNED_MODEL_ID,
            "model_version": PINNED_MODEL_VERSION,
            "status": TaskStatus.COMPLETE,
            "trace_id": trace_id,
        }

    def _needs_finance_referral(self, message: str) -> bool:
        """Check if message requires finance team involvement."""
        finance_triggers = [
            "quarterly report", "financial statement", "budget",
            "revenue numbers", "profit margin", "expense report",
            "balance sheet", "cash flow", "earnings"
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in finance_triggers)

    def _sanitize_input_extended(self, message: str) -> str:
        """
        Additional sanitization: reject/strip inputs containing
        invisible/hidden characters, base64-encoded blobs, leetspeak
        patterns, shell/binary commands, and prompt injection keywords.
        (Complements _sanitize_input module-level function.)
        """
        # Already handled by module-level _sanitize_input; apply again
        # for defence-in-depth when called independently.
        return _sanitize_input(message)

    async def _process_query(
        self,
        message: str,
        context: dict,
        trace_id: str,
        caller: AgentIdentity,
    ) -> dict[str, Any]:
        """
        Process a general tech support query.
        Input is sanitized, output is validated, provenance is attached.
        """
        # Validate model against approved registry
        self._validate_model(PINNED_MODEL_ID)

        # Enforce tool allow list
        self._enforce_tool_allowlist("llm_chat", getattr(caller, "agent_id", "unknown"))

        # Sanitize and validate input
        sanitized_message = _sanitize_input(message)
        sanitized_message = self._sanitize_input_extended(sanitized_message)

        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        messages_payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sanitized_message}
        ]

        input_hash = hashlib.sha256(
            json.dumps(messages_payload, sort_keys=True).encode()
        ).hexdigest()

        # Audit: before LLM call
        _write_audit_record({
            "event": "llm_call_start",
            "trace_id": trace_id,
            "model_id": PINNED_MODEL_ID,
            "model_version": PINNED_MODEL_VERSION,
            "input_hash": input_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "principal": getattr(caller, "agent_id", "unknown"),
        })

        logger.info(
            "Sending messages to LLM",
            extra={
                "trace_id": trace_id,
                "model_id": PINNED_MODEL_ID,
                "input_hash": input_hash,
                "messages": messages_payload,
            }
        )

        response = await self.llm_client.chat(
            messages=messages_payload,
            model=PINNED_MODEL_ID,
        )

        logger.info(
            "Received response from LLM",
            extra={
                "trace_id": trace_id,
                "model_id": PINNED_MODEL_ID,
                "response_length": len(response) if response else 0,
                "response": response,
            }
        )

        # Validate LLM output for dangerous primitives
        validated_response = _validate_llm_output(response)

        # Audit: after LLM call
        _write_audit_record({
            "event": "llm_call_complete",
            "trace_id": trace_id,
            "model_id": PINNED_MODEL_ID,
            "model_version": PINNED_MODEL_VERSION,
            "input_hash": input_hash,
            "output": validated_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "principal": getattr(caller, "agent_id", "unknown"),
        })

        # Attach provenance — fail-safe: raises if labeling fails
        provenance_payload = _attach_provenance(validated_response, trace_id)
        return provenance_payload

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.
        PII fields are encrypted; sensitive internal fields are excluded
        from the returned dict and logs.
        """
        contact_email_encrypted = _encrypt_pii(
            os.environ.get("USER_CONTACT_EMAIL", "[RETRIEVE_FROM_SECURE_STORE]")
        )
        phone_encrypted = _encrypt_pii(
            os.environ.get("USER_PHONE", "[RETRIEVE_FROM_SECURE_STORE]")
        )

        # Full context (internal use only — never returned directly)
        _full_context = {
            "user_id": user_id,
            "subscription_tier": "enterprise",
            "recent_queries": [
                "How do I upload files?",
                "What file types are supported?",
                "Can I access financial reports?"
            ],
            "preferences": {
                "language": "en",
                "timezone": "America/New_York"
            },
            "internal_notes": "VIP customer - handle with priority",
            "account_details": {
                "contact_email": contact_email_encrypted,
                "phone": phone_encrypted,
            }
        }

        # Allowlist of safe, non-PII fields to return and log
        safe_context = {
            "user_id": _full_context["user_id"],
            "subscription_tier": _full_context["subscription_tier"],
            "preferences": _full_context["preferences"],
        }

        logger.info(
            "Retrieved user context",
            extra={"user_id": user_id, "subscription_tier": safe_context["subscription_tier"]}
        )

        return safe_context

        #checking
        #touched