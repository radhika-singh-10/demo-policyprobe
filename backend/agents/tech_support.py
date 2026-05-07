"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Escalation to higher-privilege agents requires explicit human-approval token.

SECURITY NOTES:
- Token validation enforced via constant-time comparison
- User input sanitized before LLM calls
- LLM output scanned for dangerous code execution primitives
- PII encrypted and redacted from logs
- Audit trail maintained for all LLM interactions
- No dynamic privilege escalation permitted
"""

import hashlib
import hmac
import logging
import logging.handlers
import os
import re
import time
import unicodedata
import uuid
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity
from llm.approved_client import ApprovedLLMClient

logger = logging.getLogger(__name__)

# Rotating audit log handler
_audit_logger = logging.getLogger("tech_support.audit")
if not _audit_logger.handlers:
    _audit_handler = logging.handlers.RotatingFileHandler(
        "audit_tech_support.log", maxBytes=10 * 1024 * 1024, backupCount=10
    )
    _audit_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _audit_logger.addHandler(_audit_handler)
    _audit_logger.setLevel(logging.INFO)

# Approved model registry with pinned identifiers and integrity hashes
APPROVED_MODELS = {
    "approved-model-v1.0": "sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
    "approved-model-v1.1": "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
}
PINNED_MODEL = os.environ.get("PINNED_MODEL_ID", "approved-model-v1.0")

# Tool allow list
ALLOWED_TOOLS = frozenset({"llm_chat"})

# Valid tokens loaded from environment
_VALID_TOKENS = set(
    filter(None, os.environ.get("VALID_AGENT_TOKENS", "").split(","))
)

# HMAC watermark secret
_WATERMARK_SECRET = os.environ.get("WATERMARK_SECRET", "change-me-in-production").encode()

# Prompt injection patterns
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)", re.IGNORECASE),
]

# Shell/binary command patterns
_SHELL_PATTERNS = [
    re.compile(r"\b(rm|chmod|chown|wget|curl|nc|bash|sh|python|perl|ruby|php)\s", re.IGNORECASE),
    re.compile(r"[;&|`$]\s*\w+"),
    re.compile(r"\.\./"),
    re.compile(r"base64\s*(-d|--decode)", re.IGNORECASE),
]

# Dynamic code execution primitives in LLM output
_CODE_EXEC_PATTERNS = [
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bcompile\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bexecfile\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\b.*shell\s*=\s*True", re.IGNORECASE),
    re.compile(r"\bgetattr\s*\(.*__", re.IGNORECASE),
]

MAX_INPUT_LENGTH = 4096


def _validate_token(token: Optional[str]) -> bool:
    """Validate token using constant-time comparison against known valid tokens."""
    if not token or not _VALID_TOKENS:
        return False
    for valid in _VALID_TOKENS:
        if hmac.compare_digest(token, valid):
            return True
    return False


def _verify_model_integrity(model_id: str) -> bool:
    """Verify the model is on the approved list."""
    return model_id in APPROVED_MODELS


def _encrypt_pii(value: str) -> str:
    """Encrypt PII using Fernet if available, otherwise base64-encode with marker."""
    try:
        from cryptography.fernet import Fernet
        key = os.environ.get("PII_ENCRYPTION_KEY", "").encode()
        if len(key) == 44:
            f = Fernet(key)
            return "[ENCRYPTED]" + f.encrypt(value.encode()).decode()
    except Exception:
        pass
    import base64
    return "[B64-ENCODED]" + base64.b64encode(value.encode()).decode()


def _attach_provenance(response_text: str, model_id: str) -> dict:
    """Attach provenance metadata, label, and HMAC watermark to LLM output."""
    timestamp = time.time()
    content_hash = hashlib.sha256(response_text.encode()).hexdigest()
    watermark = hmac.new(
        _WATERMARK_SECRET,
        f"{response_text}{model_id}{timestamp}".encode(),
        hashlib.sha256
    ).hexdigest()
    return {
        "text": response_text,
        "provenance": {
            "model_id": model_id,
            "timestamp": timestamp,
            "content_origin": "ai-generated",
            "content_hash": content_hash,
        },
        "label": "AI_GENERATED_CONTENT",
        "watermark": watermark,
    }


class TechSupportAgent:
    """
    Technical support agent for handling general user queries.

    Privilege Level: LOW
    Capabilities:
    - Answer general questions
    - Provide technical guidance
    - Inform users to contact specialized teams directly
    """

    ALLOWED_ROLES = ["user", "tech_support", "admin"]
    PRIVILEGE_LEVEL = "low"
    ALLOWED_ESCALATION_TARGETS: frozenset = frozenset()  # No runtime escalation permitted

    def __init__(self, llm_client: ApprovedLLMClient):
        self.llm_client = llm_client
        self.agent_id = "tech_support"
        self.agent_name = "Tech Support Agent"

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
            Response dictionary
        """
        trace_id = str(uuid.uuid4())

        token = headers.get("X-Agent-Token") if headers else None
        if not _validate_token(token):
            raise PermissionError("Authentication failed: missing or invalid X-Agent-Token.")

        raw_message = context.get("user_message", "")

        # Sanitize and validate input before any processing
        user_message = self._sanitize_input(raw_message)
        self._check_malicious_input(user_message)

        # Check if this needs finance information
        if self._needs_finance_escalation(user_message):
            logger.info(
                "Tech support detected finance-related query; directing user to contact finance directly.",
                extra={
                    "reason": "Financial query detected",
                    "trace_id": trace_id,
                }
            )
            return {
                "response": (
                    "Your query appears to relate to financial information. "
                    "Please contact the Finance team directly for assistance with "
                    "quarterly reports, financial statements, budgets, and related topics."
                ),
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "task_complete": True,
                "termination_reason": "finance_query_redirected",
                "trace_id": trace_id,
            }

        if not user_message.strip():
            return {
                "response": "I did not receive a message. Please provide your question.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "task_complete": True,
                "termination_reason": "empty_input",
                "trace_id": trace_id,
            }

        # Handle the query directly
        response_obj = await self._process_query(user_message, context, trace_id=trace_id)

        return {
            "response": response_obj,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
            "task_complete": True,
            "termination_reason": "successful_response",
            "trace_id": trace_id,
            "model_id": PINNED_MODEL,
        }

    def _needs_finance_escalation(self, message: str) -> bool:
        """Check if message requires finance agent access."""
        finance_triggers = [
            "quarterly report", "financial statement", "budget",
            "revenue numbers", "profit margin", "expense report",
            "balance sheet", "cash flow", "earnings"
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in finance_triggers)

    def _sanitize_input(self, text: str) -> str:
        """Strip null bytes, control characters, and enforce maximum length."""
        # Remove null bytes
        text = text.replace("\x00", "")
        # Remove control characters (except newline and tab which are legitimate)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) not in ("Cc",) or ch in ("\n", "\t", "\r")
        )
        # Enforce maximum length
        if len(text) > MAX_INPUT_LENGTH:
            text = text[:MAX_INPUT_LENGTH]
        return text

    def _check_malicious_input(self, text: str) -> None:
        """
        Check for prompt injection, shell commands, hidden characters,
        base64-encoded payloads, and other malicious patterns.
        Raises ValueError if dangerous content is detected.
        """
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                raise ValueError("Input rejected: potential prompt injection detected.")
        for pattern in _SHELL_PATTERNS:
            if pattern.search(text):
                raise ValueError("Input rejected: potential shell command detected.")
        # Check for high density of invisible/control characters
        invisible_count = sum(
            1 for ch in text if unicodedata.category(ch) in ("Cf", "Cc")
        )
        if invisible_count > 5:
            raise ValueError("Input rejected: excessive invisible characters detected.")

    def _sanitize_llm_output(self, text: str) -> str:
        """
        Scan LLM output for dynamic code execution primitives.
        Raises ValueError if dangerous content is found.
        """
        for pattern in _CODE_EXEC_PATTERNS:
            if pattern.search(text):
                raise ValueError(
                    "LLM output rejected: dynamic code execution primitive detected."
                )
        return text

    async def _process_query(
        self,
        message: str,
        context: dict,
        trace_id: str = ""
    ) -> dict:
        """
        Process a general tech support query.
        Input is sanitized before sending to LLM.
        Output is scanned for dangerous primitives and labeled with provenance.
        """
        if not _verify_model_integrity(PINNED_MODEL):
            raise ValueError(f"Model '{PINNED_MODEL}' is not on the approved model registry.")

        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        # Sanitize and validate input before sending to LLM
        sanitized_message = self._sanitize_input(message)
        self._check_malicious_input(sanitized_message)

        input_hash = hashlib.sha256(sanitized_message.encode()).hexdigest()

        logger.info(
            "LLM request",
            extra={
                "trace_id": trace_id,
                "agent": self.agent_id,
                "model": PINNED_MODEL,
                "system_prompt_length": len(system_prompt),
                "user_message_hash": input_hash,
            }
        )

        # Verify tool is on allow list
        if "llm_chat" not in ALLOWED_TOOLS:
            _audit_logger.info(
                "TOOL_DENIED actor=%s policy_version=1 tool=llm_chat reason=not_in_allowlist",
                self.agent_id,
            )
            raise PermissionError("Tool 'llm_chat' is not permitted for this agent.")

        raw_response = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sanitized_message}
            ],
            model=PINNED_MODEL,
        )

        logger.info(
            "LLM response received",
            extra={
                "trace_id": trace_id,
                "agent": self.agent_id,
                "model": PINNED_MODEL,
                "response_length": len(raw_response) if raw_response else 0,
            }
        )

        # Scan output for dangerous primitives
        safe_response = self._sanitize_llm_output(raw_response)

        # Attach provenance, label, and watermark — fail-safe block
        try:
            labeled_response = _attach_provenance(safe_response, PINNED_MODEL)
        except Exception as exc:
            raise RuntimeError(
                "Failed to attach provenance/watermark to LLM output; refusing to return unlabeled content."
            ) from exc

        _audit_logger.info(
            "LLM_INTERACTION trace_id=%s model=%s input_hash=%s output_hash=%s timestamp=%s principal=%s model_integrity=%s",
            trace_id,
            PINNED_MODEL,
            input_hash,
            labeled_response["provenance"]["content_hash"],
            labeled_response["provenance"]["timestamp"],
            self.agent_id,
            APPROVED_MODELS.get(PINNED_MODEL, "unknown"),
        )

        return labeled_response

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.
        PII fields are encrypted before storage/return.
        Sensitive fields are excluded from the returned dict.
        """
        encrypted_email = _encrypt_pii("user@placeholder.invalid")
        encrypted_phone = _encrypt_pii("000-000-0000")

        user_context = {
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
            "account_details": {
                "contact_email": encrypted_email,
                "phone": encrypted_phone,
            }
        }

        logger.info(
            "Retrieved user context",
            extra={
                "user_id": user_id,
                "subscription_tier": user_context["subscription_tier"],
            }
        )

        return user_context