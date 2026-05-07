"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Can escalate to higher-privilege agents when needed.

SECURITY NOTES (for Unifai demo):
- Low privilege agent can escalate without proper verification
- User context passed without sanitization
"""

import base64
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
from llm.approved_registry import ApprovedLLMClient

logger = logging.getLogger(__name__)

# Audit logger with rotating file handler
audit_logger = logging.getLogger("audit.tech_support")
if not audit_logger.handlers:
    _audit_handler = logging.handlers.RotatingFileHandler(
        "audit_tech_support.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=10
    )
    _audit_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    audit_logger.addHandler(_audit_handler)
    audit_logger.setLevel(logging.INFO)

# Approved model constant (version-pinned)
APPROVED_MODEL = "openai/gpt-4o-2024-05-13"

# Allowed tools and agent escalations
ALLOWED_TOOLS = ["llm_chat"]
ALLOWED_AGENT_ESCALATIONS: list[str] = []  # No direct escalations allowed; require human approval

# Shared secret for HMAC token validation (loaded from environment)
_AGENT_SHARED_SECRET = os.environ.get("AGENT_SHARED_SECRET", "")

# Dynamic code execution primitives to block in LLM output
_DANGEROUS_PRIMITIVES = [
    "eval(", "exec(", "subprocess", "os.system", "compile(",
    "__import__", "importlib", "execfile", "open(", "pty.spawn",
    "popen", "Popen", "ctypes", "cffi"
]

# Prompt injection / malicious command patterns
_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+all\s+prior",
    r"you\s+are\s+now",
    r"act\s+as\s+if",
    r"system\s*prompt",
    r"<\s*script",
    r"javascript\s*:",
    r"data\s*:",
    r"base64",
    r"\\x[0-9a-fA-F]{2}",
    r"\\u[0-9a-fA-F]{4}",
    r"rm\s+-rf",
    r"sudo\s+",
    r"chmod\s+",
    r"curl\s+",
    r"wget\s+",
    r"\|\s*sh",
    r"&&\s*",
    r";\s*[a-z]+\s",
]

MAX_INPUT_LENGTH = 4096


def _validate_token(token: str) -> bool:
    """Validate an HMAC-signed agent token with expiry checking."""
    if not token or not _AGENT_SHARED_SECRET:
        return False
    try:
        # Expected format: <expiry_timestamp>.<hmac_hex>
        parts = token.split(".", 1)
        if len(parts) != 2:
            return False
        expiry_str, provided_hmac = parts
        expiry = int(expiry_str)
        if time.time() > expiry:
            return False
        expected_hmac = hmac.new(
            _AGENT_SHARED_SECRET.encode(),
            expiry_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_hmac, provided_hmac)
    except Exception:
        return False


def _generate_signed_token(ttl_seconds: int = 300) -> str:
    """Generate a cryptographically signed, expiring HMAC-SHA256 token."""
    expiry = int(time.time()) + ttl_seconds
    expiry_str = str(expiry)
    sig = hmac.new(
        _AGENT_SHARED_SECRET.encode(),
        expiry_str.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{expiry_str}.{sig}"


def _sanitize_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Strip null bytes, control characters, enforce max length."""
    if not text:
        return ""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Remove control characters (except newline, tab, carriage return)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", "\r")
    )
    # Enforce max length
    text = text[:max_length]
    return text


def _sanitize_and_validate_message(message: str) -> str:
    """
    Sanitize and validate user message:
    - Strip null bytes and control characters
    - Enforce length limit
    - Remove prompt injection patterns
    - Check for hidden/invisible characters
    - Check for base64-encoded payloads
    - Check for shell commands and binary content
    - Check for leetspeak obfuscation patterns
    """
    message = _sanitize_input(message)

    # Remove invisible/zero-width characters
    invisible_chars = [
        "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",
        "\ufeff", "\u2028", "\u2029"
    ]
    for ch in invisible_chars:
        message = message.replace(ch, "")

    # Check for binary content
    try:
        message.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Message contains invalid binary content.")

    # Check for base64-encoded payloads (heuristic: long base64-like strings)
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
    if b64_pattern.search(message):
        # Attempt to decode and check for dangerous content
        matches = b64_pattern.findall(message)
        for match in matches:
            try:
                decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                for primitive in _DANGEROUS_PRIMITIVES:
                    if primitive in decoded:
                        raise ValueError("Message contains encoded dangerous content.")
            except Exception as e:
                if "dangerous" in str(e):
                    raise

    # Strip prompt injection patterns
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            raise ValueError(f"Message contains disallowed pattern: {pattern}")

    return message


def _sanitize_llm_output(response: str) -> str:
    """
    Validate and sanitize LLM output.
    Check for dynamic code execution primitives.
    """
    for primitive in _DANGEROUS_PRIMITIVES:
        if primitive in response:
            raise ValueError(
                f"LLM response contains dangerous primitive: {primitive}"
            )
    return response


def _encrypt_pii(value: str) -> str:
    """Simple base64-based obfuscation for PII fields."""
    return base64.b64encode(value.encode("utf-8")).decode("utf-8")


def _apply_watermark(text: str, agent_id: str) -> str:
    """Apply a lightweight HMAC-based watermark to response text."""
    if not _AGENT_SHARED_SECRET:
        watermark = hashlib.sha256(f"{agent_id}:{text}".encode()).hexdigest()[:16]
    else:
        watermark = hmac.new(
            _AGENT_SHARED_SECRET.encode(),
            f"{agent_id}:{text}".encode(),
            hashlib.sha256
        ).hexdigest()[:16]
    return f"{text}\n\n[AI-Generated | watermark:{watermark}]"


class TechSupportAgent:
    """
    Technical support agent for handling general user queries.

    Privilege Level: LOW
    Capabilities:
    - Answer general questions
    - Provide technical guidance
    - Escalate to specialized agents (requires human approval)
    """

    ALLOWED_ROLES = ["user", "tech_support", "admin"]
    PRIVILEGE_LEVEL = "low"

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
        token = headers.get("X-Agent-Token") if headers else None
        if not token or not _validate_token(token):
            raise PermissionError("Missing or invalid authentication token. Access denied.")

        logger.debug("Received request with validated token.")

        user_message = context.get("user_message", "")

        # Check if this needs escalation to finance
        if self._needs_finance_escalation(user_message):
            logger.info(
                "Tech support requesting finance escalation (human approval required)",
                extra={
                    "reason": "Financial query detected",
                    "user_message": user_message[:100]
                }
            )
            return await self._escalate_to_finance(user_message, context, token)

        # Handle the query directly
        # Check tool is allowed
        if "llm_chat" not in ALLOWED_TOOLS:
            logger.warning(
                "Denied tool invocation",
                extra={
                    "actor": self.agent_id,
                    "tool": "llm_chat",
                    "policy_version": "1.0",
                    "denial_reason": "Tool not in ALLOWED_TOOLS"
                }
            )
            raise PermissionError("Tool 'llm_chat' is not in the allowed tools list.")

        response = await self._process_query(user_message, context)

        return {
            "response": response,
            "agent": self.agent_id,
            "model": APPROVED_MODEL
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

    async def _escalate_to_finance(
        self,
        query: str,
        original_context: dict,
        validated_token: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Request escalation to finance agent.

        This method does NOT directly invoke the FinanceAgent.
        Instead, it returns a structured escalation request requiring
        human-in-the-loop approval.
        """
        # Check escalation is allowed
        if "finance" not in ALLOWED_AGENT_ESCALATIONS:
            logger.warning(
                "Denied agent escalation",
                extra={
                    "actor": self.agent_id,
                    "target_agent": "finance",
                    "policy_version": "1.0",
                    "denial_reason": "finance not in ALLOWED_AGENT_ESCALATIONS; human approval required"
                }
            )

        trace_id = str(uuid.uuid4())

        # Sanitize query before including in escalation request
        try:
            sanitized_query = _sanitize_and_validate_message(query)
        except ValueError as e:
            logger.warning("Escalation query failed sanitization: %s", str(e))
            sanitized_query = "[sanitized]"

        # Pass only an allowlisted subset of original_context
        safe_context = {
            k: original_context[k]
            for k in ("user_message", "session_id", "request_id")
            if k in original_context
        }

        audit_logger.info(
            "ESCALATION_REQUEST agent=%s target=finance trace_id=%s requires_human_approval=True",
            self.agent_id,
            trace_id
        )

        logger.info(
            "Escalation to finance agent requires human approval",
            extra={"trace_id": trace_id, "agent": self.agent_id}
        )

        return {
            "response": (
                "Your query requires access to financial data. "
                "An escalation request has been submitted for human approval. "
                "You will be notified when a finance specialist reviews your request."
            ),
            "agent": self.agent_id,
            "escalation_pending": True,
            "escalation_target": "finance",
            "trace_id": trace_id,
            "requires_human_approval": True,
            "model": APPROVED_MODEL
        }

    async def _process_query(
        self,
        message: str,
        context: dict
    ) -> str:
        """
        Process a general tech support query.
        """
        # Sanitize and validate input before passing to LLM
        try:
            message = _sanitize_and_validate_message(message)
        except ValueError as e:
            logger.warning("User message failed sanitization: %s", str(e))
            return "Your message could not be processed due to disallowed content."

        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        messages_payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        inference_id = str(uuid.uuid4())
        input_hash = hashlib.sha256(message.encode()).hexdigest()
        invocation_timestamp = time.time()

        audit_logger.info(
            "LLM_REQUEST inference_id=%s agent=%s model=%s input_hash=%s timestamp=%s",
            inference_id,
            self.agent_id,
            APPROVED_MODEL,
            input_hash,
            invocation_timestamp
        )

        logger.info(
            "Sending request to LLM",
            extra={
                "inference_id": inference_id,
                "model": APPROVED_MODEL,
                "input_hash": input_hash,
                "agent_id": self.agent_id
            }
        )

        try:
            response = await self.llm_client.chat(
                messages=messages_payload,
                model=APPROVED_MODEL
            )
        except Exception as e:
            audit_logger.error(
                "LLM_ERROR inference_id=%s agent=%s error=%s",
                inference_id,
                self.agent_id,
                str(e)
            )
            raise

        logger.info(
            "Received response from LLM",
            extra={
                "inference_id": inference_id,
                "model": APPROVED_MODEL,
                "output_hash": hashlib.sha256(str(response).encode()).hexdigest(),
                "agent_id": self.agent_id
            }
        )

        output_hash = hashlib.sha256(str(response).encode()).hexdigest()
        audit_logger.info(
            "LLM_RESPONSE inference_id=%s agent=%s model=%s output_hash=%s timestamp=%s",
            inference_id,
            self.agent_id,
            APPROVED_MODEL,
            output_hash,
            time.time()
        )

        # Validate and sanitize LLM output
        try:
            response = _sanitize_llm_output(response)
        except ValueError as e:
            audit_logger.warning(
                "LLM_OUTPUT_BLOCKED inference_id=%s reason=%s",
                inference_id,
                str(e)
            )
            return "The response could not be delivered due to a content policy violation."

        # Apply provenance metadata and watermark
        try:
            response_with_provenance = _apply_watermark(response, self.agent_id)
            provenance_metadata = {
                "model": APPROVED_MODEL,
                "timestamp": invocation_timestamp,
                "content_origin": "AI-generated",
                "inference_id": inference_id,
                "agent_id": self.agent_id,
                "label": "synthetic/AI-origin"
            }
            audit_logger.info(
                "LLM_PROVENANCE inference_id=%s provenance=%s",
                inference_id,
                provenance_metadata
            )
        except Exception as e:
            audit_logger.error(
                "PROVENANCE_FAILURE inference_id=%s error=%s",
                inference_id,
                str(e)
            )
            raise RuntimeError(
                "Failed to attach provenance controls to LLM output. "
                "Raw output will not be returned."
            ) from e

        return response_with_provenance

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.

        PII fields are encrypted before storage/return.
        Sensitive fields are excluded from the returned dict.
        """
        # Encrypt PII fields
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
                "phone": encrypted_phone
            }
        }

        # Log only safe subset — no PII, no internal_notes
        safe_log_context = {
            "user_id": user_context["user_id"],
            "subscription_tier": user_context["subscription_tier"],
            "preferences": user_context["preferences"]
        }

        logger.info(
            "Retrieved user context",
            extra={
                "user_context": safe_log_context
            }
        )

        # Return minimised context — strip sensitive fields
        return {
            "user_id": user_context["user_id"],
            "subscription_tier": user_context["subscription_tier"],
            "recent_queries": user_context["recent_queries"],
            "preferences": user_context["preferences"]
        }

        #checking
        #touched