"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Can escalate to higher-privilege agents when needed.

SECURITY NOTES (for Unifai demo):
- Low privilege agent can escalate without proper verification
- User context passed without sanitization
"""

import logging
import os
import re
import base64
import unicodedata
from typing import Any, Optional

import jwt
from cryptography.fernet import Fernet

from .auth.agent_auth import AgentIdentity, AgentAuth
from llm.approved_provider import ApprovedLLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII encryption helpers
# ---------------------------------------------------------------------------
_FERNET_KEY = os.environ.get("PII_ENCRYPTION_KEY", "").encode()
if _FERNET_KEY:
    _fernet = Fernet(_FERNET_KEY)
else:
    _fernet = None


def _encrypt_pii(value: str) -> str:
    """Encrypt a PII string using Fernet symmetric encryption."""
    if _fernet is None:
        raise RuntimeError("PII_ENCRYPTION_KEY environment variable is not set.")
    return _fernet.encrypt(value.encode()).decode()


# ---------------------------------------------------------------------------
# JWT / token validation helpers
# ---------------------------------------------------------------------------
_JWT_SECRET = os.environ.get("AGENT_JWT_SECRET", "")
_JWT_ALGORITHM = os.environ.get("AGENT_JWT_ALGORITHM", "HS256")
_JWT_ISSUER = os.environ.get("AGENT_JWT_ISSUER", "unifai")
_JWT_AUDIENCE = os.environ.get("AGENT_JWT_AUDIENCE", "tech_support")


def _validate_token(token: str) -> dict:
    """
    Validate a JWT token's signature, issuer, audience, and expiry.
    Raises jwt.PyJWTError on any validation failure.
    """
    return jwt.decode(
        token,
        _JWT_SECRET,
        algorithms=[_JWT_ALGORITHM],
        issuer=_JWT_ISSUER,
        audience=_JWT_AUDIENCE,
        options={"require": ["exp", "iss", "aud"]},
    )


# ---------------------------------------------------------------------------
# Input sanitization helpers
# ---------------------------------------------------------------------------
_MAX_MESSAGE_LENGTH = 4096

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a\s+)?", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"###\s*(system|user|assistant)\s*:", re.IGNORECASE),
    re.compile(r"role\s*:\s*(system|assistant)", re.IGNORECASE),
]

_DANGEROUS_OUTPUT_PATTERNS = [
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\b.*shell\s*=\s*True", re.IGNORECASE | re.DOTALL),
    re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bjavascript\s*:", re.IGNORECASE),
    re.compile(r"\beval\b", re.IGNORECASE),
]

_SHELL_COMMAND_PATTERNS = [
    re.compile(r";\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\b", re.IGNORECASE),
    re.compile(r"\|\s*(bash|sh|python|perl|ruby)\b", re.IGNORECASE),
    re.compile(r"`[^`]+`"),
    re.compile(r"\$\([^)]+\)"),
    re.compile(r"\beval\b", re.IGNORECASE),
    re.compile(r"\bexec\b", re.IGNORECASE),
]

_BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")


def _contains_invisible_chars(text: str) -> bool:
    """Check for hidden/invisible Unicode characters."""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Cc") and ch not in ("\n", "\r", "\t"):
            return True
    return False


def _contains_base64_payload(text: str) -> bool:
    """Heuristic check for embedded base64 content."""
    matches = _BASE64_PATTERN.findall(text)
    for m in matches:
        if len(m) >= 20:
            try:
                decoded = base64.b64decode(m).decode("utf-8", errors="ignore")
                if any(kw in decoded.lower() for kw in ["eval", "exec", "system", "bash", "sh "]):
                    return True
            except Exception:
                pass
    return False


def _sanitize_input(message: str) -> str:
    """
    Sanitize and validate user input before passing to the LLM.
    Raises ValueError if the message is invalid or dangerous.
    """
    if not message or not message.strip():
        raise ValueError("Empty message is not allowed.")

    message = message.strip()

    if len(message) > _MAX_MESSAGE_LENGTH:
        message = message[:_MAX_MESSAGE_LENGTH]

    if _contains_invisible_chars(message):
        raise ValueError("Message contains hidden or invisible characters.")

    if _contains_base64_payload(message):
        raise ValueError("Message contains suspicious encoded content.")

    for pattern in _SHELL_COMMAND_PATTERNS:
        if pattern.search(message):
            raise ValueError("Message contains potentially malicious shell command patterns.")

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(message):
            raise ValueError("Message contains prompt injection patterns.")

    return message


def _sanitize_llm_output(response: str) -> str:
    """
    Sanitize LLM output by checking for dynamic code execution primitives.
    Raises ValueError if dangerous patterns are detected.
    """
    for pattern in _DANGEROUS_OUTPUT_PATTERNS:
        if pattern.search(response):
            raise ValueError(
                "LLM response contains potentially dangerous code execution patterns and has been blocked."
            )
    return response


class TechSupportAgent:
    """
    Technical support agent for handling general user queries.

    Privilege Level: LOW
    Capabilities:
    - Answer general questions
    - Provide technical guidance
    - Escalate to specialized agents (via allow list only)
    """

    ALLOWED_ROLES = ["user", "tech_support", "admin"]
    PRIVILEGE_LEVEL = "low"

    # Explicit tool allow list — only tools named here may be invoked.
    TOOL_ALLOW_LIST: list[str] = []

    # Termination / iteration limits
    MAX_ITERATIONS = 3

    # Policy version for audit logging
    POLICY_VERSION = "1.0"

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
        # --- Authentication: validate inbound token ---
        token = headers.get("X-Agent-Token") if headers else None
        if not token:
            logger.warning("Request received without authentication token.")
            return {"error": "Unauthorized: missing authentication token.", "agent": self.agent_id}

        try:
            _validate_token(token)
        except Exception as exc:
            logger.warning("Token validation failed: %s", str(exc))
            return {"error": "Unauthorized: invalid or expired token.", "agent": self.agent_id}

        # --- Termination criteria: validate caller role ---
        if caller.privilege_level not in [r.lower() for r in self.ALLOWED_ROLES]:
            logger.warning("Caller role '%s' is not permitted.", caller.privilege_level)
            return {"error": "Forbidden: unsupported caller role.", "agent": self.agent_id}

        user_message = context.get("user_message", "")

        # --- Termination criteria: empty message ---
        if not user_message or not user_message.strip():
            logger.info("Empty user message received; terminating early.")
            return {
                "response": "No message provided. Please describe your issue.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "terminated": True,
            }

        # --- Iteration / retry guard ---
        iteration = 0
        response = None

        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            try:
                response = await self._process_query(user_message, context)
                break  # Successful response — exit loop
            except ValueError as exc:
                logger.warning("Iteration %d failed with validation error: %s", iteration, str(exc))
                if iteration >= self.MAX_ITERATIONS:
                    return {
                        "error": "Unable to process request after maximum retries.",
                        "agent": self.agent_id,
                        "terminated": True,
                    }

        if response is None:
            return {
                "error": "Unable to produce a response.",
                "agent": self.agent_id,
                "terminated": True,
            }

        return {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
        }

    async def _process_query(
        self,
        message: str,
        context: dict
    ) -> str:
        """
        Process a general tech support query.
        Input is sanitized and LLM output is validated before returning.
        """
        # --- Input sanitization and validation ---
        try:
            sanitized_message = _sanitize_input(message)
        except ValueError as exc:
            logger.warning("Input sanitization failed: %s", str(exc))
            raise

        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sanitized_message},
        ]

        logger.info(
            "Sending request to LLM",
            extra={"messages": messages, "agent": self.agent_id}
        )

        response = await self.llm_client.chat(messages=messages)

        logger.info(
            "Received response from LLM",
            extra={"response_length": len(response) if response else 0, "agent": self.agent_id}
        )

        # --- Output sanitization ---
        try:
            response = _sanitize_llm_output(response)
        except ValueError as exc:
            logger.error("LLM output sanitization failed: %s", str(exc))
            raise

        return response

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.
        PII fields are encrypted before storage/return.
        """
        # Simulated user context retrieval
        # In a real app, this would query a database
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
            "internal_notes": "VIP customer - handle with priority",
            "account_details": {
                "contact_email": _encrypt_pii("<RUNTIME_EMAIL>"),
                "phone": _encrypt_pii("<RUNTIME_PHONE>"),
            }
        }

        logger.info(
            "Retrieved user context",
            extra={
                "user_id": user_context.get("user_id"),
                "subscription_tier": user_context.get("subscription_tier"),
            }
        )

        return user_context

        #checking
        #touched