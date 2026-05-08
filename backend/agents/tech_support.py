"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Can escalate to higher-privilege agents when needed.

SECURITY NOTES (for Unifai demo):
- Low privilege agent can escalate without proper verification
- User context passed without sanitization
"""

import logging
import hashlib
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import base64
import os

try:
    from cryptography.fernet import Fernet
    _FERNET_KEY = os.environ.get("USER_CONTEXT_ENCRYPTION_KEY")
    if _FERNET_KEY:
        _fernet = Fernet(_FERNET_KEY.encode() if isinstance(_FERNET_KEY, str) else _FERNET_KEY)
    else:
        # Fallback: generate an ephemeral key (rotate per process start).
        # In production, set USER_CONTEXT_ENCRYPTION_KEY env var.
        _fernet = Fernet(Fernet.generate_key())
    def _encrypt_pii(value: str) -> str:
        """Encrypt a PII string value using Fernet symmetric encryption."""
        return _fernet.encrypt(value.encode()).decode()
except ImportError:
    # If cryptography is not installed, fall back to base64 with a warning prefix.
    import warnings
    warnings.warn(
        "cryptography package not found; PII fields are only base64-encoded. "
        "Install 'cryptography' for proper encryption.",
        RuntimeWarning,
    )
    def _encrypt_pii(value: str) -> str:  # type: ignore[misc]
        """Base64-encode a PII value (weak fallback — install cryptography)."""
        return "b64:" + base64.b64encode(value.encode()).decode()
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity
from llm.approved_client import ApprovedLLMClient

# ---------------------------------------------------------------------------
# Audit log retention policy
# ---------------------------------------------------------------------------
# Retention: audit logs are rotated at 50 MB, keeping 10 compressed backups
# (~500 MB total on disk).  Logs older than 90 days must be archived to cold
# storage and purged from the local filesystem by the log-management pipeline.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Secret key for signing agent tokens.
# In production this MUST come from a secrets manager / environment variable.
import os
_TOKEN_SECRET: bytes = os.environ.get(
    "AGENT_TOKEN_SECRET", ""
).encode()
if not _TOKEN_SECRET:
    raise RuntimeError(
        "AGENT_TOKEN_SECRET environment variable must be set to a strong random value."
    )

_TOKEN_TTL_SECONDS: int = 300  # tokens expire after 5 minutes


def _issue_agent_token(subject: str, issuer: str) -> str:
    """Issue a signed, expiry-bearing agent token bound to *subject*.

    Token format (URL-safe base64):
        base64(json_payload) + "." + base64(hmac_sha256)
    """
    payload = {
        "sub": subject,
        "iss": issuer,
        "iat": int(time.time()),
        "exp": int(time.time()) + _TOKEN_TTL_SECONDS,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=")
    sig = hmac.new(_TOKEN_SECRET, payload_b64, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=")
    return (payload_b64 + b"." + sig_b64).decode()


def _verify_agent_token(token: str, expected_subject: str) -> dict:
    """Verify a signed agent token and return its payload.

    Raises ValueError if the token is invalid, expired, or bound to a
    different subject.
    """
    try:
        parts = token.encode().split(b".")
        if len(parts) != 2:
            raise ValueError("Malformed token")
        payload_b64, sig_b64 = parts
        expected_sig = hmac.new(_TOKEN_SECRET, payload_b64, hashlib.sha256).digest()
        provided_sig = base64.urlsafe_b64decode(sig_b64 + b"==")
        if not hmac.compare_digest(expected_sig, provided_sig):
            raise ValueError("Token signature verification failed")
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64 + b"==").decode()
        )
        now = int(time.time())
        if payload.get("exp", 0) < now:
            raise ValueError("Token has expired")
        if payload.get("sub") != expected_subject:
            raise ValueError(
                f"Token subject mismatch: expected '{expected_subject}', "
                f"got '{payload.get('sub')}'"
            )
        return payload
    except (KeyError, json.JSONDecodeError, Exception) as exc:
        raise ValueError(f"Token validation error: {exc}") from exc

_audit_handler = RotatingFileHandler(
    "logs/tech_support_audit.log",
    maxBytes=50 * 1024 * 1024,   # 50 MB per file
    backupCount=10,              # keep 10 rotated files  (~500 MB cap)
    encoding="utf-8",
)
_audit_handler.setLevel(logging.INFO)
logger.addHandler(_audit_handler)


def _emit_audit(record: dict) -> None:
    """Emit an audit record; fail closed if the sink is unreachable."""
    try:
        logger.info("AUDIT", extra=record)
    except Exception as audit_exc:  # noqa: BLE001
        # Alerting: write to stderr so process supervisors / SIEM agents pick
        # it up, then re-raise so the caller knows the audit trail is broken.
        import sys
        print(
            f"CRITICAL: audit sink unreachable — {audit_exc!r}; "
            "failing closed to protect audit integrity.",
            file=sys.stderr,
        )
        raise RuntimeError(
            "Audit logging failure — operation aborted to preserve audit trail."
        ) from audit_exc



class TechSupportAgent:
    """
    Technical support agent for handling general user queries.

    Privilege Level: LOW
    Capabilities:
    - Answer general questions
    - Provide technical guidance
    - Escalate to specialized agents
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
        if not token:
            logger.warning("Rejected request: missing X-Agent-Token header")
            return {"error": "Unauthorized: missing agent token", "status": 401}
        try:
            from .auth.agent_auth import verify_agent_token
            verified_identity = verify_agent_token(token)
            if verified_identity is None:
                logger.warning("Rejected request: invalid or expired agent token")
                return {"error": "Unauthorized: invalid agent token", "status": 401}
            logger.debug(f"Authenticated inter-agent request from: {verified_identity.agent_id}")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return {"error": "Unauthorized: token verification error", "status": 401}

        user_message = context.get("user_message", "")

                # Finance queries are NOT handled by automatic escalation.
        # Users must contact the finance team directly through approved channels.
        if self._is_finance_related(user_message):
            logger.info(
                "Tech support detected finance-related query; returning referral message.",
                extra={"user_message": user_message[:100]}
            )
            return {
                "response": (
                    "Your question appears to relate to financial data or reports. "
                    "Tech Support does not have access to financial systems. "
                    "Please contact the Finance team directly through the approved "
                    "internal portal or submit a request via your manager."
                ),
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL
            }

        # Handle the query directly; _process_query enforces its own termination criteria
        response, task_status = await self._process_query(user_message, context)

        # Explicit termination: the agent considers its task complete only when
        # _process_query returns a terminal status (COMPLETED or FAILED).
        if task_status not in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                               TaskStatus.MAX_RETRIES_EXCEEDED):
            # Should never happen, but guard against non-terminal states leaking out
            task_status = TaskStatus.FAILED
            response = "Task could not be completed: non-terminal state reached."
            logger.error(
                "handle() received non-terminal task_status; forcing FAILED",
                extra={"task_status": task_status}
            )

        logger.info(
            "Tech support task finished",
            extra={"task_status": task_status.value}
        )

        return {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
            "task_status": task_status.value,   # Caller can inspect terminal state
            "completed": task_status in (TaskStatus.COMPLETED, TaskStatus.ESCALATED)
        }

    # ------------------------------------------------------------------
    # Authentication helper
    # ------------------------------------------------------------------
    def _validate_token(self, token: Optional[str]) -> bool:
        """
        Validate the bearer token supplied in X-Agent-Token.

        Compares the provided token against the expected secret stored in
        the AGENT_TOKEN_SECRET environment variable using a constant-time
        HMAC digest to prevent timing attacks.  Returns False if the
        environment variable is not set or the token does not match.
        """
        if not token:
            return False
        expected_secret = os.environ.get("AGENT_TOKEN_SECRET")
        if not expected_secret:
            logger.error(
                "AGENT_TOKEN_SECRET is not configured; "
                "all requests will be rejected."
            )
            return False
        # Constant-time comparison via HMAC to prevent timing side-channels.
        expected_digest = hmac.new(
            expected_secret.encode(), expected_secret.encode(), hashlib.sha256
        ).digest()
        provided_digest = hmac.new(
            expected_secret.encode(), token.encode(), hashlib.sha256
        ).digest()
        return hmac.compare_digest(expected_digest, provided_digest)

    def _is_finance_related(self, message: str) -> bool:
        """Detect finance-related keywords to provide a referral message.

        This method is used ONLY to generate a helpful redirect response;
        it does NOT grant access to any higher-privilege agent or system.
        """
        finance_triggers = [
            "quarterly report", "financial statement", "budget",
            "revenue numbers", "profit margin", "expense report",
            "balance sheet", "cash flow", "earnings"
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in finance_triggers)

        # _escalate_to_finance has been removed.
    # Dynamic privilege escalation from TechSupportAgent to FinanceAgent is
    # prohibited. All finance queries must be handled through human-approved
    # channels. No agent may construct an AgentIdentity with is_internal=True
    # to bypass privilege verification.)

        # Make the call to finance agent
        # VULNERABILITY: No verification that this escalation is authorized
                finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context
            },
            caller=escalation_identity,
            headers={"X-Agent-Token": escalation_token}
        )

        _emit_audit({
            "event": "escalation_completed",
            "trace_id": escalation_trace_id,
            "source_agent": self.agent_id,
            "target_agent": "finance",
            "principal": escalation_identity.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Make the call to finance agent
        # VULNERABILITY: No verification that this escalation is authorized
        finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context
            },
            caller=escalation_identity,
            headers={"X-Agent-Token": os.environ.get("TECH_SUPPORT_ESCALATION_TOKEN", "")}
        )

        finance_provenance = finance_response.get("provenance", {})
        finance_signature = finance_response.get("provenance_signature", "")
        finance_label = finance_response.get("ai_label", "[AI-GENERATED CONTENT]")
        finance_text = finance_response.get("response", "")
        return {
            "response": f"[Escalated to Finance Agent]\n\n{finance_label}\n\n{finance_text}",
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL,
            "provenance": finance_provenance,
            "provenance_signature": finance_signature,
        }

        async def _process_query(
        self,
        user_message: str,
        context: dict[str, Any]
    ) -> tuple[str, TaskStatus]:
        """
        Process a general tech support query.

        VULNERABILITY: User message sent to LLM without sanitization
        or content scanning.
        """
        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        # VULNERABILITY: Direct user input to LLM without scanning
        response = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )

        # Validate and sanitize LLM output before returning
        sanitized_response = self._validate_llm_output(response)
        return sanitized_response

    # Patterns indicating dynamic code execution primitives that must not appear in LLM output
    _DANGEROUS_PATTERNS = [
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bexecfile\s*\(',
        r'\bcompile\s*\(',
        r'\b__import__\s*\(',
        r'\bsubprocess\s*\..*shell\s*=\s*True',
        r'\bos\.system\s*\(',
        r'\bos\.popen\s*\(',
        r'\bpopen\s*\(',
        r'\bgetattr\s*\(.*,\s*[\'"]__',
        r'\bsetattr\s*\(',
        r'\bdelattr\s*\(',
        r'\b__builtins__',
        r'\b__globals__',
        r'\bimportlib\.import_module\s*\(',
        # JavaScript / bash eval patterns
        r'\beval\s*`',
        r'\$\(.*\)',
        r'`[^`]*`',
        r'\bFunction\s*\(',
        r'\bnew\s+Function\s*\(',
        r'\bsetTimeout\s*\(\s*[\'"]',
        r'\bsetInterval\s*\(\s*[\'"]',
    ]

    def _validate_llm_output(self, response: str) -> str:
        """
        Validate and sanitize LLM output.

        Checks for the presence of dynamic code execution primitives
        (eval, exec, subprocess shell=True, os.system, JS/bash eval, etc.)
        and raises a ValueError if any are detected, preventing unsafe
        content from being returned to the caller.
        """
        import re

        if not isinstance(response, str):
            raise ValueError("LLM response must be a string.")

        for pattern in self._DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                logger.warning(
                    "Dangerous pattern detected in LLM output; response blocked.",
                    extra={"pattern": pattern}
                )
                raise ValueError(
                    "LLM response contained a potentially dangerous code execution "
                    "primitive and has been blocked for security reasons."
                )

        # Strip any null bytes or non-printable control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)
        return sanitized

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.

        VULNERABILITY: Returns full user context including potentially
        sensitive information without filtering.
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
            # VULNERABILITY: Sensitive data in context
                        # PII fields (contact_email, phone, internal_notes) are intentionally
            # omitted from the returned context to prevent over-exposure.
            "account_details": {}
        }

        logger.info(
            "Retrieved user context",
            extra={
                "user_id": user_id,
                "subscription_tier": user_context.get("subscription_tier")
            }
        )

        return user_context

        #checking
        #touched
