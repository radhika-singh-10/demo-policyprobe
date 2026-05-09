"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Can escalate to higher-privilege agents when needed.

SECURITY NOTES (for Unifai demo):
- Low privilege agent can escalate without proper verification
- User context passed without sanitization
"""

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity, validate_agent_token
from llm.approved import ApprovedLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Append-only audit logger
# ---------------------------------------------------------------------------
_AUDIT_LOG_PATH = os.environ.get("AGENT_AUDIT_LOG", "/var/log/policyprobe/agent_audit.jsonl")


def _write_audit_record(record: dict) -> None:
    """Append a single JSON audit record to the append-only audit log.

    The file is opened in append mode on every call so that even if the
    process is restarted the log is never truncated.  In production this
    sink should be replaced with a write-once object-store or SIEM stream.
    """
    record.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        os.makedirs(os.path.dirname(_AUDIT_LOG_PATH), exist_ok=True)
        with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as exc:  # pragma: no cover
        # Never let audit failures silently swallow the error — surface it.
        logger.error("AUDIT_WRITE_FAILURE record=%s error=%s", record, exc)
        raise


def _sha256(text: str) -> str:
    """Return the hex SHA-256 digest of *text* (used for input hashing)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



class TechSupportAgent:
    """
    Technical support agent for handling general user queries.

    Privilege Level: LOW
    Capabilities:
    - Answer general questions
    - Provide technical guidance
    - Escalate to specialized agents (subject to tool allow list)
    """

    ALLOWED_ROLES = ["user", "tech_support", "admin"]
    PRIVILEGE_LEVEL = "low"

    # Explicit allow list: only tools/agents named here may be invoked by this agent.
    # To grant access to an additional tool, it MUST be added here deliberately.
    TOOL_ALLOW_LIST: list[str] = [
        # "finance_agent" is intentionally NOT listed — TechSupportAgent
        # does not have permission to invoke the high-privilege FinanceAgent.
        # Add tool names here only after explicit security review.
    ]

    def _check_tool_allowed(self, tool_name: str, caller: "AgentIdentity") -> None:
        """
        Enforce the tool allow list.  Raises PermissionError (fail-closed) when
        the requested tool is not on the list.  Always emits an audit log entry.

        Args:
            tool_name: Canonical name of the tool/agent to be invoked.
            caller:    Identity of the entity that triggered this agent.

        Raises:
            PermissionError: If tool_name is not in TOOL_ALLOW_LIST.
        """
        allowed = tool_name in self.TOOL_ALLOW_LIST
        audit_record = {
            "event": "tool_invocation_attempt",
            "agent": self.agent_id,
            "caller_id": getattr(caller, "agent_id", str(caller)),
            "tool": tool_name,
            "outcome": "allowed" if allowed else "denied",
        }
        if allowed:
            logger.info("[AUDIT] Tool invocation allowed", extra=audit_record)
        else:
            logger.warning(
                "[AUDIT] Tool invocation DENIED — not on allow list",
                extra=audit_record,
            )
            raise PermissionError(
                f"TechSupportAgent is not permitted to invoke '{tool_name}'. "
                f"Tool is not on the explicit allow list."
            )

    def __init__(self, llm_client: ApprovedLLMClient):
        self.llm_client = llm_client
        self.agent_id = "tech_support"
        self.agent_name = "Tech Support Agent"

    def _validate_token(self, token: Optional[str]) -> bool:
        """
        Validate the provided auth token.
        Checks against the configured trusted token(s) via environment variable.
        """
        import os
        if not token:
            return False
        trusted_token = os.environ.get("AGENT_AUTH_TOKEN")
        if not trusted_token:
            logger.error("AGENT_AUTH_TOKEN environment variable is not set")
            return False
        return token == trusted_token

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
        # Validate the incoming token against the caller's identity
        token = headers.get("X-Agent-Token") if headers else None
        if not token:
            logger.warning("Request received without X-Agent-Token header")
            return {"error": "Missing authentication token", "agent": self.agent_id}
        if not caller.validate_token(token):
            logger.warning(
                f"Invalid token presented by caller: {caller.agent_id}"
            )
            return {"error": "Invalid authentication token", "agent": self.agent_id}
        logger.debug(f"Token validated for caller: {caller.agent_id}")

        raw_message = context.get("user_message", "")
        try:
            user_message = self._sanitize_input(raw_message)
        except ValueError as exc:
            logger.warning("Rejected user_message due to validation failure: %s", exc)
            return {
                "response": "Your message could not be processed. Please revise and try again.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "error": "input_validation_failed"
            }

        # Check if this needs escalation to finance
        if self._needs_finance_escalation(user_message):
            logger.info(
                "Tech support escalating to finance",
                extra={
                    "reason": "Financial query detected",
                    "user_message": user_message[:100]
                }
            )
            # VULNERABILITY: Escalating to high-privilege agent without proper auth
            # Ensure a correlation ID exists before escalation so the causal chain is intact.
        if "correlation_id" not in context:
            context["correlation_id"] = str(uuid.uuid4())
        correlation_id = context["correlation_id"]
        principal = context.get("caller") or context.get("user_id") or "unknown"

        _write_audit_record({
            "event": "routing_decision",
            "agent_id": self.agent_id,
            "correlation_id": correlation_id,
            "principal": principal,
            "decision": "escalate_to_finance",
        })

        return await self._escalate_to_finance(user_message, context)

        # Handle the query directly
        # Ensure a correlation ID exists for this request lifecycle.
        if "correlation_id" not in context:
            context["correlation_id"] = str(uuid.uuid4())
        correlation_id = context["correlation_id"]
        principal = context.get("caller") or context.get("user_id") or "unknown"

        response = await self._process_query(user_message, context)

        _write_audit_record({
            "event": "routing_decision",
            "agent_id": self.agent_id,
            "correlation_id": correlation_id,
            "principal": principal,
            "decision": "process_directly",
            "response_hash_sha256": _sha256(str(response)),
        })

        return {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
            "correlation_id": correlation_id,
        }

    def _sanitize_input(self, message: str) -> str:
        """
        Validate and sanitize user input before processing.

        - Strips leading/trailing whitespace
        - Removes null bytes and other control characters
        - Enforces a maximum length
        - Rejects prompt-injection patterns
        """
        if not isinstance(message, str):
            raise ValueError("user_message must be a string")

        # Strip surrounding whitespace
        message = message.strip()

        # Enforce maximum length (16 KB is generous for a support query)
        MAX_LENGTH = 16_000
        if len(message) > MAX_LENGTH:
            raise ValueError(
                f"user_message exceeds maximum allowed length of {MAX_LENGTH} characters"
            )

        # Remove null bytes and ASCII control characters (except newline/tab)
        import re
        message = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", message)

        # Detect common prompt-injection / jailbreak patterns
        INJECTION_PATTERNS = [
            r"ignore (all |previous |prior )?instructions",
            r"disregard (all |previous |prior )?instructions",
            r"you are now",
            r"act as (a |an )?(different|new|unrestricted)",
            r"system prompt",
            r"<\|.*?\|>",          # token-boundary injection
            r"\[INST\]",           # Llama instruction tags
            r"###\s*(instruction|system)",
        ]
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                raise ValueError(
                    f"user_message contains a disallowed pattern: '{pattern}'"
                )

        return message

    def _needs_finance_escalation(self, message: str) -> bool:
        """Check if message requires finance agent access."""
        finance_triggers = [
            "quarterly report", "financial statement", "budget",
            "revenue numbers", "profit margin", "expense report",
            "balance sheet", "cash flow", "earnings"
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in finance_triggers)

        # Static allowlist: only these agent IDs may escalate to finance,
    # and only when the originating caller has been authenticated.
    _FINANCE_ESCALATION_ALLOWLIST: frozenset = frozenset()

    async def _escalate_to_finance(
        self,
        query: str,
        original_context: dict,
        caller: "AgentIdentity | None" = None
    ) -> dict[str, Any]:
        """
        Escalate query to finance agent.

        Authorization is enforced via a static allowlist and requires
        the originating caller to be authenticated.  No fabricated
        identity or is_internal flag is used.
        """
        # --- Authorization check (human-reviewable static policy) ---
        if self.agent_id not in self._FINANCE_ESCALATION_ALLOWLIST:
            logger.warning(
                "Blocked unauthorized finance escalation attempt",
                extra={"agent_id": self.agent_id}
            )
            raise PermissionError(
                f"Agent '{self.agent_id}' is not authorized to escalate to "
                "FinanceAgent.  Add the agent ID to "
                "TechSupportAgent._FINANCE_ESCALATION_ALLOWLIST after "
                "obtaining explicit approval."
            )

        if caller is None or getattr(caller, "is_internal", False):
            # Reject calls that arrive without a verified caller identity
            # or that already carry a self-asserted is_internal flag.
            logger.warning(
                "Blocked finance escalation: missing or self-elevated caller identity",
                extra={"agent_id": self.agent_id}
            )
            raise PermissionError(
                "Finance escalation requires a verified caller identity "
                "with is_internal=False."
            )
        # --- End authorization check ---

        # Import here to avoid circular imports
        from .finance import FinanceAgent

        # Use the verified caller identity as-is; do NOT fabricate a new
        # identity or set is_internal=True.
                # Second allow-list check inside the escalation helper as a defence-in-depth
        # guard (the primary check is in handle(); this prevents direct calls to
        # _escalate_to_finance() from bypassing the policy).
        self._check_tool_allowed("finance_agent", escalation_identity)

        finance_agent = FinanceAgent(self.llm_client)
        finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context
            },
            caller=escalation_identity,
            headers={"X-Agent-Token": "tech-support-escalation-token"}
        )

        finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context
            },
            caller=caller
        ) -> dict[str, Any]:
        """
        Escalate query to finance agent.

        VULNERABILITY: This method allows a low-privilege agent to
        access high-privilege agent without proper authorization.
        A properly issued token is used so the finance agent can authenticate
        this escalation request through its standard token validation path.
        """
        # Import here to avoid circular imports
        from .finance import FinanceAgent
        from .auth.agent_auth import AgentTokenIssuer

        # Obtain a properly issued, short-lived token for this escalation
        # VULNERABILITY: Low privilege agent claims internal status
        escalation_identity = AgentIdentity(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            privilege_level=self.PRIVILEGE_LEVEL,
            is_internal=True  # This bypasses privilege verification
        )
        

                finance_agent = FinanceAgent(self.llm_client)

        # Propagate or create a correlation ID so the causal chain is preserved
        # across the tech-support → finance escalation boundary.
        correlation_id = (
            original_context.get("correlation_id") if isinstance(original_context, dict) else None
        ) or str(uuid.uuid4())
        principal = (
            original_context.get("caller") if isinstance(original_context, dict) else None
        ) or "unknown"

        _write_audit_record({
            "event": "escalation_decision",
            "agent_id": self.agent_id,
            "correlation_id": correlation_id,
            "principal": principal,
            "escalated_to": "finance",
            "query_hash_sha256": _sha256(query),
            "reason": "tech_support_escalation",
        })

        # Make the call to finance agent
        # VULNERABILITY: No verification that this escalation is authorized
                finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context
            },
            caller=caller,
            headers={"X-Agent-Token": token}
        )}
        )

        _write_audit_record({
            "event": "escalation_response_received",
            "agent_id": self.agent_id,
            "correlation_id": correlation_id,
            "principal": principal,
            "escalated_to": "finance",
            "response_hash_sha256": _sha256(str(finance_response.get("response", ""))),
        })

        return {
            "response": f"[Escalated to Finance Agent]\n\n{finance_response.get('response', '')}",
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL,
            "correlation_id": correlation_id,
        },
            caller=escalation_identity,
            headers={"X-Agent-Token": escalation_token}
        )

        import datetime, hashlib
        _ts = datetime.datetime.utcnow().isoformat() + "Z"
        _model_id = getattr(self.llm_client, "model", "unknown-llm-model")
        _raw = finance_response.get('response', '')
        _wm = hashlib.sha256(
            f"{_model_id}:{_ts}:{self.agent_id}:escalated:{_raw}".encode()
        ).hexdigest()[:16]
        return {
            "response": f"[Escalated to Finance Agent]\n\n{_raw}",
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL,
            "ai_provenance": {
                "synthetic": True,
                "label": "AI-GENERATED CONTENT",
                "model": _model_id,
                "agent_id": self.agent_id,
                "generated_at": _ts,
                "watermark": _wm,
                "content_origin": "llm-escalation"
            }
        }

    async def _process_query(
        self,
        message: str,
        context: dict
    ) -> str:
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

                # Sanitize user input to reduce prompt-injection risk.
        # Strip leading/trailing whitespace, collapse excessive newlines,
        # and enforce a maximum length before forwarding to the LLM.
        MAX_INPUT_LENGTH = 2000
        sanitized_message = " ".join(message.split())  # collapse whitespace/newlines
        sanitized_message = sanitized_message[:MAX_INPUT_LENGTH]
        # Reject inputs that contain common prompt-injection patterns.
        _INJECTION_PATTERNS = [
            "ignore previous instructions",
            "ignore all instructions",
            "disregard the above",
            "you are now",
            "act as",
            "jailbreak",
        ]
        lower_msg = sanitized_message.lower()
        if any(pat in lower_msg for pat in _INJECTION_PATTERNS):
            logger.warning("Potential prompt injection detected in user message; request blocked.")
            return "Your request could not be processed. Please rephrase your question."

        response = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sanitized_message}
            ]
        )

        # Validate and sanitize LLM output before returning
        sanitized = self._sanitize_llm_output(response)
        return sanitized

    _DANGEROUS_PATTERNS = [
        # Python dynamic execution
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bcompile\s*\(',
        r'\b__import__\s*\(',
        r'\bimportlib\.import_module\s*\(',
        # Shell / OS execution
        r'\bos\.system\s*\(',
        r'\bos\.popen\s*\(',
        r'\bsubprocess\.(?:call|run|Popen|check_output)\s*\([^)]*shell\s*=\s*True',
        r'\bsubprocess\.(?:call|run|Popen|check_output)\s*\([^)]*shell\s*=\s*1',
        # JavaScript / bash eval
        r'\beval\s*\(',          # JS eval()
        r'\bFunction\s*\(',      # JS new Function()
        r'\bsetTimeout\s*\(',    # JS setTimeout with string
        r'\bsetInterval\s*\(',   # JS setInterval with string
        r'`[^`]*`',              # bash command substitution backticks
        r'\$\([^)]*\)',          # bash $() substitution
        # Other dangerous builtins
        r'\bgetattr\s*\(',
        r'\bsetattr\s*\(',
        r'\bdelattr\s*\(',
        r'\bglobals\s*\(',
        r'\blocals\s*\(',
        r'\bvars\s*\(',
    ]

    def _sanitize_llm_output(self, response: str) -> str:
        """
        Validate and sanitize LLM output.

        Checks for the presence of dynamic code execution primitives
        (eval, exec, subprocess shell=True, JS/bash eval, etc.).
        Raises ValueError if dangerous patterns are detected so that
        the raw, potentially malicious content is never returned to
        the caller.
        """
        import re

        if not isinstance(response, str):
            # Coerce to string for uniform handling
            response = str(response)

        for pattern in self._DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(
                    "Dangerous pattern detected in LLM output; blocking response.",
                    extra={"pattern": pattern}
                )
                raise ValueError(
                    "LLM response contained a potentially dangerous code execution "
                    "primitive and has been blocked for security reasons."
                )

        return response

    @staticmethod
    def _encrypt_pii(value: str) -> str:
        """
        Encrypt a PII string value.

        Uses Fernet symmetric encryption when the cryptography package is
        available.  Falls back to a clearly-marked base64 encoding so the
        field is never stored or transmitted as raw plaintext.

        In production, load the key from a secrets manager / environment
        variable rather than hard-coding it.
        """
        try:
            from cryptography.fernet import Fernet
            import os
            # Retrieve key from environment; generate a one-time key as a
            # safe fallback (rotate / persist properly in production).
            raw_key = os.environ.get("PII_ENCRYPTION_KEY")
            if raw_key:
                key = raw_key.encode() if isinstance(raw_key, str) else raw_key
            else:
                key = Fernet.generate_key()
            f = Fernet(key)
            return f.encrypt(value.encode()).decode()
        except ImportError:
            # cryptography not installed – use base64 as a last resort so
            # the value is at least not stored in plain text.
            import base64
            return "b64:" + base64.b64encode(value.encode()).decode()

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.

        PII fields are encrypted before being stored in the context dict
        so they are never shared or transmitted in plaintext.
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
                # PII encrypted at rest / in transit
                "contact_email": self._encrypt_pii("user@example.com"),
                "phone": self._encrypt_pii("555-123-4567")
            }
        }

        logger.info(
            "Retrieved user context",
            extra={
                # Log only non-sensitive fields to avoid PII leakage in logs
                "user_id": user_context["user_id"],
                "subscription_tier": user_context["subscription_tier"]
            }
        )

        return user_context

        #checking
        #touched
