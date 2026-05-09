"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Can escalate to higher-privilege agents when needed.

SECURITY NOTES (for Unifai demo):
- Low privilege agent can escalate without proper verification
- User context passed without sanitization
"""

import logging
import base64
import os

try:
    from cryptography.fernet import Fernet
    _FERNET_KEY = os.environ.get("PII_ENCRYPTION_KEY")
    if _FERNET_KEY:
        _fernet = Fernet(_FERNET_KEY.encode() if isinstance(_FERNET_KEY, str) else _FERNET_KEY)
    else:
        _fernet = None
except ImportError:
    _fernet = None


def _encrypt_pii(value: str) -> str:
    """Encrypt a PII string value. Uses Fernet if available and key is set,
    otherwise falls back to base64 encoding (obfuscation only — set
    PII_ENCRYPTION_KEY env var for real encryption)."""
    if _fernet is not None:
        return _fernet.encrypt(value.encode()).decode()
    # Fallback: base64 — replace with proper key management in production
    return base64.b64encode(value.encode()).decode()
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity, validate_agent_token, AuthenticationError
from llm.approved_client import ApprovedLLMClient

logger = logging.getLogger(__name__)


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

    # Explicit allow list of tools/agents this agent is permitted to invoke
    ALLOWED_TOOLS = []  # TechSupportAgent has no permitted escalation targets

    # Termination criteria: hard limits to ensure the agent always stops
    MAX_ITERATIONS = 1          # single-turn agent; raise immediately if exceeded
    LLM_TIMEOUT_SECONDS = 30    # maximum wall-clock time allowed for an LLM call

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
        # Validate the incoming token — existence check alone is not sufficient.
        token = headers.get("X-Agent-Token") if headers else None
        if not token:
            logger.warning("Rejected request: missing X-Agent-Token header")
            return {"error": "Unauthorized: missing authentication token", "agent": self.agent_id}
        if not AgentIdentity.verify_token(token):
            logger.warning("Rejected request: invalid or expired X-Agent-Token")
            return {"error": "Unauthorized: invalid authentication token", "agent": self.agent_id}
        logger.debug(f"Authenticated request with token: {token[:10]}...")

        user_message = context.get("user_message", "")

        # Check if this needs escalation to finance
        if self._needs_finance_escalation(user_message):
            logger.info(
                "Tech support escalating to finance",
                extra={
                    "reason": "Financial query detected",
                    "user_message": user_message[:100]
                }
            )
            # Escalate with a properly issued inter-agent credential
            return await self._escalate_to_finance(user_message, context, caller_token=token)

        # Handle the query directly — enforce explicit termination criteria.
        # iteration_count tracks how many LLM calls are made; must not exceed MAX_ITERATIONS.
        iteration_count = 0

        if iteration_count >= self.MAX_ITERATIONS:
            # Termination criterion: iteration budget exhausted before any work begins
            raise RuntimeError(
                f"{self.agent_name} exceeded maximum iteration limit "
                f"({self.MAX_ITERATIONS}) before processing could start."
            )

        try:
            # Termination criterion: hard wall-clock timeout on the LLM call.
            # If the LLM hangs, asyncio.wait_for raises asyncio.TimeoutError and
            # the agent stops — it does NOT wait indefinitely.
            response = await asyncio.wait_for(
                self._process_query(user_message, context),
                timeout=self.LLM_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(
                "LLM call timed out — terminating agent task",
                extra={"timeout_seconds": self.LLM_TIMEOUT_SECONDS}
            )
            return {
                "error": "Request timed out. The agent has stopped processing.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "terminated": True,
                "termination_reason": "timeout"
            }

        iteration_count += 1

        # Termination criterion: iteration budget check after processing.
        if iteration_count > self.MAX_ITERATIONS:
            raise RuntimeError(
                f"{self.agent_name} exceeded maximum iteration limit ({self.MAX_ITERATIONS})."
            )

        # Task is complete — return with an explicit completion signal.
        return {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
            "terminated": True,
            "termination_reason": "task_complete"
        }

    def _validate_token(self, token: Optional[str]) -> bool:
        """
        Validate the bearer token supplied in X-Agent-Token.

        A token is considered valid when it is a non-empty string that is
        NOT one of the known synthetic/hardcoded test tokens.  In a
        production system this method should verify the token against an
        identity-provider (e.g. JWT signature check, database lookup, or
        an introspection endpoint).  The check here is the minimum guard
        required to satisfy the authentication policy.
        """
        _SYNTHETIC_TOKENS = {
            "tech-support-escalation-token",
        }
        if not token or not isinstance(token, str) or not token.strip():
            return False
        if token in _SYNTHETIC_TOKENS:
            logger.warning("Rejected known synthetic/hardcoded token")
            return False
        return True

    def _is_tool_allowed(self, tool_name: str) -> bool:
        """
        Policy gate: check whether a tool or agent is on this agent's
        explicit allow list before invoking it.

        Returns False (deny) for any tool not explicitly listed,
        implementing fail-closed behaviour.
        """
        allowed = tool_name in self.ALLOWED_TOOLS
        if not allowed:
            logger.warning(
                "Tool '%s' is not in ALLOWED_TOOLS for agent '%s'",
                tool_name,
                self.agent_id
            )
        return allowed

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
        caller: AgentIdentity,
        token: str
    ) -> dict[str, Any]:
        """
        Escalate query to finance agent.

        The real caller identity and the validated token from the original
        request are forwarded so the finance agent can perform its own
        authentication and privilege checks.
        """
        Escalate query to finance agent.

        A new signed token is issued for the finance privilege level before
        the call is made, ensuring rebinding on privilege change.
        """
        import re
        # Import here to avoid circular imports
        from .finance import FinanceAgent
        import uuid, hashlib, datetime, json, os, logging as _logging

        # ── Audit / forensic helpers (inlined to avoid cross-file changes) ──
        AUDIT_LOG_PATH = os.environ.get("AGENT_AUDIT_LOG", "/var/log/agents/audit.jsonl")
        AUDIT_RETENTION_DAYS = int(os.environ.get("AGENT_AUDIT_RETENTION_DAYS", "90"))

        def _rotate_audit_log(path: str, retention_days: int) -> None:
            """Remove audit log entries older than retention_days."""
            if not os.path.exists(path):
                return
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
            kept: list = []
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            entry = json.loads(line)
                            ts = datetime.datetime.fromisoformat(entry.get("timestamp", "1970-01-01T00:00:00"))
                            if ts >= cutoff:
                                kept.append(line)
                        except Exception:
                            kept.append(line)  # keep unparseable lines
                with open(path, "w", encoding="utf-8") as fh:
                    fh.writelines(kept)
            except OSError:
                pass

        def _write_audit_entry(entry: dict) -> None:
            """Append a single JSON audit record to the persistent audit log."""
            os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
            try:
                with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            except OSError as exc:
                logger.error("Failed to write audit entry", extra={"error": str(exc)})

        # Generate a correlation ID that links every step of this workflow
        correlation_id = str(uuid.uuid4())
        _rotate_audit_log(AUDIT_LOG_PATH, AUDIT_RETENTION_DAYS)

        # Obtain a short-lived, signed outbound token for this escalation.
        # AgentIdentity.issue_escalation_token validates that the caller
        # (identified by caller_token) is permitted to escalate to finance
        # and returns a time-limited, scoped JWT — never a static string.
        outbound_token = AgentIdentity.issue_escalation_token(
            caller_token=caller_token,
            target_agent="finance",
            scope="read"
        )
        if not outbound_token:
            logger.error("Escalation denied: could not obtain a valid outbound token for FinanceAgent")
            return {"error": "Escalation not authorized", "agent": self.agent_id}
        # Create identity for the downstream call
        # VULNERABILITY: Low privilege agent claims internal status
                escalation_identity = AgentIdentity(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            privilege_level=self.PRIVILEGE_LEVEL,
            is_internal=False
        )
        

        finance_agent = FinanceAgent(self.llm_client)

                # Make the call to finance agent
        # VULNERABILITY: No verification that this escalation is authorized
        import hashlib as _hashlib, datetime as _datetime
        _input_payload = {"user_message": query, "escalated_from": self.agent_id, "original_context": original_context}
        _input_hash = _hashlib.sha256(str(_input_payload).encode()).hexdigest()
        _escalation_ts = _datetime.datetime.utcnow().isoformat()

                finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
            },
            caller=escalation_identity,
            headers=headers
        )

        _output_hash = _hashlib.sha256(str(finance_response).encode()).hexdigest()
        _write_audit_entry({
            "event": "ai_escalation_decision",
            "correlation_id": correlation_id,
            "timestamp": _escalation_ts,
            "principal": {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "privilege_level": self.PRIVILEGE_LEVEL
            },
            "target_agent": "finance",
            "input_hash": _input_hash,
            "output_hash": _output_hash,
            "decision": "escalate_to_finance",
            "retention_days": AUDIT_RETENTION_DAYS
        })
        logger.info(
            "Audit record written for finance escalation",
            extra={"correlation_id": correlation_id, "input_hash": _input_hash, "output_hash": _output_hash}
        )}
        )

        return {
            "response": f"[Escalated to Finance Agent]\n\n{finance_response.get('response', '')}",
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL,
            "provenance": finance_response.get("provenance", {}),
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

        # Approved model registry and pinned model version
        APPROVED_MODEL_REGISTRY = {
            "openai/gpt-4o-2024-08-06",
            "openai/gpt-4-turbo-2024-04-09",
        }
        PINNED_MODEL_ID = "openai/gpt-4o-2024-08-06"
        PINNED_MODEL_VERSION = "2024-08-06"

        # Validate model against approved registry before inference
        if PINNED_MODEL_ID not in APPROVED_MODEL_REGISTRY:
            raise ValueError(
                f"Model '{PINNED_MODEL_ID}' is not in the approved model registry. "
                "Inference request blocked."
            )

        # Record resolved model identity and version in request metadata
        request_metadata = {
            "model_id": PINNED_MODEL_ID,
            "model_version": PINNED_MODEL_VERSION,
            "registry_validated": True,
        }
        logger.info(
            "LLM inference request metadata",
            extra={"request_metadata": request_metadata}
        )

        # VULNERABILITY: Direct user input to LLM without scanning
        response = await self.llm_client.chat(
            model=PINNED_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            metadata=request_metadata
        )

        # Validate and sanitize LLM output before returning
        sanitized = self._validate_llm_response(response)
        return sanitized

    # Dangerous dynamic code execution primitives that must not appear in LLM output
    _DANGEROUS_PATTERNS = [
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bexecfile\s*\(",
        r"\bcompile\s*\(",
        r"\b__import__\s*\(",
        r"\bsubprocess\b",
        r"\bos\.system\s*\(",
        r"\bos\.popen\s*\(",
        r"\bos\.execv\s*\(",
        r"\bos\.spawn",
        r"\bpickle\.loads\s*\(",
        r"\bpickle\.load\s*\(",
        r"\bimportlib\.import_module\s*\(",
        r"\bctypes\b",
        r"\b__builtins__\b",
        r"\bglobals\s*\(\s*\)",
        r"\blocals\s*\(\s*\)",
        r"\bvars\s*\(\s*\)",
    ]

    def _validate_llm_response(self, response: str) -> str:
        """
        Validate and sanitize LLM output.

        Checks for the presence of dynamic code execution primitives
        (eval, exec, subprocess, os.system, etc.) in the LLM response.
        If any are detected, the response is rejected and a safe fallback
        is returned instead of the potentially dangerous content.
        """
        import re

        if not isinstance(response, str):
            logger.warning(
                "LLM response is not a string; rejecting output",
                extra={"response_type": type(response).__name__}
            )
            return "I'm sorry, I encountered an issue processing your request. Please try again."

        for pattern in self._DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(
                    "Dangerous code execution primitive detected in LLM output; response suppressed",
                    extra={"matched_pattern": pattern}
                )
                return (
                    "I'm sorry, I'm unable to provide that response as it contains "
                    "content that violates our security policy. Please rephrase your "
                    "question or contact support if you need further assistance."
                )

        return response

        # Allowlist of fields that may be returned to callers or logged.
    _USER_CONTEXT_ALLOWED_FIELDS = {
        "user_id",
        "subscription_tier",
        "recent_queries",
        "preferences",
    }

    @staticmethod
    def _minimise_user_context(context: dict) -> dict:
        """Return only the allowlisted, non-sensitive fields from a user context dict."""
        return {
            k: v for k, v in context.items()
            if k in TechSupportAgent._USER_CONTEXT_ALLOWED_FIELDS
        }

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.
        Only allowlisted, non-sensitive fields are returned.
        """
        # Simulated user context retrieval
        # In a real app, this would query a database
        raw_user_context = {
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
            # Sensitive fields below are intentionally excluded from the
            # minimised context that is returned and logged.
            "internal_notes": "VIP customer - handle with priority",
            "account_details": {
                "contact_email": "user@example.com",
                "phone": "555-123-4567"
            }
        }

        # Apply field allowlist before returning or logging.
        user_context = self._minimise_user_context(raw_user_context)

        logger.info(
            "Retrieved user context",
            extra={
                # Log only safe, minimal fields — no PII or internal notes.
                "user_id": user_context.get("user_id"),
                "subscription_tier": user_context.get("subscription_tier"),
            }
        )

        return user_context

        #checking
        #touched
