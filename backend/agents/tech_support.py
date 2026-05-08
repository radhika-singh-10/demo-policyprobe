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
import base64
from cryptography.fernet import Fernet
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity
from llm.approved import ApprovedLLMClient

import hashlib
import uuid
import datetime
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

# --- Audit logger with retention/rotation policy ---
_audit_logger = logging.getLogger(f"{__name__}.audit")
if not _audit_logger.handlers:
    _audit_handler = RotatingFileHandler(
        "logs/tech_support_audit.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=90,               # retain ~90 rotated files
        encoding="utf-8",
    )
    _audit_handler.setFormatter(
        logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", %(message)s}')
    )
    _audit_logger.addHandler(_audit_handler)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False

# ---------------------------------------------------------------------------
# PII encryption helper
# ---------------------------------------------------------------------------
# The encryption key MUST be stored in an environment variable (or a secrets
# manager) — never hard-coded.  Generate once with Fernet.generate_key() and
# set the result as PII_ENCRYPTION_KEY in your environment.
_PII_KEY_ENV = "PII_ENCRYPTION_KEY"
_raw_key = os.environ.get(_PII_KEY_ENV)
if _raw_key is None:
    raise EnvironmentError(
        f"Required environment variable '{_PII_KEY_ENV}' is not set. "
        "Generate a key with Fernet.generate_key() and export it."
    )
_fernet = Fernet(_raw_key.encode() if isinstance(_raw_key, str) else _raw_key)


def _encrypt_pii(value: str) -> str:
    """Return a base64-encoded Fernet-encrypted representation of *value*."""
    return _fernet.encrypt(value.encode()).decode()


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

    # Explicit allow list of tools/agents this agent is permitted to invoke.
    # Any tool not present here must never be called.
    ALLOWED_TOOLS: list[str] = []  # FinanceAgent is NOT on the allow list for a low-privilege agent

    def __init__(self, llm_client: ApprovedLLMClient):
        self.llm_client = llm_client
        self.agent_id = "tech_support"
        self.agent_name = "Tech Support Agent"

    # ---------------------------------------------------------------------------
    # Termination helpers
    # ---------------------------------------------------------------------------

    def _is_task_complete(self, response: str) -> bool:
        """Return True only when the agent has produced a non-empty, usable reply.

        This is the single, authoritative exit criterion: a task is considered
        complete if and only if the LLM returned a non-blank string.  All other
        outcomes (empty string, None, whitespace-only) are treated as INCOMPLETE
        and must NOT be returned to the caller as a finished result.
        """
        return bool(response and response.strip())

    def _build_result(
        self,
        response: str,
        status: TaskStatus,
        extra: Optional[dict] = None
    ) -> dict[str, Any]:
        """Build a standardised response dict that always carries a termination flag."""
        result: dict[str, Any] = {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
            # Explicit termination signal consumed by the orchestrator.
            "task_complete": status in (TaskStatus.COMPLETE, TaskStatus.ESCALATED, TaskStatus.FAILED),
            "task_status": status.value,
        }
        if extra:
            result.update(extra)
        return result

    # ---------------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------------

    async def handle(
        self,
        context: dict[str, Any],
        caller: AgentIdentity,
        headers: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Handle incoming request from orchestrator or direct call.

        Termination criteria
        --------------------
        The agent considers its task **complete** (``task_complete=True``) when
        exactly one of the following conditions is met:

        1. **COMPLETE** – ``_process_query`` returned a non-empty LLM response.
        2. **ESCALATED** – the query was handed off to the finance agent; this
           agent must stop executing immediately after returning.
        3. **FAILED** – an unrecoverable error occurred; the agent stops and
           surfaces the error to the caller.

        If none of these conditions is met the status is **INCOMPLETE** and
        ``task_complete`` is ``False``, signalling the orchestrator that the
        agent did not finish.

        Args:
            context: Request context with user message and metadata
            caller: Identity of the calling agent/user
            headers: Request headers (including auth token)

        Returns:
            Response dictionary with mandatory ``task_complete`` and
            ``task_status`` fields.
        """
        token = headers.get("X-Agent-Token") if headers else None
        if not token or not AgentIdentity.verify_token(token):
            logger.warning("Rejected request: missing or invalid agent token")
            return {
                "error": "Unauthorized: invalid or missing agent token",
                "agent": self.agent_id
            }
                    logger.debug("Received request with a valid agent token")

        user_message = context.get("user_message", "")

        # --- Termination criterion 2: escalation ---
        if self._needs_finance_escalation(user_message):
            logger.info(
                "Tech support escalating to finance",
                extra={
                    "reason": "Financial query detected",
                    "user_message": user_message[:100]
                }
            )
            # VULNERABILITY: Escalating to high-privilege agent without proper auth
            escalation_result = await self._escalate_to_finance(user_message, context)
            # Merge the escalation payload but enforce our termination fields.
            escalation_result["task_complete"] = True
            escalation_result["task_status"] = TaskStatus.ESCALATED.value
            logger.info("Task terminated: escalated to finance agent.")
            return escalation_result

        # --- Termination criterion 1: successful LLM response ---
        try:
            response = await self._process_query(user_message, context)
        except Exception as exc:  # noqa: BLE001
            logger.error("_process_query raised an exception: %s", exc)
            # --- Termination criterion 3: unrecoverable failure ---
            return self._build_result(
                response="An internal error occurred. Please try again later.",
                status=TaskStatus.FAILED,
            )

        if self._is_task_complete(response):
            logger.info("Task terminated: LLM returned a valid response.")
            return self._build_result(response=response, status=TaskStatus.COMPLETE)

        # --- No termination criterion met ---
        logger.warning(
            "Task did not complete: LLM returned an empty or unusable response."
        )
        return self._build_result(
            response="I was unable to generate a response. Please rephrase your question.",
            status=TaskStatus.INCOMPLETE,
        )

    def _is_tool_allowed(self, tool_name: str) -> bool:
        """
        Check whether a tool or agent is on the explicit allow list.

        This is the single enforcement gate for all tool/agent invocations.
        Returns False (fail-closed) if the tool is not explicitly permitted.

        Args:
            tool_name: The name of the tool or agent to check.

        Returns:
            True only if tool_name is in ALLOWED_TOOLS.
        """
        allowed = tool_name in self.ALLOWED_TOOLS
        if not allowed:
            logger.warning(
                "Tool invocation blocked by allow list",
                extra={"tool": tool_name, "agent": self.agent_id, "allowed_tools": self.ALLOWED_TOOLS}
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
        caller: AgentIdentity
    ) -> dict[str, Any]:
        """
        Escalate query to finance agent.

        Only permitted when the originating caller has sufficient privilege.
        The caller's verified identity is forwarded; no bypass flags are used.
        """
        # Import here to avoid circular imports
        from .finance import FinanceAgent

        # Create a properly authenticated escalation identity
        # VULNERABILITY: Low privilege agent claims internal status
                escalation_token = AgentIdentity.issue_token(
            agent_id=self.agent_id,
            role="tech_support",
            privilege_level="low"
        )
        escalation_identity = AgentIdentity(
            agent_id=self.agent_id,
            role="tech_support",
            privilege_level="low",
            is_internal=False
        )
        

        finance_agent = FinanceAgent(self.llm_client)

        # Sanitize query before escalating to finance agent
        sanitized_query = self._sanitize_input(query)
                # --- Audit: generate shared trace ID for the full escalation chain ---
        escalation_trace_id = str(uuid.uuid4())
        escalation_ts = datetime.datetime.utcnow().isoformat() + "Z"
        _audit_logger.info(
            '"event": "escalation_start", "trace_id": "%s", "from_agent": "%s", '
            '"to_agent": "finance", "principal": "%s", "timestamp": "%s"',
            escalation_trace_id, self.agent_id,
            getattr(escalation_identity, "agent_id", "unknown"),
            escalation_ts,
        )

        # Make the call to finance agent
        # VULNERABILITY: No verification that this escalation is authorized
        finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context,
                "trace_id": escalation_trace_id,   # propagate causal chain
            },
            caller=escalation_identity,
            headers={
                "X-Agent-Token": "tech-support-escalation-token",
                "X-Trace-Id": escalation_trace_id,  # propagate in transport headers
            }
        )

        # --- Audit: record finance agent response under the same trace ID ---
        completion_ts = datetime.datetime.utcnow().isoformat() + "Z"
        _audit_logger.info(
            '"event": "escalation_complete", "trace_id": "%s", "from_agent": "%s", '
            '"to_agent": "finance", "response_keys": "%s", "timestamp": "%s"',
            escalation_trace_id, self.agent_id,
            list(finance_response.keys()) if isinstance(finance_response, dict) else "non-dict",
            completion_ts,
        )}
        )

        import datetime as _dt
        return {
            "response": f"[Escalated to Finance Agent]\n\n{finance_response.get('response', '')}",
            "ai_generated": True,
            "content_label": "AI-GENERATED CONTENT",
            "provenance": {
                "model_id": getattr(self.llm_client, "model", "unknown-model"),
                "agent_id": self.agent_id,
                "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
                "origin": "llm-chat-completion-escalated",
            },
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL
        }

    @staticmethod
    def _sanitize_input(text: str, max_length: int = 4000) -> str:
        """Sanitize user input before passing to LLM or downstream agents.

        - Strips leading/trailing whitespace
        - Removes null bytes and ASCII control characters (except newline/tab)
        - Truncates to max_length to prevent prompt-stuffing / DoS
        """
        import re
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        # Remove null bytes
        text = text.replace("\x00", "")
        # Remove ASCII control characters except \t (0x09) and \n (0x0A)
        text = re.sub(r"[\x01-\x08\x0b-\x1f\x7f]", "", text)
        # Collapse excessively repeated whitespace lines (anti-jailbreak padding)
        text = re.sub(r"(\n){4,}", "\n\n\n", text)
        # Enforce maximum length
        text = text[:max_length]
        return text.strip()

        # Maximum allowed length for a user message sent to the LLM.
    _MAX_MESSAGE_LENGTH: int = 4000

    # Patterns that indicate prompt-injection or system-prompt override attempts.
    _INJECTION_PATTERNS: list = [
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"(?i)you\s+are\s+now",
        r"(?i)act\s+as\s+(a\s+)?(?!helpful)",
        r"(?i)new\s+system\s+prompt",
        r"(?i)\[system\]",
        r"(?i)<\s*system\s*>",
        r"(?i)jailbreak",
        r"(?i)do\s+anything\s+now",
        r"(?i)dan\s+mode",
    ]

    def _sanitize_and_validate_message(self, message: str) -> str:
        """
        Sanitize and validate a user-supplied message before it is forwarded
        to the LLM.

        Steps performed:
        1. Type check — must be a string.
        2. Strip leading/trailing whitespace.
        3. Non-empty check.
        4. Length check — must not exceed _MAX_MESSAGE_LENGTH characters.
        5. Prompt-injection scan — reject messages that contain patterns
           commonly used to override system instructions.

        Returns the sanitized message string, or raises ValueError if the
        message fails any validation step.
        """
        import re

        if not isinstance(message, str):
            raise ValueError("User message must be a string.")

        sanitized = message.strip()

        if not sanitized:
            raise ValueError("User message must not be empty.")

        if len(sanitized) > self._MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"User message exceeds maximum allowed length of "
                f"{self._MAX_MESSAGE_LENGTH} characters."
            )

        for pattern in self._INJECTION_PATTERNS:
            if re.search(pattern, sanitized):
                raise ValueError(
                    "User message contains disallowed content and cannot be processed."
                )

        return sanitized

    async def _process_query(
        self,
        message: str,
        context: dict
    ) -> str:
        """
        Process a general tech support query.

        The user message is sanitized and validated before being forwarded
        to the LLM to prevent prompt injection and other input-based attacks.
        """
        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        # Sanitize and validate the user message before sending to the LLM.
        try:
            sanitized_message = self._sanitize_and_validate_message(message)
        except ValueError as exc:
            logger.warning(
                "User message rejected during input validation: %s", exc
            )
            return "Your message could not be processed. Please revise your input and try again."

                # Re-validate the message immediately before sending to the LLM to
        # guard against any path that bypasses the check in handle().
        try:
            sanitize_user_input(message)
        except ValueError as exc:
            logger.warning("Rejected malicious user input in _process_query(): %s", exc)
            raise

        response = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )

        return response

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
        r'\beval\s*\(',           # JS eval()
        r'\bFunction\s*\(',       # JS new Function()
        r'\bsetTimeout\s*\(',     # JS setTimeout with string
        r'\bsetInterval\s*\(',    # JS setInterval with string
        r'`[^`]*\$\{',            # JS template literal injection
        r'\$\(\s*["\']',          # bash command substitution
        r'`[^`]+`',               # backtick shell execution
    ]

    def _validate_llm_output(self, response: str) -> str:
        """
        Validate and sanitize LLM output.

        Checks for the presence of dynamic code execution primitives
        such as eval, exec, subprocess(shell=True), os.system, and
        JavaScript/bash eval patterns. Raises a ValueError if any
        dangerous pattern is detected so the raw unsafe output is
        never returned to the caller.
        """
        import re

        if not isinstance(response, str):
            # Coerce to string for uniform handling
            response = str(response)

        for pattern in self._DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(
                    "Dangerous pattern detected in LLM output; response blocked.",
                    extra={"matched_pattern": pattern}
                )
                raise ValueError(
                    "LLM response contained a potentially dangerous code execution "
                    "primitive and has been blocked for security reasons."
                )

        return response

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.

        VULNERABILITY: Returns full user context including potentially
        sensitive information without filtering.
        """
        # Simulated user context retrieval
        # In a real app, this would query a database
                # Full internal record — never exposed outside this method
        _raw_user_context = {
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
                "contact_email": "user@example.com",
                "phone": "555-123-4567"
            }
        }

        # Allowlist: only non-sensitive fields are retained for output
        _ALLOWED_USER_CONTEXT_FIELDS = {"user_id", "subscription_tier", "recent_queries", "preferences"}
        user_context = {k: v for k, v in _raw_user_context.items() if k in _ALLOWED_USER_CONTEXT_FIELDS}

        # Log only the redacted context — no PII or internal metadata
        logger.info(
            "Retrieved user context",
            extra={
                "user_context": user_context
            }
        )

        return user_context

        #checking
        #touched
