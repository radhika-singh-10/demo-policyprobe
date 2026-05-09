"""
Tech Support Agent

Handles general technical support queries with low privilege level.
Can escalate to higher-privilege agents when needed.

SECURITY NOTES (for Unifai demo):
- Low privilege agent can escalate without proper verification
- User context passed without sanitization
"""

import logging
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity
from llm.approved import ApprovedLLMClient

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
        # Guard: refuse re-entry once the task has reached a terminal state.
        if self._task_status in (TaskStatus.COMPLETE, TaskStatus.FAILED):
            raise TaskCompleteError(
                f"Agent '{self.agent_id}' has already completed its task "
                f"(status={self._task_status.value}). Create a new instance for a new task."
            )

        self._task_status = TaskStatus.RUNNING

        # Validate the inbound agent token — reject requests with missing or invalid tokens
        token = headers.get("X-Agent-Token") if headers else None
        if not token:
            logger.warning("Rejected inter-agent request: missing X-Agent-Token header")
            return {"error": "Unauthorized: missing agent token", "status": 401}
        if not AgentIdentity.verify_token(token):
            logger.warning("Rejected inter-agent request: invalid or expired token")
            return {"error": "Unauthorized: invalid agent token", "status": 401}
        logger.debug(f"Received request with validated token: {token[:10]}...")

        user_message = context.get("user_message", "")

        try:
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
                result = await self._escalate_to_finance(user_message, context)
                # --- TERMINATION CRITERION: escalation path complete ---
                self._terminate(TaskStatus.COMPLETE)
                return result

            # Handle the query directly
            response = await self._process_query(user_message, context)

            # --- TERMINATION CRITERION: direct-response path complete ---
            self._terminate(TaskStatus.COMPLETE)
            return {
                "response": response,
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL
            }
        except Exception:
            # Mark the task as failed so the agent does not silently loop.
            self._terminate(TaskStatus.FAILED)
            raise

    # Static policy: only these caller privilege levels may request finance escalation.
    # This list is defined in source and never derived from user-supplied input.
    FINANCE_ESCALATION_ALLOWED_PRIVILEGE_LEVELS: frozenset = frozenset({"admin", "finance"})

        # ---------------------------------------------------------------------------
    # Tool allow list — only tools listed here may be invoked by this agent.
    # Modify this list through a reviewed change process; never derive it from
    # user-supplied input.
    # ---------------------------------------------------------------------------
    TOOL_ALLOW_LIST: frozenset = frozenset()          # FinanceAgent is NOT permitted
    POLICY_VERSION: str = "tech-support-tool-policy-v1"

    def _check_tool_allowed(self, tool_name: str, actor_id: str) -> bool:
        """
        Enforce the explicit tool allow list.

        Emits a structured audit log entry for every check regardless of
        outcome so that denials are always captured.
        """
        allowed = tool_name in self.TOOL_ALLOW_LIST
        audit_entry = {
            "event": "tool_allow_list_check",
            "policy_version": self.POLICY_VERSION,
            "actor_agent_id": actor_id,
            "tool_requested": tool_name,
            "decision": "PERMIT" if allowed else "DENY",
            "deny_reason": None if allowed else (
                f"Tool '{tool_name}' is not in the explicit allow list "
                f"for agent '{self.agent_id}' (policy: {self.POLICY_VERSION})"
            ),
        }
        if allowed:
            logger.info("Tool allow-list check PERMIT", extra=audit_entry)
        else:
            logger.warning("Tool allow-list check DENY", extra=audit_entry)
        return allowed

    def _needs_finance_escalation(self, message: str) -> bool:
        """
        Keyword pre-filter only — does NOT grant permission.

        Actual authorisation is enforced by _check_tool_allowed().
        """
        finance_triggers = [
            "quarterly report", "financial statement", "budget",
            "revenue numbers", "profit margin", "expense report",
            "balance sheet", "cash flow", "earnings"
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in finance_triggers)

    def _caller_is_authorized_for_finance_escalation(self, caller: AgentIdentity) -> bool:
        """Return True only when the caller's privilege level is on the static allowlist.

        The allowlist (FINANCE_ESCALATION_ALLOWED_PRIVILEGE_LEVELS) is defined
        as a class-level constant and is never derived from user-supplied input,
        satisfying the 'baseline comparison' requirement.
        """
        return (
            caller is not None
            and getattr(caller, "privilege_level", None)
            in self.FINANCE_ESCALATION_ALLOWED_PRIVILEGE_LEVELS
        )

    async def _request_human_approval_for_escalation(
        self, query: str, caller: AgentIdentity
    ) -> bool:
        """Human-in-the-loop gate for privilege escalation.

        In production this should integrate with your approval workflow
        (e.g. PagerDuty, Slack approval bot, ticketing system).  The stub
        below always returns False (deny) so that no escalation can occur
        without a real approval integration being wired in.
        """
        logger.warning(
            "Finance escalation requested but human approval integration is not "
            "configured — request denied. caller=%s query_prefix=%s",
            getattr(caller, "agent_id", "unknown"),
            query[:80],
        )
        # TODO: replace with real async approval-workflow call.
        return False

        async def _escalate_to_finance(
        self,
        query: str,
        original_context: dict,
        approved_caller: AgentIdentity,
    ) -> dict[str, Any]:
        """
        Escalate query to finance agent using a properly issued and scoped token.
        The token is verified by the finance agent before processing the request.
        """
        # Import here to avoid circular imports
        from .finance import FinanceAgent

# ---------------------------------------------------------------------------
# Audit / forensic logger — separate from the operational logger so that
# audit records are never mixed with debug noise and can be retained
# independently.  RotatingFileHandler enforces a retention / rotation policy:
#   maxBytes=10 MB per file, backupCount=90 files ≈ 90 days of daily logs.
# In production replace the file path with a SIEM-forwarding handler.
# ---------------------------------------------------------------------------
import logging.handlers as _lh
import os as _os

_AUDIT_LOG_PATH = _os.environ.get("AUDIT_LOG_PATH", "/var/log/policyprobe/tech_support_audit.jsonl")
_os.makedirs(_os.path.dirname(_AUDIT_LOG_PATH), exist_ok=True)

audit_logger = logging.getLogger("audit.tech_support")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # keep audit records out of the root logger

if not audit_logger.handlers:
    _audit_handler = _lh.RotatingFileHandler(
        _AUDIT_LOG_PATH,
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=90,               # retain 90 rotated files (~90 days)
        encoding="utf-8",
    )
    _audit_handler.setFormatter(
        logging.Formatter(
            fmt='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", %(extra_fields)s}',
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )
    audit_logger.addHandler(_audit_handler)

        # Create a properly scoped identity for this escalation — no privilege bypass
        # VULNERABILITY: Low privilege agent claims internal status
                                # Authorization check: only proceed if this agent is explicitly
        # permitted to escalate to finance (configured externally).
        import os
        allowed_escalators = os.environ.get(
            "FINANCE_ESCALATION_ALLOWED_AGENTS", ""
        ).split(",")
        if self.agent_id not in [a.strip() for a in allowed_escalators if a.strip()]:
            logger.warning(
                "Unauthorized escalation attempt blocked",
                extra={"agent_id": self.agent_id}
            )
            return {
                "response": "Escalation to finance agent is not authorized for this agent.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL
            }

        escalation_identity = AgentIdentity(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            privilege_level=self.PRIVILEGE_LEVEL,
            is_internal=False
        )",
            privilege_level="high",
            is_internal=False
        )
        

        finance_agent = FinanceAgent(self.llm_client)

        # Make the call to finance agent
        # VULNERABILITY: No verification that this escalation is authorized
        # Sanitize the query before forwarding to the finance agent
        sanitized_query = self._sanitize_input(query)

                import os
        escalation_token = os.environ.get("TECH_SUPPORT_ESCALATION_TOKEN")
        if not escalation_token:
            raise ValueError("TECH_SUPPORT_ESCALATION_TOKEN environment variable is not set")
                import hashlib, datetime, uuid

        # Reuse trace_id from upstream LLM call if available, else create one
        trace_id = original_context.get("_trace_id", str(uuid.uuid4()))
        escalation_ts = datetime.datetime.utcnow().isoformat() + "Z"
        input_hash = hashlib.sha256(query.encode()).hexdigest()

        # --- AUDIT RECORD: escalation decision (pre-call) ---
        audit_logger.info(
            "escalation_decision",
            extra={
                "event_type": "escalation_decision",
                "trace_id": trace_id,
                "timestamp": escalation_ts,
                "principal": self.agent_id,
                "decision": "escalate_to_finance",
                "input_hash": input_hash,
                "escalated_from": self.agent_id,
                "escalated_to": "finance",
                "privilege_level": self.PRIVILEGE_LEVEL,
            }
        )

                finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context
            },
            caller=escalation_identity,
            headers={"X-Agent-Token": escalation_token}
        )

        finance_output = finance_response.get("response", "")
        output_hash = hashlib.sha256(finance_output.encode()).hexdigest()
        response_ts = datetime.datetime.utcnow().isoformat() + "Z"

        # --- AUDIT RECORD: escalation outcome ---
        audit_logger.info(
            "escalation_outcome",
            extra={
                "event_type": "escalation_outcome",
                "trace_id": trace_id,
                "timestamp": response_ts,
                "principal": self.agent_id,
                "escalated_to": "finance",
                "output_hash": output_hash,
                "output_preview": finance_output[:200],
                "privilege_level": self.PRIVILEGE_LEVEL,
            }
        )

        return {
            "response": f"[Escalated to Finance Agent]\n\n{finance_output}",
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL,
            "_trace_id": trace_id,
        },
            caller=escalation_identity,
            headers={"X-Agent-Token": escalation_token}
        )

        import hashlib
        import hmac
        import json
        import os
        from datetime import datetime, timezone

        escalated_text = f"[Escalated to Finance Agent]\n\n{finance_response.get('response', '')}"

        try:
            model_id = getattr(self.llm_client, "model", "unknown-llm-model")
            generated_at = datetime.now(timezone.utc).isoformat()
            content_origin = f"agent:{self.agent_id}:tech_support:_escalate_to_finance"

            provenance_payload = {
                "model_id": model_id,
                "generated_at": generated_at,
                "content_origin": content_origin,
                "synthetic_content_label": "AI_GENERATED",
            }

            signing_key = os.environ.get("PROVENANCE_SIGNING_KEY", "")
            if not signing_key:
                raise RuntimeError(
                    "PROVENANCE_SIGNING_KEY environment variable is not set; "
                    "cannot sign AI-generated escalation response — blocking."
                )

            canonical = json.dumps(provenance_payload, sort_keys=True, separators=(",", ":"))
            signature = hmac.new(
                signing_key.encode("utf-8"),
                canonical.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            provenance = {
                **provenance_payload,
                "provenance_signature": f"hmac-sha256:{signature}",
            }
        except Exception as provenance_error:
            logger.error(
                "Provenance labeling/signing failed for escalated response — blocking",
                extra={"error": str(provenance_error)},
            )
            raise RuntimeError(
                "Escalated AI-generated content could not be labeled or signed; "
                "response blocked for policy compliance."
            ) from provenance_error

        return {
            "response": escalated_text,
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL,
            "synthetic_content_label": provenance["synthetic_content_label"],
            "provenance": provenance,
        }

    # ---------------------------------------------------------------------------
    # Input validation / sanitization
    # ---------------------------------------------------------------------------
    _MAX_INPUT_LENGTH: int = 4096
    # Patterns that indicate prompt-injection or policy-bypass attempts
    _BLOCKED_PATTERNS: list = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)disregard\s+(all\s+)?previous",
        r"(?i)you\s+are\s+now\s+(?:a|an|the)\b",
        r"(?i)act\s+as\s+(?:a|an|the)\b",
        r"(?i)jailbreak",
        r"(?i)system\s*prompt",
        r"<\s*script[^>]*>",          # XSS / HTML injection
        r"(?i)\bexec\s*\(",           # code-execution patterns
        r"(?i)\beval\s*\(",
    ]

    def _sanitize_input(self, text: str) -> str:
        """
        Validate and sanitize a user-supplied string before it is used
        in an LLM call or forwarded to another agent.

        Raises ValueError if the input is rejected outright; otherwise
        returns a cleaned string safe for downstream use.
        """
        import re
        import html

        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        # 1. Length guard
        if len(text) > self._MAX_INPUT_LENGTH:
            raise ValueError(
                f"Input exceeds maximum allowed length of {self._MAX_INPUT_LENGTH} characters."
            )

        # 2. Reject empty / whitespace-only input
        stripped = text.strip()
        if not stripped:
            raise ValueError("Input must not be empty.")

        # 3. Block known injection / jailbreak patterns
        for pattern in self._BLOCKED_PATTERNS:
            if re.search(pattern, stripped):
                raise ValueError(
                    "Input contains disallowed content and cannot be processed."
                )

        # 4. Escape HTML entities to neutralise markup injection
        sanitized = html.escape(stripped, quote=True)

        return sanitized

    # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
    # Input sanitization helpers
    # ---------------------------------------------------------------------------
    _MAX_MESSAGE_LENGTH: int = 4000  # characters

    # Patterns commonly used in prompt-injection / jailbreak attempts
    _INJECTION_PATTERNS: list = [
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"(?i)you\s+are\s+now\s+(a|an|the)\s+",
        r"(?i)act\s+as\s+(a|an|the)\s+",
        r"(?i)pretend\s+(you\s+are|to\s+be)\s+",
        r"(?i)system\s*:\s*",          # attempts to inject a new system turn
        r"(?i)<\s*/?\s*system\s*>",    # XML-style system tag injection
        r"(?i)\[\s*system\s*\]",       # bracket-style system tag injection
        r"(?i)jailbreak",
        r"(?i)dan\s+mode",
    ]

    def _sanitize_and_validate_message(self, message: str) -> str:
        """
        Sanitize and validate a raw user message before it is forwarded
        to the LLM.

        Steps
        -----
        1. Type check – must be a string.
        2. Strip leading/trailing whitespace.
        3. Empty-message guard.
        4. Length cap to prevent token-flooding / context-window abuse.
        5. Prompt-injection pattern scan.

        Returns the cleaned message string, or raises ValueError if the
        message fails validation.
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
                f"{self._MAX_MESSAGE_LENGTH} characters "
                f"(received {len(sanitized)} characters)."
            )

        for pattern in self._INJECTION_PATTERNS:
            if re.search(pattern, sanitized):
                logger.warning(
                    "Potential prompt-injection attempt detected and blocked.",
                    extra={"pattern": pattern}
                )
                raise ValueError(
                    "Message contains content that is not permitted."
                )

        return sanitized

    # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
    # Prompt-injection / malicious-content sanitization
    # ---------------------------------------------------------------------------
    _B64_RE = re.compile(
        r'(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?'
    )
    # Common shell / OS command patterns
    _SHELL_RE = re.compile(
        r'(?:^|\s|;|&&|\|\|)'
        r'(?:bash|sh|zsh|cmd|powershell|exec|eval|system|popen|subprocess'
        r'|curl|wget|nc|ncat|netcat|chmod|chown|sudo|su|rm\s+-rf'
        r'|dd\s+if|mkfifo|python\s+-c|perl\s+-e|ruby\s+-e|php\s+-r'
        r'|os\.system|os\.popen|subprocess\.)',
        re.IGNORECASE,
    )
    # Prompt-injection trigger phrases
    _INJECTION_RE = re.compile(
        r'(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions'
        r'|disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions'
        r'|you\s+are\s+now\s+(?:a\s+)?(?:DAN|jailbreak|unrestricted)'
        r'|act\s+as\s+(?:if\s+you\s+(?:have\s+no\s+restrictions|are\s+unrestricted))'
        r'|system\s*:\s*you\s+are'
        r'|<\s*(?:script|iframe|object|embed)'
        r'|\\x[0-9a-fA-F]{2}'
        r'|\\u[0-9a-fA-F]{4})',
        re.IGNORECASE,
    )
    # Leetspeak / obfuscation heuristic: high ratio of digit-for-letter substitutions
    _LEET_RE = re.compile(r'(?:[3@][xX]?[3e][cC]|[1!][gG][nN][oO0][rR][3e])', re.IGNORECASE)
    # Binary / non-printable bytes
    _BINARY_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

    _MAX_MESSAGE_LENGTH = 4000  # characters

    def _sanitize_message(self, message: str) -> str:
        """
        Scan *message* for prompt-injection and malicious-content patterns.

        Raises ValueError if clearly malicious content is detected.
        Returns a normalised copy of the message otherwise.
        """
        if not isinstance(message, str):
            raise ValueError("User message must be a string.")

        # 1. Length guard
        if len(message) > self._MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"User message exceeds maximum allowed length "
                f"({self._MAX_MESSAGE_LENGTH} characters)."
            )

        # 2. Binary / non-printable content
        if self._BINARY_RE.search(message):
            raise ValueError(
                "User message contains non-printable or binary characters "
                "and cannot be processed."
            )

        # 3. Prompt-injection trigger phrases
        if self._INJECTION_RE.search(message):
            raise ValueError(
                "User message contains prompt-injection patterns "
                "and cannot be processed."
            )

        # 4. Shell / OS command patterns
        if self._SHELL_RE.search(message):
            raise ValueError(
                "User message contains shell command patterns "
                "and cannot be processed."
            )

        # 5. Leetspeak obfuscation
        if self._LEET_RE.search(message):
            raise ValueError(
                "User message contains obfuscated (leetspeak) content "
                "and cannot be processed."
            )

        # 6. Base64-encoded blobs (potential payload smuggling)
        b64_matches = self._B64_RE.findall(message)
        for candidate in b64_matches:
            try:
                decoded = base64.b64decode(candidate, validate=True).decode("utf-8", errors="replace")
                # If the decoded payload itself looks like a shell command or injection,
                # reject the whole message.
                if self._SHELL_RE.search(decoded) or self._INJECTION_RE.search(decoded):
                    raise ValueError(
                        "User message contains a base64-encoded payload with "
                        "malicious content and cannot be processed."
                    )
            except Exception as exc:
                if "malicious" in str(exc):
                    raise
                # Decoding failed — not valid base64; ignore this candidate.

        # Normalise whitespace to remove zero-width / invisible characters
        sanitized = "".join(ch for ch in message if ch.isprintable() or ch in ("\n", "\t"))
        sanitized = sanitized.strip()

        return sanitized

    async def _process_query(
        self,
        message: str,
        context: dict
    ) -> str:
        """
        Process a general tech support query.

        The user message is sanitized and scanned for prompt-injection,
        shell commands, base64-encoded payloads, leetspeak, and binary
        content before being forwarded to the LLM.
        """
        # Sanitize / validate the incoming message before sending to the LLM.
        sanitized_message = self._sanitize_message(message)

        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        response = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sanitized_message}
            ]
        )

        return response

    # Patterns that indicate dynamic code execution primitives
    _DANGEROUS_PATTERNS = [
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bsubprocess\b",
        r"shell\s*=\s*True",
        r"\bos\.system\s*\(",
        r"\bos\.popen\s*\(",
        r"__import__\s*\(",
        r"\bcompile\s*\(",
        r"\bexecfile\s*\(",
        r"`[^`]+`",           # backtick shell execution (bash/JS)
        r"\$\([^)]+\)",       # $(command) shell substitution
        r"\bFunction\s*\(",   # JS dynamic Function constructor
        r"\bsetTimeout\s*\(", # JS setTimeout with string arg
        r"\bsetInterval\s*\(",
    ]

    def _sanitize_llm_output(self, response: str) -> str:
        """
        Validate and sanitize LLM output.

        Checks for dynamic code execution primitives and raises a
        ValueError if any are detected, preventing potentially
        malicious content from reaching the caller.
        """
        import re

        if not isinstance(response, str):
            logger.warning(
                "LLM response is not a string; coercing to string for safety",
                extra={"response_type": type(response).__name__}
            )
            response = str(response)

        for pattern in self._DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                logger.error(
                    "Dangerous code execution primitive detected in LLM output",
                    extra={"pattern": pattern}
                )
                raise ValueError(
                    "LLM response contained a potentially dangerous code "
                    "execution primitive and was blocked for security reasons."
                )

        return response

    @staticmethod
    def _encrypt_pii(value: str) -> str:
        """
        Encrypt a PII string value.
        Uses Fernet symmetric encryption if the cryptography package is
        available; falls back to a clearly-marked base64 encoding so the
        field is never stored as plain text.
        The encryption key should be loaded from a secrets manager or
        environment variable in production (PII_ENCRYPTION_KEY).
        """
        import os
        import base64
        try:
            from cryptography.fernet import Fernet
            raw_key = os.environ.get("PII_ENCRYPTION_KEY")
            if raw_key:
                key = raw_key.encode() if isinstance(raw_key, str) else raw_key
            else:
                # Derive a stable key from a fixed seed for demo purposes.
                # In production, always use a securely stored key.
                seed = b"policyprobe-pii-key-placeholder!"  # exactly 32 bytes
                key = base64.urlsafe_b64encode(seed)
            f = Fernet(key)
            return f.encrypt(value.encode()).decode()
        except ImportError:
            # cryptography not installed — use base64 as a last resort
            encoded = base64.b64encode(value.encode()).decode()
            return f"b64:{encoded}"

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.
        PII fields are encrypted before being stored in the returned dict.
        """
        # Simulated user context retrieval
        # In a real app, this would query a database
                # Full internal record — never exposed directly
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
            # Sensitive fields (internal_notes, account_details) are
            # intentionally omitted from the returned context.
        }

        # Allowlist: only non-sensitive fields are returned to callers
        _ALLOWED_USER_CONTEXT_FIELDS = {"user_id", "subscription_tier", "recent_queries", "preferences"}
        user_context = {k: v for k, v in _raw_user_context.items() if k in _ALLOWED_USER_CONTEXT_FIELDS}

        logger.info(
            "Retrieved user context",
            extra={
                # Log only the already-filtered context
                "user_context": user_context
            }
        )

        return user_context

import re
import base64

        #checking
        #touched
