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
from cryptography.fernet import Fernet
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity
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
            logger.warning("Rejected inter-agent request: missing X-Agent-Token header")
            return {
                "error": "Unauthorized: missing agent token",
                "agent": self.agent_id
            }
        verified_identity = verify_agent_token(token)
        if verified_identity is None:
            logger.warning("Rejected inter-agent request: invalid or expired agent token")
            return {
                "error": "Unauthorized: invalid agent token",
                "agent": self.agent_id
            }
        logger.debug(f"Authenticated inter-agent request from: {verified_identity.agent_id}")

        user_message = context.get("user_message", "")

        # Financial queries must be handled by the finance team directly.
        # Automated privilege escalation to the FinanceAgent is not permitted;
        # a human-in-the-loop approval step is required before any cross-agent
        # escalation can occur.
        if self._needs_finance_escalation(user_message):
            logger.info(
                "Tech support detected financial query; directing user to finance team",
                extra={"user_message": user_message[:100]}
            )
            return {
                "response": (
                    "Your question appears to involve financial data. "
                    "Please contact the Finance team directly or submit a "
                    "formal escalation request through the approved workflow "
                    "so that a human reviewer can authorise access."
                ),
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "escalation_required": True
            }

        # Handle the query directly
        response = await self._process_query(user_message, context)

        return {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL
        }

    # ------------------------------------------------------------------ #
    # Token helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _token_secret() -> bytes:
        """Return the shared HMAC secret from the environment."""
        import os
        secret = os.environ.get("AGENT_TOKEN_SECRET", "")
        if not secret:
            raise RuntimeError(
                "AGENT_TOKEN_SECRET environment variable is not set; "
                "cannot sign or verify agent tokens."
            )
        return secret.encode()

    def _generate_escalation_token(self) -> str:
        """
        Generate a signed, expiry-bearing, subject-bound escalation token.

        Format (pipe-separated, then HMAC-SHA256 appended):
            <subject>|<issued_at>|<expires_at>|<hmac_hex>
        """
        import hmac as _hmac
        import hashlib
        import time

        subject = f"{self.agent_id}->finance"
        issued_at = int(time.time())
        expires_at = issued_at + 300  # 5-minute validity window
        payload = f"{subject}|{issued_at}|{expires_at}"
        mac = _hmac.new(
            self._token_secret(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{payload}|{mac}"

    def _verify_token(self, token: str) -> bool:
        """
        Verify an incoming agent token.

        Checks:
        - Correct number of fields
        - HMAC-SHA256 signature matches (constant-time comparison)
        - Token has not expired
        """
        import hmac as _hmac
        import hashlib
        import time

        try:
            parts = token.split("|")
            if len(parts) != 4:
                return False
            subject, issued_at_str, expires_at_str, received_mac = parts
            payload = f"{subject}|{issued_at_str}|{expires_at_str}"
            expected_mac = _hmac.new(
                self._token_secret(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            if not _hmac.compare_digest(expected_mac, received_mac):
                return False
            if int(time.time()) > int(expires_at_str):
                return False
            return True
        except Exception:
            return False

        # Explicit allow list of agent/tool targets this agent may escalate to.
    # Any target NOT in this set is denied regardless of message content.
    ALLOWED_ESCALATION_TARGETS: frozenset = frozenset()  # finance escalation disabled by default

    # Increment this version string whenever the allow list changes so audit
    # records can be correlated with the policy that was in effect.
    POLICY_VERSION: str = "tech-support-v1"

    def _needs_finance_escalation(self, message: str) -> bool:
        """Check if message requires finance agent access."""
        finance_triggers = [
            "quarterly report", "financial statement", "budget",
            "revenue numbers", "profit margin", "expense report",
            "balance sheet", "cash flow", "earnings"
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in finance_triggers)

    def _assert_escalation_allowed(self, target: str, actor_id: str) -> None:
        """
        Enforce the explicit tool/agent allow list.

        Raises PermissionError and writes a structured audit record if
        *target* is not in ALLOWED_ESCALATION_TARGETS.
        """
        if target not in self.ALLOWED_ESCALATION_TARGETS:
            audit_record = {
                "event": "escalation_denied",
                "actor_agent_id": actor_id,
                "requested_target": target,
                "allowed_targets": sorted(self.ALLOWED_ESCALATION_TARGETS),
                "policy_version": self.POLICY_VERSION,
                "denial_reason": f"Target '{target}' is not in the explicit allow list for agent '{actor_id}'",
            }
            logger.warning("[AUDIT] Tool escalation denied", extra=audit_record)
            raise PermissionError(
                f"Escalation to '{target}' denied: not in allow list "
                f"(policy {self.POLICY_VERSION}, actor {actor_id})"
            )

        # _escalate_to_finance has been removed.
    # Automated cross-agent privilege escalation is prohibited by policy.
    # Any escalation to the FinanceAgent must be initiated through the
    # human-supervised escalation workflow with explicit approval.
        )

        # Sanitize the query before forwarding to the finance agent
        sanitized_query = self._sanitize_input(query)

                # Sanitize query before escalating to FinanceAgent
        safe_query = self._sanitize_prompt(query)

        # Make the call to finance agent
        finance_response = await finance_agent.handle(
            context={
                "user_message": safe_query,
                "escalated_from": self.agent_id,
                "original_context": self._minimise(original_context, self._FINANCE_CONTEXT_ALLOWLIST)
            },
            caller=escalation_identity,
            headers={"X-Agent-Token": os.environ.get("TECH_SUPPORT_ESCALATION_TOKEN", "")}
        )

        return {
            "response": f"[Escalated to Finance Agent]\n\n{self._minimise(finance_response, self._FINANCE_RESPONSE_ALLOWLIST).get('response', '')}",
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL
        }

    @staticmethod
    def _sanitize_input(text: str, max_length: int = 4096) -> str:
        """
        Validate and sanitize user-supplied input before forwarding
        to LLM or downstream agents.

        - Rejects non-string values.
        - Strips null bytes and other non-printable control characters.
        - Truncates to max_length to prevent prompt-stuffing / DoS.
        - Removes common prompt-injection patterns.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        # Strip null bytes and ASCII control characters (except tab/newline/CR)
        import re
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Truncate to maximum allowed length
        sanitized = sanitized[:max_length]

        # Remove common prompt-injection patterns (case-insensitive)
        injection_patterns = [
            r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
            r"(?i)disregard\s+(all\s+)?previous\s+instructions?",
            r"(?i)you\s+are\s+now\s+(?:a|an)\s+",
            r"(?i)system\s*:\s*",
            r"(?i)<\s*/?\s*(?:system|assistant|user)\s*>",
        ]
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized)

        return sanitized.strip()

        # ---------------------------------------------------------------------------
    # Input sanitization helpers
    # ---------------------------------------------------------------------------
    _MAX_MESSAGE_LENGTH: int = 4000  # characters

    # Patterns that are commonly used in prompt-injection attacks.
    _INJECTION_PATTERNS: list = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard your instructions",
        "forget your instructions",
        "you are now",
        "act as",
        "pretend you are",
        "jailbreak",
        "dan mode",
        "developer mode",
        "override instructions",
        "system prompt",
        "<|im_start|>",
        "<|im_end|>",
        "[system]",
        "[/system]",
    ]

    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize and validate a user-supplied message before it is forwarded
        to the LLM.

        Steps
        -----
        1. Type check – must be a non-empty string.
        2. Length check – truncate to _MAX_MESSAGE_LENGTH characters.
        3. Strip null bytes and ASCII control characters (except common
           whitespace: tab, newline, carriage-return).
        4. Prompt-injection keyword scan – raise ValueError if a known
           injection phrase is detected.

        Returns the sanitized string.
        """
        if not isinstance(message, str):
            raise ValueError("User message must be a string.")

        message = message.strip()

        if not message:
            raise ValueError("User message must not be empty.")

        # --- 1. Length check ---
        if len(message) > self._MAX_MESSAGE_LENGTH:
            logger.warning(
                "User message truncated from %d to %d characters.",
                len(message),
                self._MAX_MESSAGE_LENGTH,
            )
            message = message[: self._MAX_MESSAGE_LENGTH]

        # --- 2. Strip null bytes and dangerous control characters ---
        # Keep \t (0x09), \n (0x0A), \r (0x0D); remove everything else < 0x20
        # and the DEL character (0x7F).
        import re as _re
        message = _re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", message)

        # --- 3. Prompt-injection keyword scan ---
        lower_message = message.lower()
        for pattern in self._INJECTION_PATTERNS:
            if pattern in lower_message:
                logger.warning(
                    "Potential prompt-injection attempt detected. Pattern: '%s'",
                    pattern,
                )
                raise ValueError(
                    "Message contains disallowed content and cannot be processed."
                )

        return message

    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Prompt-safety helper
    # ---------------------------------------------------------------------------
    _MALICIOUS_PATTERNS = [
        # Shell / OS commands
        r"(?i)(\b(rm|wget|curl|chmod|chown|sudo|su|bash|sh|zsh|python|perl|ruby|nc|ncat|netcat|exec|eval|system|popen|subprocess)\s)",
        # Hidden / override prompt patterns
        r"(?i)(ignore (previous|all|above)|disregard (previous|all|above)|new (instruction|prompt|system)|you are now|act as|pretend (you are|to be)|forget (your|all)|override (your|the))",
        # Base64-encoded blobs (≥20 chars of base64 alphabet)
        r"[A-Za-z0-9+/]{20,}={0,2}",
        # Leetspeak substitution patterns (common swaps)
        r"(?i)(3x3c|3v4l|5y5t3m|sh3ll|c0mm4nd)",
        # Binary / null bytes
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
        # Prompt-delimiter injection
        r"(###|\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>)",
    ]

    import re as _re

    def _sanitize_prompt(self, text: str) -> str:
        """
        Raise ValueError if the text contains patterns associated with
        prompt injection, hidden commands, base64 payloads, leetspeak,
        shell commands, or binary content.
        Returns the original text unchanged when it is considered safe.
        """
        for pattern in self._MALICIOUS_PATTERNS:
            if self._re.search(pattern, text):
                raise ValueError(
                    f"Prompt safety check failed: input matches disallowed pattern '{pattern}'. "
                    "The request has been blocked."
                )
        # Length guard – extremely long inputs are suspicious
        if len(text) > 8000:
            raise ValueError(
                "Prompt safety check failed: input exceeds maximum allowed length."
            )
        return text

    # ---------------------------------------------------------------------------

        # Approved model registry: only these pinned model identifiers are permitted.
    APPROVED_MODEL_REGISTRY: dict = {
        "openai/gpt-4o-2024-05-13": {
            "provider": "openai",
            "family": "gpt-4o",
            "approved": True,
        },
        "anthropic/claude-3-5-sonnet-20241022": {
            "provider": "anthropic",
            "family": "claude-3-5-sonnet",
            "approved": True,
        },
    }

    # Pinned model identifier — must exist in APPROVED_MODEL_REGISTRY.
    PINNED_MODEL_ID: str = "openai/gpt-4o-2024-05-13"

    def _validate_model_registry(self, model_id: str) -> None:
        """Raise ValueError if model_id is not in the approved registry."""
        entry = self.APPROVED_MODEL_REGISTRY.get(model_id)
        if entry is None or not entry.get("approved"):
            raise ValueError(
                f"Model '{model_id}' is not in the approved model registry. "
                "Update PINNED_MODEL_ID to an approved identifier."
            )

    async def _process_query(
        self,
        message: str,
        context: dict
    ) -> str:
        """
        Process a general tech support query.

        Model identity is pinned to PINNED_MODEL_ID, validated against
        APPROVED_MODEL_REGISTRY before every invocation, and recorded in
        the request metadata returned alongside the response.
        """
        # --- Registry & version-pin enforcement ---
        model_id = self.PINNED_MODEL_ID
        self._validate_model_registry(model_id)
        registry_entry = self.APPROVED_MODEL_REGISTRY[model_id]

        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        # Record model identity metadata before the call so it is always
        # available for audit regardless of whether the call succeeds.
        model_metadata = {
            "pinned_model_id": model_id,
            "provider": registry_entry["provider"],
            "family": registry_entry["family"],
            "registry_approved": registry_entry["approved"],
        }
        logger.info("LLM invocation", extra={"model_metadata": model_metadata})

        # Pass the pinned model identifier explicitly to the client.
        response = await self.llm_client.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            metadata=model_metadata,
        )

        return response

    # Patterns indicating dynamic code execution primitives that must not appear in LLM output
    _DANGEROUS_PATTERNS = [
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bexecfile\s*\(',
        r'\bcompile\s*\(',
        r'\b__import__\s*\(',
        r'\bimportlib\.import_module\s*\(',
        r'\bsubprocess\s*\..*shell\s*=\s*True',
        r'\bos\.system\s*\(',
        r'\bos\.popen\s*\(',
        r'\bpopen\s*\(',
        r'\bcall\s*\(.*shell\s*=\s*True',
        r'\bcheck_output\s*\(.*shell\s*=\s*True',
        r'\bcheck_call\s*\(.*shell\s*=\s*True',
        r'\brun\s*\(.*shell\s*=\s*True',
        r'\bgetattr\s*\(.*,\s*[\'"]__',
        r'\bsetattr\s*\(',
        r'\bdelattr\s*\(',
        r'<script[^>]*>',
        r'javascript\s*:',
        r'\bFunction\s*\(',
        r'\bsetTimeout\s*\(',
        r'\bsetInterval\s*\(',
        r'\$\(.*\)\.html\s*\(',
        r'\binnerHTML\s*=',
        r'\bdocument\.write\s*\(',
        r'\bsh\s+-c\b',
        r'\bbash\s+-c\b',
        r'\bpython\s+-c\b',
        r'\bnode\s+-e\b',
        r'\bperl\s+-e\b',
        r'\bruby\s+-e\b',
    ]

    def _validate_llm_output(self, response: str) -> str:
        """
        Validate and sanitize LLM output.

        Checks for the presence of dynamic code execution primitives
        (eval, exec, subprocess shell=True, JS eval, bash eval, etc.).
        Raises ValueError if dangerous patterns are detected so the caller
        can handle the error safely instead of propagating malicious content.
        """
        import re

        if not isinstance(response, str):
            logger.warning(
                "LLM response is not a string; converting to string for safety",
                extra={"response_type": type(response).__name__}
            )
            response = str(response)

        for pattern in self._DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                logger.error(
                    "Dangerous code execution primitive detected in LLM output",
                    extra={"pattern": pattern}
                )
                raise ValueError(
                    "LLM response contains a potentially dangerous code execution "
                    "primitive and has been blocked for security reasons."
                )

        return response

    @staticmethod
    def _get_pii_cipher() -> "Fernet":
        """
        Return a Fernet cipher instance keyed from the environment.
        Falls back to a deterministic key derived from a secret env var so
        that the same key is used across calls within a process.  In
        production, PII_ENCRYPTION_KEY should be a securely generated
        Fernet key stored in a secrets manager.
        """
        raw_key = os.environ.get("PII_ENCRYPTION_KEY")
        if raw_key:
            # Expect a URL-safe base64-encoded 32-byte key (standard Fernet format)
            key = raw_key.encode() if isinstance(raw_key, str) else raw_key
        else:
            # Derive a stable fallback key from a secret seed (dev/test only)
            seed = os.environ.get("PII_KEY_SEED", "change-me-in-production")
            key = base64.urlsafe_b64encode(seed.encode().ljust(32)[:32])
        return Fernet(key)

    @staticmethod
    def _encrypt_pii(value: str, cipher: "Fernet") -> str:
        """Encrypt a PII string and return a base64-encoded ciphertext string."""
        return cipher.encrypt(value.encode()).decode()

        # Allowlist of fields that may be returned to callers.
    _USER_CONTEXT_ALLOWLIST = {"user_id", "subscription_tier", "recent_queries", "preferences"}

    # Allowlist of fields that may be forwarded to the finance agent.
    _FINANCE_CONTEXT_ALLOWLIST = {"user_id", "subscription_tier"}

    # Allowlist of fields that may be returned from the finance agent.
    _FINANCE_RESPONSE_ALLOWLIST = {"response", "status"}

    @staticmethod
    def _minimise(data: dict, allowlist: set) -> dict:
        """Return a copy of *data* containing only keys in *allowlist*."""
        return {k: v for k, v in data.items() if k in allowlist}

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.

        Only non-sensitive fields defined in _USER_CONTEXT_ALLOWLIST are
        returned; PII and internal notes are never exposed to callers.
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
            # Sensitive fields — never leave this method
            "internal_notes": "VIP customer - handle with priority",
            "account_details": {
                "contact_email": "user@example.com",
                "phone": "555-123-4567"
            }
        }

        # Apply allowlist before logging or returning
        user_context = self._minimise(raw_user_context, self._USER_CONTEXT_ALLOWLIST)

        logger.info(
            "Retrieved user context",
            extra={
                "user_context": self._minimise(user_context, self._USER_CONTEXT_ALLOWLIST)  # minimised — no PII
            }
        )

        return user_context

        #checking
        #touched
