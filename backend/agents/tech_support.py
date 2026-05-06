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
from enum import Enum
from typing import Any, Optional

from .auth.agent_auth import AgentIdentity
from llm.approved_registry import get_approved_llm_client, ApprovedLLMClientInterface

logger = logging.getLogger(__name__)

# Fernet encryption for PII fields
try:
    from cryptography.fernet import Fernet
    _FERNET_KEY = os.environ.get("PII_ENCRYPTION_KEY")
    if _FERNET_KEY:
        _fernet = Fernet(_FERNET_KEY.encode() if isinstance(_FERNET_KEY, str) else _FERNET_KEY)
    else:
        _fernet_key_generated = Fernet.generate_key()
        _fernet = Fernet(_fernet_key_generated)
except ImportError:
    _fernet = None


def _encrypt_pii(value: str) -> str:
    """Encrypt a PII string value using Fernet symmetric encryption."""
    if _fernet is None:
        raise RuntimeError("cryptography library is required for PII encryption")
    return _fernet.encrypt(value.encode()).decode()


class TaskStatus(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ESCALATED = "escalated"
    FAILED = "failed"


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
    MAX_INPUT_LENGTH = 4096
    MAX_ITERATIONS = 5

    # Prompt injection patterns to block
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
        re.compile(r"system\s*prompt", re.IGNORECASE),
        re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
        re.compile(r"\[INST\]", re.IGNORECASE),
    ]

    # Dangerous code execution patterns for LLM output validation
    DANGEROUS_OUTPUT_PATTERNS = [
        re.compile(r"\beval\s*\(", re.IGNORECASE),
        re.compile(r"\bexec\s*\(", re.IGNORECASE),
        re.compile(r"\bcompile\s*\(", re.IGNORECASE),
        re.compile(r"\b__import__\s*\(", re.IGNORECASE),
        re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
        re.compile(r"\bsubprocess\b.*shell\s*=\s*True", re.IGNORECASE),
        re.compile(r"\bexecfile\s*\(", re.IGNORECASE),
    ]

    def __init__(self, llm_client: ApprovedLLMClientInterface):
        self.llm_client = llm_client
        self.agent_id = "tech_support"
        self.agent_name = "Tech Support Agent"

    def sanitize_input(self, text: str) -> str:
        """
        Strip whitespace, enforce max length, remove null bytes/non-printable
        control characters, and block common prompt-injection patterns.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove null bytes and non-printable control characters (except newline/tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Enforce maximum length
        if len(text) > self.MAX_INPUT_LENGTH:
            raise ValueError(f"Input exceeds maximum allowed length of {self.MAX_INPUT_LENGTH} characters")

        if not text:
            raise ValueError("Input must not be empty after sanitization")

        # Block prompt injection patterns
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(text):
                raise ValueError("Input contains disallowed prompt-injection pattern")

        return text

    def _sanitize_input(self, message: str) -> str:
        """
        Sanitize and validate input: strip whitespace, enforce max length,
        remove null bytes and non-printable control characters, raise ValueError
        for empty or oversized input.
        """
        if not isinstance(message, str):
            raise ValueError("Input must be a string")

        message = message.strip()

        # Remove null bytes and non-printable control characters (except newline/tab)
        message = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', message)

        if not message:
            raise ValueError("Input must not be empty after sanitization")

        if len(message) > self.MAX_INPUT_LENGTH:
            raise ValueError(f"Input exceeds maximum allowed length of {self.MAX_INPUT_LENGTH} characters")

        # Block prompt injection patterns
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(message):
                raise ValueError("Input contains disallowed prompt-injection pattern")

        return message

    def _validate_llm_output(self, output: str) -> str:
        """
        Validate LLM output for dangerous code execution primitives.
        Returns the output if safe, raises ValueError if dangerous patterns detected.
        """
        for pattern in self.DANGEROUS_OUTPUT_PATTERNS:
            if pattern.search(output):
                logger.warning(
                    "Dangerous pattern detected in LLM output; returning safe fallback",
                    extra={"pattern": pattern.pattern}
                )
                return "I'm sorry, I cannot provide that response as it contains potentially unsafe content."
        return output

    def _is_task_complete(self, response: str, status: TaskStatus) -> bool:
        """
        Evaluate whether the agent has enough information to produce a final answer.
        """
        if status in (TaskStatus.COMPLETE, TaskStatus.ESCALATED, TaskStatus.FAILED):
            return True
        if response and len(response.strip()) > 0:
            return True
        return False

    def _validate_incoming_token(self, token: str) -> bool:
        """
        Validate the incoming agent token against the auth subsystem.
        """
        from .auth.agent_auth import verify_agent_token
        try:
            return verify_agent_token(token)
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return False

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
        # Validate the incoming token against the auth subsystem
        token = headers.get("X-Agent-Token") if headers else None
        if token:
            if not self._validate_incoming_token(token):
                logger.warning("Received request with invalid or unverifiable agent token")
                return {
                    "response": "Authentication failed: invalid agent token.",
                    "agent": self.agent_id,
                    "privilege_level": self.PRIVILEGE_LEVEL,
                    "status": TaskStatus.FAILED.value
                }
            logger.debug("Incoming agent token validated successfully")
        else:
            logger.warning("Received request without agent token")
            return {
                "response": "Authentication failed: missing agent token.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "status": TaskStatus.FAILED.value
            }

        raw_user_message = context.get("user_message", "")

        # Sanitize user input before any processing
        try:
            user_message = self.sanitize_input(raw_user_message)
        except ValueError as e:
            logger.warning(f"Input sanitization failed: {e}")
            return {
                "response": "Your request could not be processed due to invalid input.",
                "agent": self.agent_id,
                "privilege_level": self.PRIVILEGE_LEVEL,
                "status": TaskStatus.FAILED.value
            }

        status = TaskStatus.IN_PROGRESS
        response = ""
        iterations = 0
        completed = False

        while not completed and iterations < self.MAX_ITERATIONS:
            iterations += 1

            # Check if this needs escalation to finance
            if self._needs_finance_escalation(user_message):
                logger.info(
                    "Tech support escalating to finance",
                    extra={
                        "reason": "Financial query detected",
                        "user_message": user_message[:100]
                    }
                )
                result = await self._escalate_to_finance(user_message, context)
                status = TaskStatus.ESCALATED
                completed = True
                return {**result, "status": status.value}

            # Handle the query directly
            response = await self._process_query(user_message, context)
            status = TaskStatus.COMPLETE

            if self._is_task_complete(response, status):
                completed = True

        if not completed:
            logger.warning("Max iterations reached without task completion")
            status = TaskStatus.FAILED

        return {
            "response": response,
            "agent": self.agent_id,
            "privilege_level": self.PRIVILEGE_LEVEL,
            "status": status.value,
            "completed": completed
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
        original_context: dict
    ) -> dict[str, Any]:
        """
        Escalate query to finance agent with proper authentication.
        """
        # Import here to avoid circular imports
        from .finance import FinanceAgent
        from .auth.agent_auth import issue_agent_token

        # Obtain a properly issued, verifiable credential from the auth subsystem
        escalation_token = issue_agent_token(
            agent_id=self.agent_id,
            privilege_level=self.PRIVILEGE_LEVEL
        )

        # Create identity without fabricated is_internal bypass
        escalation_identity = AgentIdentity(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            privilege_level=self.PRIVILEGE_LEVEL,
            is_internal=False
        )

        finance_agent = FinanceAgent(self.llm_client)

        # Make the call to finance agent with a properly issued token
        finance_response = await finance_agent.handle(
            context={
                "user_message": query,
                "escalated_from": self.agent_id,
                "original_context": original_context
            },
            caller=escalation_identity,
            headers={"X-Agent-Token": escalation_token}
        )

        return {
            "response": f"[Escalated to Finance Agent]\n\n{finance_response.get('response', '')}",
            "agent": self.agent_id,
            "escalated_to": "finance",
            "privilege_level": self.PRIVILEGE_LEVEL
        }

    async def _process_query(
        self,
        message: str,
        context: dict
    ) -> str:
        """
        Process a general tech support query.
        """
        # Sanitize and validate input before sending to LLM
        message = self._sanitize_input(message)

        system_prompt = """You are a helpful technical support agent for PolicyProbe.
You can help users with:
- General questions about the application
- Technical troubleshooting
- Document analysis guidance
- Policy compliance questions

Be helpful, professional, and concise in your responses."""

        messages_to_send = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        logger.info(
            "Sending request to LLM",
            extra={"messages": messages_to_send}
        )

        response = await self.llm_client.chat(
            messages=messages_to_send
        )

        logger.info(
            "Received response from LLM",
            extra={"response": response}
        )

        # Validate LLM output for dangerous code execution primitives
        response = self._validate_llm_output(response)

        return response

    async def get_user_context(self, user_id: str) -> dict:
        """
        Retrieve user context for personalized support.
        PII fields are encrypted before being stored or returned.
        """
        # Simulated user context retrieval
        # In a real app, this would query a database
        raw_contact_email = "user@example.com"
        raw_phone = "555-123-4567"

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
                "contact_email": _encrypt_pii(raw_contact_email),
                "phone": _encrypt_pii(raw_phone)
            }
        }

        # Log only non-sensitive fields — no PII
        logger.info(
            "Retrieved user context",
            extra={
                "user_context": {
                    "user_id": user_context["user_id"],
                    "subscription_tier": user_context["subscription_tier"],
                    "preferences": user_context["preferences"]
                }
            }
        )

        return user_context

        #checking
        #touched