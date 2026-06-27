"""Domain-specific agent failures."""


class AgentError(RuntimeError):
    """Raised when orchestration fails after user input validated successfully."""
