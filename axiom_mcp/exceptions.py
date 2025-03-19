"""Custom exceptions for Axiom MCP."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AxiomMCPError(Exception):
    """Base exception class for all Axiom MCP related errors."""

    def __init__(
        self,
        message: str = "An error occurred in Axiom MCP",
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception with optional details and cause.

        Args:
            message: Human-readable error description
            details: Additional context about the error
            cause: The original exception that caused this one
        """
        self.message = message
        self.details = details or {}
        self.cause = cause

        self._log_error()

        full_message = self._build_message()
        super().__init__(full_message)

    def _build_message(self) -> str:
        """Build a detailed error message from all available information."""
        message_parts = [self.message]

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            message_parts.append(f"Details: {details_str}")

        if self.cause:
            cause_info = f"{type(self.cause).__name__}: {str(self.cause)}"
            message_parts.append(f"Caused by: {cause_info}")

        return " | ".join(message_parts)

    def _log_error(self) -> None:
        """Log detailed error information."""
        logger.error(
            "%s: %s",
            self.__class__.__name__,
            self.message,
            exc_info=self.cause,
            extra=self.details,
        )


class ConnectionError(AxiomMCPError):
    """Raised when there are issues connecting to external data sources."""

    def __init__(
        self,
        message: str = "Failed to connect to external data source",
        endpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with connection-specific context.

        Args:
            message: Error description
            endpoint: The URL or identifier of the connection target
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, details=details, **kwargs)


class ModelError(AxiomMCPError):
    """Raised when there are issues with AI model operations."""

    def __init__(
        self,
        message: str = "Model operation failed",
        model_id: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with model-specific context.

        Args:
            message: Error description
            model_id: Identifier for the AI model
            operation: The operation being performed (e.g., "inference")
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if model_id:
            details["model_id"] = model_id
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)


class ProtocolError(AxiomMCPError):
    """Raised when there are MCP protocol-specific issues."""

    def __init__(
        self,
        message: str = "MCP protocol error",
        protocol_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with protocol-specific context.

        Args:
            message: Error description
            protocol_version: The MCP protocol version in use
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if protocol_version:
            details["protocol_version"] = protocol_version
        super().__init__(message, details=details, **kwargs)


class DataError(AxiomMCPError):
    """Raised when there are problems with data handling."""

    def __init__(
        self,
        message: str = "Data handling error",
        data_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with data-specific context.

        Args:
            message: Error description
            data_source: The source of the problematic data
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if data_source:
            details["data_source"] = data_source
        super().__init__(message, details=details, **kwargs)


class AuthenticationError(ConnectionError):
    """Raised when authentication with an external service fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        service: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with authentication-specific context.

        Args:
            message: Error description
            service: The service authentication failed for
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if service:
            details["service"] = service
        super().__init__(message, details=details, **kwargs)


class ModelResponseError(ModelError):
    """Raised when AI model responses are invalid or unexpected."""

    def __init__(
        self,
        message: str = "Invalid or unexpected model response",
        response_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with response-specific context.

        Args:
            message: Error description
            response_type: The type of response that was invalid
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if response_type:
            details["response_type"] = response_type
        super().__init__(message, details=details, **kwargs)


class LambdaNameError(AxiomMCPError):
    """Raised when a lambda function is provided without a name."""

    def __init__(self) -> None:
        super().__init__("Lambda functions must be provided with a name")


class MissingArgumentsError(AxiomMCPError):
    """Raised when required arguments are missing."""

    def __init__(self, missing_args: list[str]):
        super().__init__(f"Missing required arguments: {', '.join(missing_args)}")


class InvalidMessageRoleError(AxiomMCPError):
    """Raised when an invalid message role is provided."""

    def __init__(self, role: str):
        super().__init__(f"Invalid message role: {role}")


class UnknownPromptError(AxiomMCPError):
    """Raised when attempting to use an unknown prompt."""

    def __init__(self, name: str):
        super().__init__(f"Unknown prompt: {name}")


class PromptRenderError(AxiomMCPError):
    """Raised when there is an error rendering a prompt."""

    def __init__(self, name: str, error: str):
        super().__init__(f"Error rendering prompt '{name}': {error}")
