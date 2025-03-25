"""Exceptions for AxiomMCP."""

import asyncio
from collections.abc import Awaitable
import logging
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

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


class ResourceError(AxiomMCPError):
    """Raised when there are issues with resource operations."""

    def __init__(
        self,
        message: str = "Resource operation failed",
        resource_uri: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with resource-specific context.

        Args:
            message: Error description
            resource_uri: The URI of the resource
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if resource_uri:
            details["resource_uri"] = resource_uri
        super().__init__(message, details=details, **kwargs)


class ResourceNotFoundError(ResourceError):
    """Raised when a resource cannot be found."""

    def __init__(self, uri: str) -> None:
        super().__init__(f"Resource not found: {uri}")


class ResourceUnavailableError(ResourceError):
    """Raised when a resource is unavailable after recovery attempt."""

    def __init__(self) -> None:
        super().__init__("Resource still unavailable after recovery attempt")


class UnknownResourceTypeError(ResourceError):
    """Raised when an unknown resource type is encountered."""

    def __init__(self, resource_type: str) -> None:
        super().__init__(f"Unknown resource type: {resource_type}")


class ResourceReadError(ResourceError):
    """Raised when there's an error reading a resource."""

    def __init__(self, uri: str, error: Exception) -> None:
        super().__init__(f"Error reading resource {uri}: {error}")


class ToolError(AxiomMCPError):
    """Raised when there are issues with tool execution."""

    def __init__(
        self,
        message: str = "Tool execution failed",
        tool_name: str | None = None,
        tool_version: str | None = None,
        execution_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with tool-specific context.

        Args:
            message: Error description
            tool_name: Name of the tool that failed
            tool_version: Version of the tool
            execution_context: Additional context about the tool execution
            **kwargs: Additional details to include
        """
        details = kwargs.pop("details", {})
        if tool_name:
            details["tool_name"] = tool_name
        if tool_version:
            details["tool_version"] = tool_version
        if execution_context:
            details["execution_context"] = execution_context

        super().__init__(message, details=details, **kwargs)


class ToolTimeoutError(ToolError):
    """Raised when a tool execution times out."""

    def __init__(
        self,
        tool_name: str,
        timeout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Tool {tool_name} timed out after {timeout} seconds",
            tool_name=tool_name,
            execution_context={"timeout": timeout},
            **kwargs,
        )


class ToolDependencyError(ToolError):
    """Raised when tool dependencies are missing or invalid."""

    def __init__(
        self,
        tool_name: str,
        missing_deps: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Missing dependencies for tool {tool_name}: {', '.join(missing_deps)}",
            tool_name=tool_name,
            execution_context={"missing_dependencies": missing_deps},
            **kwargs,
        )


class ToolValidationError(ToolError):
    """Raised when tool input/output validation fails."""

    def __init__(
        self,
        tool_name: str,
        validation_type: Literal["input", "output"],
        errors: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"{validation_type.title()} validation failed for tool {tool_name}: {'; '.join(errors)}",
            tool_name=tool_name,
            execution_context={"validation_type": validation_type, "errors": errors},
            **kwargs,
        )


class ToolExecutionError(ToolError):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Tool {tool_name} execution failed: {str(error)}",
            tool_name=tool_name,
            cause=error,
            **kwargs,
        )


class ToolStateError(ToolError):
    """Raised when a tool's state is invalid."""

    def __init__(
        self,
        tool_name: str,
        state_error: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Invalid state for tool {tool_name}: {state_error}",
            tool_name=tool_name,
            execution_context={"state_error": state_error},
            **kwargs,
        )


class ToolStreamError(ToolError):
    """Raised when streaming from a tool fails."""

    def __init__(
        self,
        tool_name: str,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Error streaming from tool {tool_name}: {str(error)}",
            tool_name=tool_name,
            cause=error,
            **kwargs,
        )


class ToolRecoveryStrategy(BaseModel):
    """Configuration for tool error recovery."""

    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
    exponential_backoff: bool = Field(
        default=True, description="Whether to use exponential backoff for retries"
    )
    retry_on_errors: set[type[Exception]] = Field(
        default_factory=lambda: {
            ToolTimeoutError,
            ToolDependencyError,
            ToolValidationError,
            ToolExecutionError,
            ToolStateError,
            ToolStreamError,
        },
        description="Set of error types to retry on",
    )
    recovery_hooks: dict[type[Exception], Callable[[Exception], Awaitable[None]]] = (
        Field(
            default_factory=dict,
            description="Custom recovery functions for specific error types",
        )
    )


class RetryableError(Exception):
    """Base class for errors that support retry with recovery."""

    def __init__(
        self,
        message: str,
        recovery_strategy: ToolRecoveryStrategy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message)
        self.recovery_strategy = recovery_strategy or ToolRecoveryStrategy()
        self.retries = 0
        self.last_error: Exception | None = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from the error.

        Returns:
            bool: Whether recovery was successful
        """
        if self.retries >= self.recovery_strategy.max_retries:
            return False

        self.retries += 1
        try:
            # Look for a custom recovery hook only if we have a last error
            if self.last_error is not None:
                recovery_hook = self.recovery_strategy.recovery_hooks.get(
                    type(self.last_error)
                )
                if recovery_hook:
                    await recovery_hook(self.last_error)

            # Wait with exponential backoff if enabled
            delay = (
                self.recovery_strategy.retry_delay * (2 ** (self.retries - 1))
                if self.recovery_strategy.exponential_backoff
                else self.recovery_strategy.retry_delay
            )
            await asyncio.sleep(delay)
            return True

        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")
            self.last_error = e
            return False


class ConnectionResetError(RetryableError, ConnectionError):
    """Raised when a connection is reset."""

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from connection reset."""
        if await super().attempt_recovery():
            # Add connection-specific recovery logic here
            try:
                # Example: Clear any stale connection state
                return True
            except Exception as e:
                self.last_error = e
                return False
        return False


class StreamInterruptedError(RetryableError, ToolStreamError):
    """Raised when a stream is interrupted."""

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from stream interruption."""
        if await super().attempt_recovery():
            try:
                # Example: Reset stream state and buffers
                return True
            except Exception as e:
                self.last_error = e
                return False
        return False


class ServiceUnavailableError(RetryableError, ConnectionError):
    """Raised when a required service is unavailable."""

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from service unavailability."""
        if await super().attempt_recovery():
            try:
                # Example: Check service health before retrying
                return True
            except Exception as e:
                self.last_error = e
                return False
        return False
