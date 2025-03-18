"""Tests for the exceptions module."""

from axiom_mcp.exceptions import (
    AuthenticationError,
    AxiomMCPError,
    ConnectionError,
    DataError,
    ModelError,
    ModelResponseError,
    ProtocolError,
)


class TestAxiomMCPError:
    """Test the base exception class."""

    def test_default_initialization(self) -> None:
        """Test creating exception with default parameters."""
        exc = AxiomMCPError()
        assert str(exc) == "An error occurred in Axiom MCP"
        assert exc.message == "An error occurred in Axiom MCP"
        assert exc.details == {}
        assert exc.cause is None

    def test_custom_message(self) -> None:
        """Test creating exception with custom message."""
        exc = AxiomMCPError("Custom error")
        assert str(exc) == "Custom error"
        assert exc.message == "Custom error"

    def test_with_details(self) -> None:
        """Test creating exception with details."""
        details = {"key": "value", "number": 42}
        exc = AxiomMCPError("Error with details", details=details)
        assert "Error with details" in str(exc)
        assert "Details: key=value, number=42" in str(exc)
        assert exc.details == details

    def test_with_cause(self) -> None:
        """Test creating exception with cause."""
        cause = ValueError("Original error")
        exc = AxiomMCPError("Wrapper error", cause=cause)
        assert "Wrapper error" in str(exc)
        assert "Caused by: ValueError: Original error" in str(exc)
        assert exc.cause is cause


class TestConnectionError:
    """Test connection error class."""

    def test_default_initialization(self) -> None:
        """Test creating connection error with defaults."""
        exc = ConnectionError()
        assert "Failed to connect to external data source" in str(exc)

    def test_with_endpoint(self) -> None:
        """Test creating connection error with endpoint."""
        exc = ConnectionError(endpoint="https://api.example.com")
        assert "Failed to connect to external data source" in str(exc)
        assert "endpoint=https://api.example.com" in str(exc)
        assert exc.details["endpoint"] == "https://api.example.com"


class TestModelError:
    """Test model error class."""

    def test_with_model_info(self) -> None:
        """Test creating model error with model info."""
        exc = ModelError(model_id="gpt-4", operation="inference")
        assert "Model operation failed" in str(exc)
        assert "model_id=gpt-4" in str(exc)
        assert "operation=inference" in str(exc)
        assert exc.details["model_id"] == "gpt-4"
        assert exc.details["operation"] == "inference"


class TestProtocolError:
    """Test protocol error class."""

    def test_with_version(self) -> None:
        """Test creating protocol error with version."""
        exc = ProtocolError(protocol_version="1.0")
        assert "MCP protocol error" in str(exc)
        assert "protocol_version=1.0" in str(exc)
        assert exc.details["protocol_version"] == "1.0"


class TestDataError:
    """Test data error class."""

    def test_with_source(self) -> None:
        """Test creating data error with source."""
        exc = DataError(data_source="database")
        assert "Data handling error" in str(exc)
        assert "data_source=database" in str(exc)
        assert exc.details["data_source"] == "database"


class TestAuthenticationError:
    """Test authentication error class."""

    def test_inheritance(self) -> None:
        """Test that AuthenticationError inherits from ConnectionError."""
        exc = AuthenticationError()
        assert isinstance(exc, ConnectionError)

    def test_with_service(self) -> None:
        """Test creating auth error with service."""
        exc = AuthenticationError(service="OAuth")
        assert "Authentication failed" in str(exc)
        assert "service=OAuth" in str(exc)
        assert exc.details["service"] == "OAuth"


class TestModelResponseError:
    """Test model response error class."""

    def test_inheritance(self) -> None:
        """Test that ModelResponseError inherits from ModelError."""
        exc = ModelResponseError()
        assert isinstance(exc, ModelError)

    def test_with_response_type(self) -> None:
        """Test creating response error with type."""
        exc = ModelResponseError(response_type="JSON")
        assert "Invalid or unexpected model response" in str(exc)
        assert "response_type=JSON" in str(exc)
        assert exc.details["response_type"] == "JSON"
