"""Tests for resource utilities."""

import pytest

from axiom_mcp.resources.utils import validate_identifier


@pytest.mark.parametrize(
    "identifier",
    [
        "test",
        "test_name",
        "TestName",
        "test123",
        "_test",
        "a" * 64,  # Test maximum length
        "T_123_test",
        "UPPERCASE_TEST",
    ],
)
def test_validate_identifier_valid(identifier):
    """Test that valid identifiers are accepted."""
    assert validate_identifier(identifier) == identifier


@pytest.mark.parametrize(
    "identifier,error_message",
    [
        ("", "Identifier cannot be empty"),
        ("123test", "Identifier cannot start with a number"),
        ("test!name", "Identifier contains invalid characters"),
        ("test-name", "Identifier contains invalid characters"),
        ("test space", "Identifier contains invalid characters"),
        ("a" * 65, "Identifier exceeds maximum length"),
        ("test.name", "Identifier contains invalid characters"),
        ("$test", "Identifier contains invalid characters"),
        ("\ttest", "Identifier contains invalid characters"),
    ],
)
def test_validate_identifier_invalid(identifier, error_message):
    """Test that invalid identifiers are rejected with appropriate error messages."""
    with pytest.raises(ValueError, match=error_message):
        validate_identifier(identifier)


def test_validate_identifier_type_check():
    """Test that non-string inputs are rejected."""
    invalid_types = [
        None,
        123,
        ["test"],
        {"name": "test"},
        b"test",
    ]
    for value in invalid_types:
        with pytest.raises(TypeError, match="Identifier must be a string"):
            validate_identifier(value)


@pytest.mark.parametrize("length", [1, 32, 64])
def test_validate_identifier_length_boundaries(length):
    """Test identifier validation at various valid length boundaries."""
    identifier = "a" * length
    assert validate_identifier(identifier) == identifier
