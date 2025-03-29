"""Tests for resource utilities."""

import pytest

from axiom_mcp.resources.utils import (
    ResourceType,
    calculate_content_hash,
    create_resource_uri,
    guess_resource_type,
    validate_identifier,
)


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
        ("@test", "Identifier contains invalid characters"),
        ("#test", "Identifier contains invalid characters"),
        ("test@123", "Identifier contains invalid characters"),
    ],
)
def test_validate_identifier_invalid(identifier, error_message):
    """Test that invalid identifiers are rejected with appropriate error messages."""
    with pytest.raises(ValueError, match=error_message):
        validate_identifier(identifier)


def test_validate_identifier_type_check():
    """Test that non-string inputs are rejected."""
    invalid_types = [None, 123, ["test"], {"name": "test"}, b"test"]
    for value in invalid_types:
        with pytest.raises(TypeError, match="Identifier must be a string"):
            validate_identifier(value)


@pytest.mark.parametrize("length", [1, 32, 64])
def test_validate_identifier_length_boundaries(length):
    """Test identifier validation at various valid length boundaries."""
    identifier = "a" * length
    assert validate_identifier(identifier) == identifier


@pytest.mark.parametrize(
    "uri,content,expected_type",
    [
        ("http://example.com/file.txt", None, ResourceType.HTTP),
        ("https://example.com/file.txt", None, ResourceType.HTTP),
        ("file:///path/to/file.txt", None, ResourceType.FILE),
        ("data:text/plain,Hello", None, ResourceType.BINARY),
        ("stream://test", None, ResourceType.STREAM),
        ("function://test", None, ResourceType.FUNCTION),
        ("template://test/{param}", None, ResourceType.TEMPLATE),
        ("memory://test", "text content", ResourceType.TEXT),
        ("memory://test", b"binary content", ResourceType.BINARY),
        ("file.txt", None, ResourceType.TEXT),
        ("file.bin", None, ResourceType.BINARY),
        ("template/{param}/test", None, ResourceType.TEMPLATE),
    ],
)
def test_guess_resource_type(uri, content, expected_type):
    """Test resource type guessing from URI and content."""
    assert guess_resource_type(uri, content) == expected_type


@pytest.mark.parametrize(
    "path,scheme,resource_type,expected",
    [
        ("/path/test.txt", "file", None, "file:///path/test.txt"),
        ("test.txt", "memory", None, "memory://test.txt"),
        ("test", None, ResourceType.TEXT, "text://test"),
        ("test.json", "data", None, "data://test.json"),
        ("/path/with spaces.txt", "file", None, "file:///path/with%20spaces.txt"),
        (
            "template/{param}",
            None,
            ResourceType.TEMPLATE,
            "template://template/{param}",
        ),
    ],
)
def test_create_resource_uri(path, scheme, resource_type, expected):
    """Test resource URI creation with various inputs."""
    result = create_resource_uri(path, scheme, resource_type)
    assert result == expected


def test_calculate_content_hash():
    """Test content hash calculation."""
    text_content = "Hello, World!"
    binary_content = b"Hello, Binary World!"

    text_hash = calculate_content_hash(text_content)
    binary_hash = calculate_content_hash(binary_content)

    # Test that hashes are strings and non-empty
    assert isinstance(text_hash, str)
    assert isinstance(binary_hash, str)
    assert len(text_hash) > 0
    assert len(binary_hash) > 0

    # Test that same content produces same hash
    assert calculate_content_hash(text_content) == text_hash
    assert calculate_content_hash(binary_content) == binary_hash

    # Test that different content produces different hashes
    assert calculate_content_hash("Different content") != text_hash
    assert calculate_content_hash(b"Different binary") != binary_hash
