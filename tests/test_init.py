"""Basic tests for the axiom-mcp package."""

from axiom_mcp import __version__


def test_version() -> None:
    """Test version sanity check"""
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"
