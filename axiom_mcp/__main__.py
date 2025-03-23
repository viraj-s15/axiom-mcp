"""Entry point for running axiom_mcp as a module."""
from .cli import main
import sys

# When running as a module, we need to set this flag
sys.argv.insert(0, "-m")


if __name__ == "__main__":
    main()
