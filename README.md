# Axiom MCP

Model Context Protocol (MCP) implementation for connecting AI systems with external data sources.

## Installation

Using uv (recommended):
```bash
uv pip install axiom-mcp
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/axiom-mcp.git
   cd axiom-mcp
   ```

2. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create and activate a virtual environment with uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. Install development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

## Core Features

### 1. Tool Definition

Tools in Axiom MCP are defined as classes that inherit from the `Tool` base class. Here's how to define a tool:

```python
from axiom_mcp.tools.base import Tool, ToolMetadata, ToolValidation

# Define input schema for tool validation
number_input_schema = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"},
    },
    "required": ["a", "b"],
}

class AddTool(Tool):
    """Tool for adding two numbers."""
    metadata = ToolMetadata(
        name="add",
        description="Add two numbers together",
        validation=ToolValidation(input_schema=number_input_schema),
        author="MathServer",
        version="1.0.0",
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        result = a + b
        return {
            "type": "text",
            "content": {"operation": "addition", "a": a, "b": b, "result": result},
        }
```

### 2. Logging Features

Axiom MCP provides built-in logging capabilities:

```python
import logging
import sys

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Usage in tools
logger.info("Operation started")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### 3. Resource Definition

Resources are lightweight endpoints that can be defined using decorators:

```python
from pathlib import Path
from axiom_mcp import AxiomMCP

mcp = AxiomMCP("MyServer", port=8888)

# Simple string resource
@mcp.resource("greeting://{name}")
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

# Resource returning a list of files
@mcp.resource("dir://desktop")
def list_files() -> list[str]:
    desktop = Path.home() / "Documents"
    return [str(f) for f in desktop.iterdir()]
```

## Running the Server

```python
if __name__ == "__main__":
    import asyncio

    # Register your tools
    mcp._tool_manager.register_tool(AddTool)

    # Run with SSE transport
    asyncio.run(mcp.run(transport="sse"))
```

## Development Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=axiom_mcp tests/

# Update dependencies
uv pip compile pyproject.toml -o requirements.txt

# Sync your environment
uv pip sync requirements.txt
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Run the tests:
   ```bash
   uv run pytest
   ```
5. Submit a pull request

## License
GNU General Public License v3 (GPLv3)
