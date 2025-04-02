# Axiom MCP

🚀  Robust and Dev friendly MCP framework 

## NOTE

This will be oss very soon, working on docs + other misc stuff, if you want
to contribute, send me an email.

## Installation

Using uv (recommended):
```bash
uv pip install axiom-mcp
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/axiomml/axiom-mcp.git
   cd axiom-mcp
   ```

2. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create and activate a virtual environment with uv:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate  # On Unix/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. Install development dependencies:
   ```bash
   uv sync --frozen --extra dev
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

### 2. Tool Logging and Metrics

Axiom MCP provides comprehensive logging and metrics capabilities for tools:

#### Basic Logging

Each tool has access to a context with built-in logging methods:

```python
class MyTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Access logging through tool context
        self.context.debug("Debug level message")
        self.context.info("Processing started")
        self.context.warning("Warning message")
        self.context.error("Error occurred")

        # Your tool logic here
        return result
```

#### Metrics Tracking

The tool manager automatically tracks:
- Total calls
- Successful calls
- Failed calls
- Average execution time
- Last used timestamp

Metrics are stored in the `logs/tools.log` file by default and include:
- Tool name
- Operation status
- Timestamp
- Execution time
- Error messages (if any)

#### Advanced Features

1. **Cache Control**:
```python
# Disable caching for specific execution
tool_context = ToolContext(cache_enabled=False)
tool = MyTool(context=tool_context)
```

2. **Execution Timeout**:
```python
# Set custom timeout
tool_context = ToolContext(timeout=30.0)  # 30 seconds
```

3. **Dry Run Mode**:
```python
# Enable dry run for testing
tool_context = ToolContext(dry_run=True)
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
axiom-mcp dev server.py # This runs in dev mode
axiom-mcp run server.py # This runs in release/prod mode
```

## Development Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest tests

# Update dependencies
uv add <dep>

# Sync your environment
uv sync
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
