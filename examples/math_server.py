"""Example math server demonstrating basic axiom_mcp functionality with simple arithmetic."""
import logging
import sys
from axiom_mcp import AxiomMCP
from axiom_mcp.prompts.base import PromptResponse, Message, UserMessage
from axiom_mcp.tools.base import Tool, ToolMetadata, ToolValidation
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configure MCP with correct port and settings
mcp = AxiomMCP("MathServer",
               warn_on_duplicate_resources=False,
               warn_on_duplicate_prompts=False,
               port=8888,
               debug=True)  # Enable debug mode for better error messages

# Define common input schema for all math operations
number_input_schema = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    },
    "required": ["a", "b"]
}


class AddTool(Tool):
    """Tool for adding two numbers."""
    metadata = ToolMetadata(
        name="add",
        description="Add two numbers together",
        validation=ToolValidation(
            input_schema=number_input_schema
        ),
        author="MathServer",
        version="1.0.0"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Adding {a} + {b}")
        result = a + b
        return {
            "type": "text",
            "content": {
                "operation": "addition",
                "a": a,
                "b": b,
                "result": result
            }
        }


class SubtractTool(Tool):
    """Tool for subtracting numbers."""
    metadata = ToolMetadata(
        name="subtract",
        description="Subtract b from a",
        validation=ToolValidation(
            input_schema=number_input_schema
        ),
        author="MathServer",
        version="1.0.0",  # Added version for compatibility
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Subtracting {b} from {a}")
        result = a - b
        return {
            "type": "text",  # Added for client compatibility
            "content": {
                "operation": "subtraction",
                "a": a,
                "b": b,
                "result": result
            }
        }


class MultiplyTool(Tool):
    """Tool for multiplying numbers."""
    metadata = ToolMetadata(
        name="multiply",
        description="Multiply two numbers",
        validation=ToolValidation(
            input_schema=number_input_schema
        ),
        author="MathServer",
        version="1.0.0",  # Added version
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Multiplying {a} * {b}")
        result = a * b
        return {
            "type": "text",  # Added for client compatibility
            "content": {
                "operation": "multiplication",
                "a": a,
                "b": b,
                "result": result
            }
        }


class DivideTool(Tool):
    """Tool for dividing numbers."""
    metadata = ToolMetadata(
        name="divide",
        description="Divide a by b",
        validation=ToolValidation(
            input_schema=number_input_schema
        ),
        author="MathServer",
        version="1.0.0",  # Added version
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        if b == 0:
            raise ValueError("Cannot divide by zero")
        logger.info(f"Dividing {a} / {b}")
        result = a / b
        return {
            "type": "text",  # Added for client compatibility
            "content": {
                "operation": "division",
                "a": a,
                "b": b,
                "result": result
            }
        }


class MathPromptResponse(PromptResponse):
    """Response wrapper for math prompts."""

    def __init__(self, operation: str, a: float, b: float):
        self.operation = operation
        self.a = a
        self.b = b
        self.__name__ = "MathPromptResponse"

    def __call__(self) -> UserMessage:
        return UserMessage(
            content=f"Please calculate {self.a} {self.operation} {self.b}",
            role="user"
        )


# Register tools with the server
mcp._tool_manager.register_tool(AddTool)
mcp._tool_manager.register_tool(SubtractTool)
mcp._tool_manager.register_tool(MultiplyTool)
mcp._tool_manager.register_tool(DivideTool)


@mcp.resource("math://result/{operation}/{a}/{b}")
async def math_resource(operation: str, a: float, b: float) -> str:
    """Perform a math operation and return the result"""
    tools = {
        "add": AddTool,
        "subtract": SubtractTool,
        "multiply": MultiplyTool,
        "divide": DivideTool
    }

    if operation not in tools:
        raise ValueError(f"Unknown operation: {operation}")

    tool = tools[operation](context=None)
    result = await tool.execute({"a": a, "b": b})
    return f"Math {operation} result: {result['content']['result']}"


@mcp.prompt()
def math_prompt(operation: str, a: float, b: float) -> PromptResponse:
    """Create a math operation prompt"""
    logger.info(f"Math prompt called with: {operation}({a}, {b})")
    return MathPromptResponse(operation, a, b)


if __name__ == "__main__":
    import asyncio
    import sys
    from typing import Literal

    # Default to SSE transport if not specified, explicitly cast to Literal type
    transport_arg = sys.argv[1] if len(sys.argv) > 1 else "sse"
    transport: Literal["stdio",
                       "sse"] = "sse" if transport_arg == "sse" else "stdio"

    try:
        # Support both SSE and stdio transport
        asyncio.run(mcp.run(transport=transport))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
