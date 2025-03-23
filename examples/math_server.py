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
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Adding {a} + {b}")
        result = a + b
        return {
            "operation": "addition",
            "a": a,
            "b": b,
            "result": result
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
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Subtracting {b} from {a}")
        result = a - b
        return {
            "operation": "subtraction",
            "a": a,
            "b": b,
            "result": result
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
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Multiplying {a} * {b}")
        result = a * b
        return {
            "operation": "multiplication",
            "a": a,
            "b": b,
            "result": result
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
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        if b == 0:
            raise ValueError("Cannot divide by zero")
        logger.info(f"Dividing {a} / {b}")
        result = a / b
        return {
            "operation": "division",
            "a": a,
            "b": b,
            "result": result
        }


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
    return f"Math {operation} result: {result['result']}"


@mcp.prompt()
def math_prompt(operation: str, a: float, b: float) -> PromptResponse:
    """Create a math operation prompt"""
    logger.info(f"Math prompt called with: {operation}({a}, {b})")
    return UserMessage(
        content=f"Please calculate {a} {operation} {b}",
        role="user"
    )


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(mcp.run("sse"))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
