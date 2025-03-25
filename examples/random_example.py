"""Example math server demonstrating basic axiom_mcp functionality with simple arithmetic."""

import logging
from pathlib import Path
import sys
from axiom_mcp import AxiomMCP
from axiom_mcp.prompts.base import PromptResponse, Message, UserMessage
from axiom_mcp.tools.base import Tool, ToolMetadata, ToolValidation, ToolContext
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)

mcp = AxiomMCP(
    "MathServer",
    warn_on_duplicate_resources=False,
    warn_on_duplicate_prompts=False,
    port=8888,
    debug=True,
)

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
        logger.info(f"Adding {a} + {b}")
        result = a + b
        return {
            "type": "text",
            "content": {"operation": "addition", "a": a, "b": b, "result": result},
        }


class SubtractTool(Tool):
    """Tool for subtracting numbers."""

    metadata = ToolMetadata(
        name="subtract",
        description="Subtract b from a",
        validation=ToolValidation(input_schema=number_input_schema),
        author="MathServer",
        version="1.0.0",
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Subtracting {b} from {a}")
        result = a - b
        return {
            "type": "text",
            "content": {"operation": "subtraction", "a": a, "b": b, "result": result},
        }


class MultiplyTool(Tool):
    """Tool for multiplying numbers."""

    metadata = ToolMetadata(
        name="multiply",
        description="Multiply two numbers",
        validation=ToolValidation(input_schema=number_input_schema),
        author="MathServer",
        version="1.0.0",
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        logger.info(f"Multiplying {a} * {b}")
        result = a * b
        return {
            "type": "text",
            "content": {
                "operation": "multiplication",
                "a": a,
                "b": b,
                "result": result,
            },
        }


class DivideTool(Tool):
    """Tool for dividing numbers."""

    metadata = ToolMetadata(
        name="divide",
        description="Divide a by b",
        validation=ToolValidation(input_schema=number_input_schema),
        author="MathServer",
        version="1.0.0",
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = args["a"], args["b"]
        if b == 0:
            raise ValueError("Cannot divide by zero")
        logger.info(f"Dividing {a} / {b}")
        result = a / b
        return {
            "type": "text",
            "content": {"operation": "division", "a": a, "b": b, "result": result},
        }


mcp._tool_manager.register_tool(AddTool)
mcp._tool_manager.register_tool(SubtractTool)
mcp._tool_manager.register_tool(MultiplyTool)
mcp._tool_manager.register_tool(DivideTool)


@mcp.resource("greeting://{name}")
def say_hello(name: str) -> str:
    return f"Hello, {name}!"


@mcp.resource("dir://desktop")
def list_some_random_files() -> list[str]:
    desktop = Path.home() / "Documents"
    return [str(f) for f in desktop.iterdir()]


class ConcretePromptResponse(PromptResponse):
    def __init__(self, messages: list[Message]):
        self.messages = messages
        self.__name__ = "ConcretePromptResponse"

    def __call__(self) -> Message:
        return (
            self.messages[0] if self.messages else UserMessage(content="", role="user")
        )

    def __iter__(self):
        yield from self.messages


@mcp.prompt()
def followMathPrompt(operation: str, a: float, b: float) -> PromptResponse:
    """Follow up on the math prompt"""
    logger.info(f"Follow-up math prompt called with: {operation}({a}, {b})")
    return ConcretePromptResponse(
        messages=[
            UserMessage(content=f"Please calculate {a} {operation} {b}", role="user")
        ]
    )


# ------------------- Only for testing purposes -------------------
# if __name__ == "__main__":
#     import asyncio
#     import sys
#     from typing import Literal

#     # Default to SSE transport if not specified, explicitly cast to Literal type
#     transport_arg = sys.argv[1] if len(sys.argv) > 1 else "sse"
#     transport: Literal["stdio",
#                        "sse"] = "sse" if transport_arg == "sse" else "stdio"

#     try:
#         # Support both SSE and stdio transport
#         asyncio.run(mcp.run(transport=transport))
#     except KeyboardInterrupt:
#         logger.info("Server stopped by user")
#     except Exception as e:
#         logger.error(f"Server error: {e}")
#         raise
