"""Example echo server demonstrating basic axiom_mcp functionality."""

import logging
import sys
from axiom_mcp import AxiomMCP
from axiom_mcp.prompts.base import PromptResponse, Message, UserMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


class EchoPromptResponse(PromptResponse):
    def __init__(self, message: str):
        self.message = message
        self.__name__ = "EchoPromptResponse"

    def __call__(self) -> UserMessage:
        return UserMessage(content=self.message, role="user")


# Configure MCP with correct port
mcp = AxiomMCP(
    "Echo",
    warn_on_duplicate_resources=False,
    warn_on_duplicate_prompts=False,
    port=8888,
)  # Set port to match client expectation


@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    logger.info(f"Resource called with message: {message}")
    return f"Resource echo: {message}"


@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    logger.info(f"Tool called with message: {message}")
    return f"Tool echo: {message}"


@mcp.prompt()
def echo_prompt(message: str) -> PromptResponse:
    """Create an echo prompt"""
    logger.info(f"Prompt called with message: {message}")
    return EchoPromptResponse(f"Please process this message: {message}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run("sse"))
