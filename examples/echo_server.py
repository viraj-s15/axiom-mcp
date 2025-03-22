"""Example echo server demonstrating basic axiom_mcp functionality."""

from typing import Any

from axiom_mcp.prompts import Message, Prompt, PromptManager, UserMessage, prompt
from axiom_mcp.tools.base import Tool
from axiom_mcp.tools.utils import tool

# Initialize managers
prompt_manager = PromptManager()


@tool(name="echo_tool", description="Echo a message back", version="1.0.0")
class EchoTool(Tool):
    """Tool to echo messages back."""

    async def execute(self, args: dict[str, Any]) -> str:
        """Execute the echo functionality."""
        return f"Tool echo: {args.get('message', '')}"


@prompt(name="echo_prompt", description="Create an echo prompt", tags=["example"])
def create_echo_prompt(message: str) -> Message:
    """Create a prompt that echoes the message."""
    return UserMessage(role="user", content=f"Echo prompt: {message}")


# Register the prompt with the manager
echo_prompt = Prompt(
    fn=create_echo_prompt,
    name="echo_prompt",
    description="Create an echo prompt",
    tags=["example"],
)
prompt_manager.add_prompt(echo_prompt)

# Example usage


async def main():
    # Execute tool
    echo_tool = EchoTool()
    result = await echo_tool({"message": "Hello world!"})
    print(result)  # Tool echo: Hello world!

    # Render prompt
    messages = await prompt_manager.render_prompt(
        "echo_prompt", {"message": "Hello from prompt!"}
    )
    print(messages[0])  # Echo prompt: Hello from prompt!


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
