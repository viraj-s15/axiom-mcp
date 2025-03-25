# Save this as test_axiom.py in your project root
import sys
import os

print("*" * 50)
print("TESTING AXIOM MCP")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print("*" * 50)

try:
    # Try to import and run the echo server directly
    sys.path.insert(0, os.getcwd())
    print("Importing echo_server...")
    from examples.echo_server import mcp

    print(f"Server object: {mcp}")
    print(f"Server settings: {mcp.settings}")

    # Try to start the server directly
    import asyncio

    print("Running server...")
    # Force debug mode and log all details
    mcp.settings.debug = True
    mcp.settings.log_level = "DEBUG"

    # Get server info
    print(f"Tools: {asyncio.run(mcp.list_tools())}")
    print(f"Prompts: {asyncio.run(mcp.list_prompts())}")
    print(f"Resources: {asyncio.run(mcp.list_resources())}")

    # Use an alternative port to avoid conflicts
    mcp.settings.port = 8888
    print(f"Starting server on port {mcp.settings.port}...")

    # Configure server with explicit logging
    asyncio.run(mcp.run(transport="sse"))
except Exception as e:
    import traceback

    print(f"ERROR: {e}")
    traceback.print_exc()
