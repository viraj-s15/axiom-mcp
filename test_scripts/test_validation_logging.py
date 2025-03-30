#!/usr/bin/env python
"""Test script to validate the fix for tool validation error logging."""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from axiom_mcp.tools.base import Tool, ToolContext, ToolMetadata, ToolValidation
from axiom_mcp.tools.manager import ToolManager


class ValidationTestTool(Tool):
    """A tool with strict validation to test error logging."""

    metadata = ToolMetadata(
        name="validation_test_tool",
        version="1.0.0",
        description="A tool to test validation error logging",
        validation=ToolValidation(
            input_schema={
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "email": {"type": "string", "format": "email"},
                }
            },
            output_schema={
                "type": "object",
                "required": ["greeting"],
                "properties": {
                    "greeting": {"type": "string"}
                }
            }
        ),
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool."""
        # This will pass input validation but fail output validation
        # by returning something that doesn't match the output schema
        if "invalid_output" in args:
            return {"message": "This will fail output validation"}
            
        return {"greeting": f"Hello, {args['name']}!"}


async def test_validation_errors():
    """Test that validation errors correctly log function arguments."""
    
    # Set up logging directory
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "tools.log"
    
    if log_file.exists():
        # Backup existing log file with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_file = logs_dir / f"tools.log.{timestamp}"
        os.rename(log_file, backup_file)
        print(f"Backed up existing log file to {backup_file}")
    
    # Create tool manager
    tool_manager = ToolManager(
        enable_metrics=True,
        metrics_dir=str(logs_dir)
    )
    
    # Register our test tool
    tool_manager.register_tool(ValidationTestTool)
    
    # Tests to run
    tests = [
        # Test 1: Missing required field (should fail input validation)
        {"test_name": "Missing required field", "args": {"name": "Alice"}},
        
        # Test 2: Wrong type (should fail input validation)
        {"test_name": "Wrong type", "args": {"name": "Bob", "age": "twenty"}},
        
        # Test 3: Below minimum (should fail input validation)
        {"test_name": "Below minimum", "args": {"name": "Charlie", "age": -5}},
        
        # Test 4: Invalid email format (should pass required fields but fail format validation)
        {"test_name": "Invalid format", "args": {"name": "Dave", "age": 30, "email": "not-an-email"}},
        
        # Test 5: Non-serializable argument (should still log safely)
        {"test_name": "Non-serializable", "args": {"name": "Eve", "age": 25, "extra": object()}},
        
        # Test 6: Output validation failure
        {"test_name": "Output validation", "args": {"name": "Frank", "age": 40, "invalid_output": True}},
        
        # Test 7: Successful execution (for comparison)
        {"test_name": "Success case", "args": {"name": "Grace", "age": 35}},
    ]
    
    # Run tests
    for test in tests:
        print(f"\nRunning test: {test['test_name']}")
        try:
            result = await tool_manager.execute_tool("validation_test_tool", test["args"])
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Give time for logs to be written
    await asyncio.sleep(0.5)
    
    # Check log file content
    if log_file.exists():
        print("\n--- Logs analysis ---")
        with open(log_file, "r") as f:
            logs = f.readlines()
        
        successful_logs = 0
        for i, log in enumerate(logs):
            try:
                log_data = json.loads(log.split(" - ")[1])
                if "arguments" in log_data:
                    print(f"\nLog {i+1}: Status '{log_data['status']}'")
                    print(f"  Arguments: {json.dumps(log_data['arguments'], indent=2)}")
                    successful_logs += 1
                    
                    # Check for validation_type field to verify our fix
                    if log_data.get("validation_type"):
                        print(f"  ✅ Validation Type: {log_data['validation_type']}")
                    
                    # Check for error messages
                    if "error" in log_data:
                        print(f"  Error: {log_data['error']}")
                else:
                    print(f"\nLog {i+1}: Missing arguments field")
            except Exception as e:
                print(f"\nLog {i+1}: Error parsing: {str(e)}")
        
        if successful_logs == 0:
            print("❌ No logs with arguments found!")
        else:
            print(f"\n✅ Found {successful_logs} logs with arguments")
    else:
        print("❌ No log file found!")


if __name__ == "__main__":
    asyncio.run(test_validation_errors())