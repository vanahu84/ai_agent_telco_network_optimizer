#!/usr/bin/env python3
"""
Debug User Geo Movement MCP Server
"""

import asyncio
from user_geo_movement_server import app

async def debug_server():
    """Debug the MCP server"""
    print("Debugging User Geo Movement MCP Server...")
    
    try:
        # Test list_tools
        tools = await app.list_tools()
        print(f"Tools registered: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        if tools:
            # Test a tool call
            first_tool = tools[0]
            print(f"\nTesting tool: {first_tool.name}")
            
            if first_tool.name == "analyze_movement_patterns":
                result = await app.call_tool("analyze_movement_patterns", {"area_id": "downtown"})
                print(f"Result: {result[0].text[:200]}...")
            elif first_tool.name == "detect_special_events":
                result = await app.call_tool("detect_special_events", {})
                print(f"Result: {result[0].text[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_server())