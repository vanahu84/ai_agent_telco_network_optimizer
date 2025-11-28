"""
OpenAI-specific Agent for Telecom Network Optimization
Uses the unified agent implementation
"""
import asyncio
import sys
import os

try:
    from telco_network_optimizer_agent.mg_prompt import MERGED_MCP_PROMPT
    from telco_network_optimizer_agent.agent_unified import UnifiedAIAgent
except ImportError:
    from mg_prompt import MERGED_MCP_PROMPT
    from agent_unified import UnifiedAIAgent

# Fix for Windows subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Create the root agent using OpenAI
try:
    root_agent = UnifiedAIAgent(
        provider="openai",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        system_prompt=MERGED_MCP_PROMPT
    )
    print(f"✓ OpenAI agent created: {root_agent}")
except Exception as e:
    print(f"✗ Failed to create OpenAI agent: {e}")
    root_agent = None

# Example usage
async def main():
    if root_agent is None:
        print("✗ No agent available. Please check your OPENAI_API_KEY.")
        return
    
    print("\n" + "="*60)
    print("Testing OpenAI Agent")
    print("="*60 + "\n")
    
    test_query = "Hello! Please introduce yourself and explain what you can do for telecom network optimization."
    
    print(f"Query: {test_query}\n")
    print("Waiting for response...\n")
    
    try:
        response = await root_agent.chat(test_query)
        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

