"""
Unified AI Agent for Telecom Network Optimization
Supports both OpenAI and OpenRouter APIs without google-adk dependency
"""
import asyncio
import sys
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from telco_network_optimizer_agent.mg_prompt import MERGED_MCP_PROMPT
except ImportError:
    from mg_prompt import MERGED_MCP_PROMPT

# Fix for Windows subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


class UnifiedAIAgent:
    """
    A unified agent that can use either OpenAI or OpenRouter API
    without dependency on google-adk
    """
    
    def __init__(
        self,
        provider: str = "openai",  # "openai" or "openrouter"
        model: Optional[str] = None,
        system_prompt: str = MERGED_MCP_PROMPT,
        api_key: Optional[str] = None,
        timeout: int = 120
    ):
        self.provider = provider.lower()
        self.system_prompt = system_prompt
        self.timeout = timeout
        
        # Set API configuration based on provider
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.api_url = "https://api.openai.com/v1/chat/completions"
            
            if not self.api_key:
                raise ValueError("⚠ OPENAI_API_KEY not set in environment variables")
                
        elif self.provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.model = model or os.getenv("OPENROUTER_MODEL", "grok-2-latest")
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            
            if not self.api_key:
                raise ValueError("⚠ OPENROUTER_API_KEY not set in environment variables")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'openrouter'")
        
        print(f"✓ Initialized {self.provider.upper()} agent with model: {self.model}")
    
    async def chat(self, user_message: str, **kwargs) -> str:
        """
        Send a message to the AI and get a response
        
        Args:
            user_message: The user's input message
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The AI's response as a string
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Add OpenRouter-specific headers
            if self.provider == "openrouter":
                headers.update({
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Telco Ops AI Agent"
                })
            
            try:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
                print(f"✗ {error_msg}")
                raise
            except Exception as e:
                error_msg = f"Error communicating with {self.provider}: {str(e)}"
                print(f"✗ {error_msg}")
                raise
    
    async def run(self, user_query: str) -> str:
        """Alias for chat() method for compatibility"""
        return await self.chat(user_query)
    
    def __repr__(self):
        return f"<UnifiedAIAgent provider={self.provider} model={self.model}>"


# Auto-detect provider from environment or use OpenAI as default
def create_agent(provider: Optional[str] = None, model: Optional[str] = None) -> UnifiedAIAgent:
    """
    Factory function to create an agent with auto-detection
    
    Args:
        provider: "openai" or "openrouter". If None, auto-detects from env vars
        model: Model name. If None, uses defaults
        
    Returns:
        UnifiedAIAgent instance
    """
    if provider is None:
        # Auto-detect: prefer OpenAI if key exists, otherwise try OpenRouter
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            raise ValueError(
                "No API key found. Set either OPENAI_API_KEY or OPENROUTER_API_KEY"
            )
    
    return UnifiedAIAgent(provider=provider, model=model)


# Create the root agent (auto-detects provider)
try:
    root_agent = create_agent()
    print(f"✓ Root agent created successfully: {root_agent}")
except Exception as e:
    print(f"✗ Failed to create root agent: {e}")
    root_agent = None


# Example usage
async def main():
    if root_agent is None:
        print("✗ No agent available. Please check your API keys.")
        return
    
    print("\n" + "="*60)
    print("Testing Unified AI Agent")
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
