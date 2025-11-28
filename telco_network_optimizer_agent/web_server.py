"""
Web Server for Telecom Network Optimizer
Provides HTTP API endpoints for the AI agent
"""
import asyncio
import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the agent
try:
    from telco_network_optimizer_agent.agent_unified import create_agent
except ImportError:
    from agent_unified import create_agent

# Create FastAPI app
app = FastAPI(
    title="Telecom Network Optimizer AI Agent",
    description="AI-powered telecom network optimization API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent
    try:
        agent = create_agent()
        print(f"✓ Agent initialized: {agent}")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize agent: {e}")
        print("   The API will still start, but queries will fail.")

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    provider: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: str
    provider: Optional[str] = None
    model: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "operational",
        "service": "Telecom Network Optimizer AI Agent",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "agent_status": "initialized" if agent else "not_initialized"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/info")
async def get_info():
    """Get API information"""
    return {
        "service": "Telecom Network Optimizer AI Agent",
        "version": "1.0.0",
        "capabilities": [
            "Tower Load Management",
            "Spectrum Allocation",
            "User Geo Movement & Demand Prediction",
            "Hardware Monitoring & Maintenance"
        ],
        "providers": ["openai", "openrouter"],
        "models": {
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            "openrouter": ["grok-2-latest", "anthropic/claude-3.5-sonnet"]
        },
        "agent_status": "initialized" if agent else "not_initialized"
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the AI agent
    
    Args:
        request: QueryRequest with query text and optional provider/model
        
    Returns:
        QueryResponse with the agent's response
    """
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please check API key configuration."
        )
    
    try:
        # Get response from agent
        response = await agent.chat(request.query)
        
        return QueryResponse(
            success=True,
            response=response,
            timestamp=datetime.utcnow().isoformat(),
            provider=agent.provider if hasattr(agent, 'provider') else None,
            model=agent.model if hasattr(agent, 'model') else None
        )
    
    except Exception as e:
        return QueryResponse(
            success=False,
            error=str(e),
            timestamp=datetime.utcnow().isoformat()
        )

@app.post("/api/chat")
async def chat(request: Request):
    """
    Alternative chat endpoint with flexible input
    """
    if not agent:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Agent not initialized",
                "message": "Please check API key configuration"
            }
        )
    
    try:
        body = await request.json()
        query = body.get("query") or body.get("message") or body.get("prompt")
        
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "No query provided"}
            )
        
        response = await agent.chat(query)
        
        return {
            "success": True,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "agent": {
            "initialized": agent is not None,
            "provider": agent.provider if agent and hasattr(agent, 'provider') else None,
            "model": agent.model if agent and hasattr(agent, 'model') else None
        },
        "environment": {
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "openrouter_key_set": bool(os.getenv("OPENROUTER_API_KEY"))
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The endpoint {request.url.path} does not exist",
            "available_endpoints": [
                "/",
                "/health",
                "/api/info",
                "/api/query",
                "/api/chat",
                "/api/status"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )

def main():
    """Run the web server"""
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("\n" + "="*70)
    print("  Telecom Network Optimizer - Web Server")
    print("="*70)
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
