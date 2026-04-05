"""FastAPI server for PolicyGuard-RL environment."""

import asyncio
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from policyguard.environment import PolicyGuardEnv
from policyguard.models import (
    PolicyGuardAction,
    PolicyGuardObservation,
    PolicyGuardState,
    Decision,
    ContentCategory,
)
from policyguard.grader import grade_episode


# Global environment instance
env: Optional[PolicyGuardEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize environment on startup."""
    global env
    env = PolicyGuardEnv()
    yield
    if env:
        await env.close()


app = FastAPI(
    title="PolicyGuard-RL",
    description="Multi-agent content moderation environment with 3-tier escalation chain",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    """Request body for /reset."""
    task: Optional[str] = None
    difficulty: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    """Request body for /step."""
    decision: str
    category: str
    confidence: float = 0.8
    reasoning: str = ""
    policy_cited: str = ""
    bias_identified: str = ""


class TaskInfo(BaseModel):
    """Info about available tasks."""
    name: str
    difficulty: str
    description: str
    scenario_count: int


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "policyguard-rl"}


@app.get("/info")
async def info():
    """Environment info."""
    tasks = env.get_available_tasks() if env else {}
    return {
        "name": "PolicyGuard-RL",
        "version": "1.0.0",
        "description": "Multi-agent content moderation with 3-tier escalation",
        "tasks": [
            TaskInfo(
                name=name,
                difficulty=config.get("difficulty", "medium"),
                description=config.get("description", ""),
                scenario_count=config.get("scenario_count", 0),
            ).model_dump()
            for name, config in tasks.items()
        ],
        "tiers": ["tier_1", "tier_2", "tier_3"],
        "decisions": ["allow", "remove", "reduce", "label", "escalate"],
        "categories": [
            "misinformation", "hate_speech", "child_safety",
            "harassment", "violence", "spam", "none"
        ],
    }


@app.post("/reset")
async def reset(request: ResetRequest) -> Dict[str, Any]:
    """Reset environment for a new episode."""
    if not env:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    try:
        obs = await env.reset(
            task=request.task,
            difficulty=request.difficulty,
            options=request.options,
        )
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """Take a step in the environment."""
    if not env:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    try:
        # Validate decision
        if request.decision not in [d.value for d in Decision]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid decision: {request.decision}. Must be one of: {[d.value for d in Decision]}"
            )
        
        # Validate category
        if request.category not in [c.value for c in ContentCategory]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category: {request.category}. Must be one of: {[c.value for c in ContentCategory]}"
            )
        
        action = PolicyGuardAction(
            decision=Decision(request.decision),
            category=ContentCategory(request.category),
            confidence=request.confidence,
            reasoning=request.reasoning,
            policy_cited=request.policy_cited,
            bias_identified=request.bias_identified,
        )
        
        obs = await env.step(action)
        return obs.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Get current environment state."""
    if not env:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    return env.state.model_dump()


@app.get("/summary")
async def summary() -> Dict[str, Any]:
    """Get episode summary with analytics."""
    if not env:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    return env.get_episode_summary()


@app.get("/tasks")
async def tasks() -> Dict[str, Any]:
    """Get available tasks."""
    if not env:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    return env.get_available_tasks()


@app.get("/grade")
async def grade() -> Dict[str, Any]:
    """Grade the current episode."""
    if not env:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    state = env.state
    if not state or state.episode_id == "none":
        raise HTTPException(status_code=400, detail="No active episode to grade")
    
    result = grade_episode(state)
    return result


# Serve static frontend if available
frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    @app.get("/")
    async def root():
        return FileResponse(str(frontend_dir / "index.html"))
else:
    @app.get("/")
    async def root():
        return {
            "message": "PolicyGuard-RL API",
            "docs": "/docs",
            "endpoints": ["/reset", "/step", "/state", "/info", "/health"]
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
