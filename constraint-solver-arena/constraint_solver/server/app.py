from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core import create_app
from constraint_solver import ConstraintSolverEnv, ConstraintAction, ConstraintObservation


def create_env():
    """Factory function to create environment instances."""
    return ConstraintSolverEnv()


app = create_app(
    env=create_env,
    action_cls=ConstraintAction,
    observation_cls=ConstraintObservation,
    env_name="ConstraintSolver Arena"
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>ConstraintSolver Arena</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1>🧩 ConstraintSolver Arena</h1>
        <p>OpenEnv benchmark for AI constraint satisfaction and planning capabilities.</p>
        <p><em>Exposing what LLMs can't do: multi-constraint reasoning.</em></p>
        
        <h3>API Endpoints</h3>
        <ul>
            <li><a href="/health">/health</a> - Health check</li>
            <li><a href="/metadata">/metadata</a> - Environment metadata</li>
            <li><a href="/schema">/schema</a> - Action/Observation schemas</li>
            <li><a href="/openapi.json">/openapi.json</a> - OpenAPI spec</li>
            <li>POST /reset - Start new episode</li>
            <li>POST /step - Take action</li>
            <li>GET /state - Current state</li>
        </ul>
        
        <h3>Tasks</h3>
        <ul>
            <li><b>Meeting Scheduler</b> (Easy) - Find time slots satisfying multiple participants</li>
            <li><b>Resource Allocator</b> (Medium) - Assign tasks to workers with skill/hour constraints</li>
            <li><b>Travel Planner</b> (Hard) - Plan trips with budget, time, and dependency constraints</li>
        </ul>
        
        <h3>Why This Matters</h3>
        <p>Research shows LLMs struggle with constraint satisfaction (SagaLLM, TRIP-PAL papers, 2026).
        This benchmark provides deterministic evaluation of planning and scheduling capabilities.</p>
        
        <h3>Grading</h3>
        <p>100% deterministic: each constraint is checked and scored 0.0-1.0.</p>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "healthy"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
