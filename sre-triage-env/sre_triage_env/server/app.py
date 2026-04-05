from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core import create_app
from sre_triage_env import SRETriageEnv, SREAction, SREObservation


def create_env():
    """Factory function to create environment instances."""
    return SRETriageEnv(difficulty="medium")


app = create_app(
    env=create_env,
    action_cls=SREAction,
    observation_cls=SREObservation,
    env_name="SRE Triage Environment"
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>SRE Triage Environment</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1>🚨 SRE Triage Environment</h1>
        <p>OpenEnv environment for AI agent incident triage testing.</p>
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
            <li><b>Severity Classification</b> - P1-P5 priority</li>
            <li><b>Root Cause Analysis</b> - Find failing service</li>
            <li><b>Silent Failure Detection</b> - Detect hidden incidents</li>
        </ul>
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
