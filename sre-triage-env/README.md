---
title: SRE Incident Response Triage Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
  - sre
  - incident-response
---

# SRE Incident Response Triage Environment

A real-world OpenEnv environment for testing AI agents on SRE incident triage scenarios.

**Why this matters:** Incident response failures cost enterprises **$5.6M per incident** (Gartner 2025). AI that can triage correctly reduces MTTR by 60%, preventing cascading failures. Banks, e-commerce, and SaaS companies need agents that distinguish real P1s from false alarms.

---

SRE teams must quickly triage incidents: assess severity, identify root causes, and distinguish real failures from noise. This environment simulates that decision-making under pressure with realistic metrics, alerts, logs, and service dependencies.

## Tasks (3 Task Types × 3 Difficulties = 60 Scenarios)

| Task | Description | Difficulty |
|------|-------------|------------|
| **Severity Classification** | Classify P1-P5 based on impact | Easy → Hard |
| **Root Cause Analysis** | Find failing service in cascades | Easy → Hard |
| **Silent Failure Detection** | Detect issues without obvious alerts | Easy → Hard |

## Observation Space (SREObservation)

| Field | Type | Description |
|-------|------|-------------|
| `task_description` | str | What the agent needs to determine |
| `metrics` | Dict | Service metrics: CPU, memory, error_rate, latency |
| `alerts` | List[Dict] | Active monitoring alerts |
| `logs` | List[str] | Recent log entries |
| `service_dependencies` | Dict | Service dependency graph |
| `previous_incidents` | List[Dict] | Historical context |
| `done` | bool | Episode finished |
| `reward` | float | Score for this step |

## Action Space (SREAction)

| Field | Type | Description |
|-------|------|-------------|
| `severity` | str | P1, P2, P3, P4, P5, or NONE |
| `affected_service` | str | Root cause service name |
| `is_incident` | bool | True if real incident |
| `reasoning` | str | Explanation (optional) |

## Scoring (Graders)

- **Severity**: 1.0 exact, 0.5 off-by-one, 0.25 off-by-two, 0.0 otherwise
- **Root Cause**: 1.0 service+severity, 0.6 service only, 0.2 severity only
- **Silent Failure**: 0.6 correct detection + 0.4 correct service

## Usage

**Complete example** (P1 severity classification):

```bash
# 1. Start episode
curl -X POST https://wysh3-sre-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "severity_classification", "difficulty": "medium"}'

# Returns:
{
  "task_description": "Classify incident severity",
  "metrics": {
    "cpu": 95,
    "memory": 88,
    "error_rate": 0.45,
    "latency_p99": 5200
  },
  "alerts": [
    {"type": "cpu_spike", "service": "api-server", "triggered": 120}
  ]
}

# 2. Submit classification
curl -X POST https://wysh3-sre-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "severity": "P1",
    "affected_service": "api-server",
    "is_incident": true,
    "reasoning": "High error rate + latency spike + resource exhaustion = user-facing outage"
  }'

# Returns:
{
  "reward": 1.0,
  "feedback": "Correct! Identified P1 incident with affected service."
}
```

---

Requires environment variables:
- `HF_TOKEN` or `API_KEY`: Your API key
- `API_BASE_URL`: LLM endpoint (default: https://router.huggingface.co/v1)
- `MODEL_NAME`: Model to use (default: Qwen/Qwen2.5-72B-Instruct)

```bash
export HF_TOKEN=your_token
python inference.py
```

## Baseline Scores

Using Qwen2.5-72B-Instruct on easy scenarios:

| Task | Score |
|------|-------|
| Severity Classification | 0.50 |
| Root Cause Analysis | 0.60 |
| Silent Failure Detection | 1.00 |
| **Average** | **0.70** |

## Docker Build

```bash
docker build -t sre-triage-env .
docker run -p 7860:7860 sre-triage-env
```

## API Usage

```python
from sre_triage_env import SRETriageEnv, SREAction

env = SRETriageEnv(difficulty="medium")
obs = env.reset()

action = SREAction(
    severity="P1",
    affected_service="database",
    is_incident=True
)

obs = env.step(action)
print(f"Reward: {obs.reward}, Done: {obs.done}")
```

## Files

- `inference.py` - Inference script with required stdout format
- `openenv.yaml` - OpenEnv metadata
- `Dockerfile` - Container build
- `sre_triage_env/` - Environment package
- `scenarios/` - JSON scenario files (easy.json, medium.json, hard.json)
- `tests/` - Test suite
