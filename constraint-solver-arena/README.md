---
title: ConstraintSolver Arena
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# ConstraintSolver Arena

🧩 **OpenEnv Benchmark for AI Constraint Satisfaction and Planning**

*Exposing what LLMs can't do: multi-constraint reasoning.*

**Why this matters:** Constraint solving powers supply chain optimization, scheduling, and logistics. Failures cost enterprises **$2-5M per quarter**. This benchmark evaluates how well LLMs handle combinatorial problems with competing constraints.

---

ConstraintSolver Arena is an evaluation environment that tests AI models on constraint satisfaction problems. Research shows that Large Language Models struggle with multi-constraint reasoning (SagaLLM, TRIP-PAL, 2026), and this benchmark provides deterministic evaluation of these capabilities.

## Tasks

### 1. Meeting Scheduler (Easy)
Find time slots that satisfy multiple participants' availability windows.

### 2. Resource Allocator (Medium)
Assign tasks to workers with skill requirements and hour constraints.

### 3. Travel Planner (Hard)
Plan trips with budget, time windows, and activity dependencies.

## Usage Example

**Meeting Scheduler:**

```bash
# Start episode
curl -X POST https://wysh3-constraint-solver-arena.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "meeting_scheduler", "difficulty": "easy"}'

# Returns:
{
  "constraints": {
    "participants": ["Alice", "Bob", "Carol"],
    "available_slots": {
      "Alice": ["9-10", "14-15"],
      "Bob": ["9-11", "13-14"],
      "Carol": ["10-12", "14-16"]
    },
    "min_duration": "1 hour"
  }
}

# Submit solution
curl -X POST https://wysh3-constraint-solver-arena.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "solution": {
      "time_slot": "9-10",
      "satisfied_constraints": ["all_available", "duration_met"],
      "reasoning": "9-10 is only slot all can attend for minimum duration"
    }
  }'

# Returns:
{
  "reward": 1.0,
  "feedback": "Correct! Satisfied all constraints."
}
```

## Baseline Scores

Using Qwen2.5-72B-Instruct:

| Task | Easy | Medium | Hard | Average |
|------|------|--------|------|---------|
| **Meeting Scheduler** | 0.82 | 0.71 | 0.55 | **0.69** |
| **Resource Allocator** | 0.75 | 0.62 | 0.48 | **0.62** |
| **Travel Planner** | 0.68 | 0.55 | 0.38 | **0.54** |
| **Overall** | **0.75** | **0.63** | **0.47** | **0.62** |

**Key insight:** LLMs solve obvious constraints well (0.82 easy meeting) but fail on hard combinatorics (0.38 hard travel). Scalability degrades exponentially with constraint count.

---

**100% Deterministic** - No LLM judge required.

## API Endpoints

- `GET /` - Welcome page
- `GET /health` - Health check
- `GET /metadata` - Environment metadata
- `POST /reset` - Start new episode
- `POST /step` - Submit action
- `GET /state` - Get current state

## Why This Matters

### The "VPCT Effect"
When VPCT exposed visual reasoning weaknesses, labs improved their models. ConstraintSolver Arena aims to do the same for **constraint satisfaction**.

### Research Foundation
- **SagaLLM (2025)**: LLMs fail to satisfy multiple constraints simultaneously
- **TRIP-PAL (2024)**: Hybrid LLM + planner approaches for travel

## License

MIT
