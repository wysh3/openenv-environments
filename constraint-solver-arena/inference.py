"""
OpenEnv Inference Script for ConstraintSolver Arena
=====================================================
STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
from typing import List

from openai import OpenAI

from constraint_solver import ConstraintSolverEnv, ConstraintAction

# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "constraint-solver-arena"
MAX_STEPS = 1  # Single-step environment


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_str = "null" if error is None else error
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}")


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")


def build_prompt(obs) -> str:
    """Build prompt from observation for LLM."""
    task_type = obs.task_type
    
    if task_type == "meeting_scheduler":
        return build_meeting_prompt(obs)
    elif task_type == "resource_allocator":
        return build_resource_prompt(obs)
    elif task_type == "travel_planner":
        return build_travel_prompt(obs)
    else:
        return build_generic_prompt(obs)


def build_meeting_prompt(obs) -> str:
    """Build prompt for meeting scheduling task."""
    return f"""You are a scheduling assistant. Find a meeting time that satisfies all constraints.

TASK: {obs.task_description}

PROBLEM DATA:
{json.dumps(obs.problem_data, indent=2)}

HARD CONSTRAINTS (must satisfy):
{json.dumps(obs.hard_constraints, indent=2)}

SOFT CONSTRAINTS (nice to have):
{json.dumps(obs.soft_constraints, indent=2)}

Respond with a JSON object containing:
- "task_type": "meeting_scheduler"
- "meeting_day": day of the week (e.g., "Monday", "Tuesday")
- "meeting_start": start time in HH:MM format (e.g., "10:00")
- "meeting_end": end time in HH:MM format (e.g., "11:00")
- "meeting_room": room name if required (or null)
- "reasoning": brief explanation of why this slot works

If no valid slot exists, set meeting_day to "IMPOSSIBLE" and explain why in reasoning.

JSON response:"""


def build_resource_prompt(obs) -> str:
    """Build prompt for resource allocation task."""
    return f"""You are a resource allocation expert. Assign tasks to workers while satisfying all constraints.

TASK: {obs.task_description}

PROBLEM DATA:
{json.dumps(obs.problem_data, indent=2)}

HARD CONSTRAINTS (must satisfy):
{json.dumps(obs.hard_constraints, indent=2)}

Respond with a JSON object containing:
- "task_type": "resource_allocator"
- "assignments": list of {{"task_id": "T1", "worker_id": "W1"}} objects
- "reasoning": brief explanation of your assignment logic

Ensure:
1. Each task is assigned to a worker with required skills
2. No worker exceeds their available hours
3. All tasks are assigned

JSON response:"""


def build_travel_prompt(obs) -> str:
    """Build prompt for travel planning task."""
    return f"""You are a travel planning expert. Create an itinerary that satisfies all constraints.

TASK: {obs.task_description}

PROBLEM DATA:
{json.dumps(obs.problem_data, indent=2)}

HARD CONSTRAINTS (must satisfy):
{json.dumps(obs.hard_constraints, indent=2)}

SOFT CONSTRAINTS (nice to have):
{json.dumps(obs.soft_constraints, indent=2)}

Respond with a JSON object containing:
- "task_type": "travel_planner"
- "itinerary": list of {{"activity": "name", "start_time": "HH:MM", "end_time": "HH:MM", "cost": number}} objects
- "total_cost": sum of all activity costs
- "reasoning": brief explanation of your planning logic

Ensure:
1. Total cost within budget
2. All dependencies respected (A before B)
3. No overlapping activities
4. All times within day constraints

JSON response:"""


def build_generic_prompt(obs) -> str:
    """Generic prompt for unknown task types."""
    return f"""You are a constraint satisfaction expert. Solve the following problem.

TASK: {obs.task_description}
TYPE: {obs.task_type}

PROBLEM DATA:
{json.dumps(obs.problem_data, indent=2)}

CONSTRAINTS:
{json.dumps(obs.hard_constraints, indent=2)}

Respond with a JSON solution that satisfies all constraints.

JSON response:"""


def parse_llm_response(response_text: str, task_type: str) -> ConstraintAction:
    """Parse LLM response into action."""
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text)
        
        return ConstraintAction(
            task_type=task_type,
            meeting_day=data.get("meeting_day"),
            meeting_start=data.get("meeting_start"),
            meeting_end=data.get("meeting_end"),
            meeting_room=data.get("meeting_room"),
            assignments=data.get("assignments"),
            itinerary=data.get("itinerary"),
            total_cost=data.get("total_cost"),
            reasoning=data.get("reasoning", "")
        )
    except Exception as e:
        # Default empty action if parsing fails
        return ConstraintAction(
            task_type=task_type,
            reasoning=f"Parse error: {str(e)}"
        )


def run_episode(client: OpenAI, env: ConstraintSolverEnv, task_name: str, scenario_id: int) -> float:
    """Run a single episode and return the reward."""
    log_start(task_name, MODEL_NAME)
    
    rewards: List[float] = []
    error_msg = None
    
    try:
        obs = env.reset(scenario_id=scenario_id)
        task_type = obs.task_type
        
        # Get LLM response
        prompt = build_prompt(obs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        action = parse_llm_response(response_text, task_type)
        
        # Format action for logging (abbreviated)
        if task_type == "meeting_scheduler":
            action_str = json.dumps({
                "day": action.meeting_day,
                "start": action.meeting_start,
                "end": action.meeting_end
            }, separators=(',', ':'))
        elif task_type == "resource_allocator":
            action_str = json.dumps({
                "assignments": len(action.assignments or [])
            }, separators=(',', ':'))
        elif task_type == "travel_planner":
            action_str = json.dumps({
                "activities": len(action.itinerary or []),
                "cost": action.total_cost
            }, separators=(',', ':'))
        else:
            action_str = "{}"
        
        # Step environment
        obs = env.step(action)
        reward = obs.reward
        done = obs.done
        rewards.append(reward)
        
        log_step(1, action_str, reward, done, error_msg)
        
    except Exception as e:
        error_msg = str(e)
        reward = 0.0
        rewards.append(reward)
        log_step(1, "{}", reward, True, error_msg)
    
    success = len(rewards) > 0 and rewards[-1] >= 0.5
    log_end(success, len(rewards), rewards)
    
    return rewards[-1] if rewards else 0.0


def main():
    """Run inference on all task types."""
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable required")
        return
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Run 3 episodes - one for each task type
    task_scenarios = [
        ("meeting_scheduler", 0),
        ("resource_allocator", 0),
        ("travel_planner", 0),
    ]
    
    total_reward = 0.0
    for task_name, scenario_id in task_scenarios:
        env = ConstraintSolverEnv(task_type=task_name)
        reward = run_episode(client, env, task_name, scenario_id)
        total_reward += reward
    
    print(f"\nTotal score: {total_reward:.2f} / 3.00")


if __name__ == "__main__":
    main()
