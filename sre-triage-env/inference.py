"""
OpenEnv Inference Script for SRE Triage Environment
====================================================
STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
from typing import List

from openai import OpenAI

from sre_triage_env import SRETriageEnv, SREAction

# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "sre-triage-env"
MAX_STEPS = 1  # Single-step environment
TASKS = ["severity", "root_cause", "silent_failure"]


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
    return f"""You are an SRE expert analyzing an incident. Based on the data below, provide your triage decision.

TASK: {obs.task_description}

METRICS:
{json.dumps(obs.metrics, indent=2)}

ALERTS:
{json.dumps(obs.alerts, indent=2)}

LOGS:
{json.dumps(obs.logs, indent=2)}

SERVICE DEPENDENCIES:
{json.dumps(obs.service_dependencies, indent=2)}

Respond with a JSON object containing:
- "severity": one of "P1", "P2", "P3", "P4", "P5", or "NONE"
- "affected_service": the service name causing the issue (empty string if none)
- "is_incident": true or false
- "reasoning": brief explanation

JSON response:"""


def parse_llm_response(response_text: str) -> SREAction:
    """Parse LLM response into action."""
    try:
        # Try to extract JSON from response
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text)
        return SREAction(
            severity=data.get("severity", "P3"),
            affected_service=data.get("affected_service", ""),
            is_incident=data.get("is_incident", True),
            reasoning=data.get("reasoning", "")
        )
    except Exception:
        # Default action if parsing fails
        return SREAction(severity="P3", is_incident=True)


def run_episode(client: OpenAI, env: SRETriageEnv, task_name: str, scenario_id: int) -> float:
    """Run a single episode and return the reward."""
    log_start(task_name, MODEL_NAME)
    
    rewards: List[float] = []
    error_msg = None
    
    try:
        obs = env.reset(scenario_id=scenario_id)
        
        # Get LLM response
        prompt = build_prompt(obs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        action = parse_llm_response(response_text)
        
        # Format action for logging
        action_str = json.dumps({
            "severity": action.severity,
            "affected_service": action.affected_service,
            "is_incident": action.is_incident
        }, separators=(',', ':'))
        
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
    # Scenario 0: severity, Scenario 1: root_cause, Scenario 3: silent_failure
    task_scenarios = [
        ("severity", 0),
        ("root_cause", 1),
        ("silent_failure", 3),
    ]
    
    total_reward = 0.0
    for task_name, scenario_id in task_scenarios:
        env = SRETriageEnv(difficulty="easy")
        reward = run_episode(client, env, task_name, scenario_id)
        total_reward += reward
    
    print(f"\nTotal score: {total_reward:.2f} / 3.00")


if __name__ == "__main__":
    main()
