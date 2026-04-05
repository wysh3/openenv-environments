#!/usr/bin/env python3
"""
Inference Script for PolicyGuard-RL
===================================
MANDATORY STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Uses OpenAI client with environment variables:
    API_BASE_URL, MODEL_NAME, HF_TOKEN
"""

import os
import sys
import json
import asyncio
import textwrap
from typing import List, Dict, Any, Optional

from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policyguard.environment import PolicyGuardEnv, TASKS
from policyguard.models import PolicyGuardAction, Decision, ContentCategory
from policyguard.grader import grade_episode


# === CONFIGURATION (from environment variables) ===
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "policyguard-rl"
MAX_STEPS_PER_TASK = 15
SUCCESS_THRESHOLD = 0.5
TEMPERATURE = 0.3
MAX_TOKENS = 500


# === STDOUT LOGGING (exact format required) ===
def log_start(task: str, env: str, model: str) -> None:
    """Print [START] marker."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Print [STEP] marker."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Print [END] marker."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# === SYSTEM PROMPT FOR LLM ===
SYSTEM_PROMPT = textwrap.dedent("""
You are an AI content moderator in a 3-tier escalation system.

Your job is to review social media content and make moderation decisions.

For each piece of content, you must output a JSON object with these fields:
{
    "decision": "allow" | "remove" | "reduce" | "label" | "escalate",
    "category": "misinformation" | "hate_speech" | "child_safety" | "harassment" | "violence" | "spam" | "none",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision",
    "bias_identified": "any cognitive bias you notice in yourself (optional)"
}

Decisions:
- allow: Content is acceptable
- remove: Content violates policies and should be deleted
- reduce: Reduce distribution (shadowban) but don't remove
- label: Add a warning/context label
- escalate: Send to senior reviewer (only at Tier 1 and 2)

Guidelines:
- At Tier 1: You're a fast first-line reviewer. Escalate edge cases.
- At Tier 2: You're a senior reviewer. Escalate only very difficult cases.
- At Tier 3: You CANNOT escalate. You must make a final decision.

Be vigilant but fair. Missing actual violations is worse than over-moderating.
""").strip()


def build_user_prompt(observation: Dict[str, Any]) -> str:
    """Build prompt from observation."""
    post = observation.get("post", {})
    tier = observation.get("current_tier", "tier_1")
    task = observation.get("task_name", "content_moderation")
    previous = observation.get("previous_decisions", [])
    feedback = observation.get("feedback", "")
    
    prompt = f"""
CONTENT TO REVIEW:
---
{post.get('text', 'No content')}
---

Context: {post.get('context', 'None provided')}
Author history: {json.dumps(post.get('author_history', {}))}
Engagement: {json.dumps(post.get('engagement', {}))}

Current Task: {task}
Your Tier: {tier.upper()} {'(FINAL DECISION REQUIRED - CANNOT ESCALATE)' if tier == 'tier_3' else ''}

{'Previous Tier Decisions: ' + json.dumps(previous) if previous else ''}
{'Last Feedback: ' + feedback if feedback else ''}

Output your decision as a JSON object:
"""
    return prompt.strip()


def parse_llm_response(response_text: str, tier: str) -> PolicyGuardAction:
    """Parse LLM response into action."""
    # Try to extract JSON from response
    try:
        # Find JSON in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            data = json.loads(json_str)
            
            decision = data.get("decision", "allow")
            category = data.get("category", "none")
            
            # Tier 3 cannot escalate
            if tier == "tier_3" and decision == "escalate":
                decision = "reduce"  # Default to reduce for uncertain cases
            
            return PolicyGuardAction(
                decision=Decision(decision),
                category=ContentCategory(category),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                policy_cited=data.get("policy_cited", ""),
                bias_identified=data.get("bias_identified", ""),
            )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        pass
    
    # Fallback: try to detect decision from text
    text_lower = response_text.lower()
    if "remove" in text_lower:
        decision = Decision.REMOVE
    elif "escalate" in text_lower and tier != "tier_3":
        decision = Decision.ESCALATE
    elif "label" in text_lower:
        decision = Decision.LABEL
    elif "reduce" in text_lower:
        decision = Decision.REDUCE
    else:
        decision = Decision.ALLOW
    
    return PolicyGuardAction(
        decision=decision,
        category=ContentCategory.NONE,
        confidence=0.5,
        reasoning=response_text[:200],
    )


def get_llm_decision(client: OpenAI, observation: Dict[str, Any]) -> PolicyGuardAction:
    """Get decision from LLM."""
    user_prompt = build_user_prompt(observation)
    tier = observation.get("current_tier", "tier_1")
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(response_text, tier)
    except Exception as e:
        # Fallback on API error
        print(f"[DEBUG] LLM request failed: {e}", flush=True)
        return PolicyGuardAction(
            decision=Decision.ALLOW,
            category=ContentCategory.NONE,
            confidence=0.5,
            reasoning=f"API error fallback: {str(e)[:50]}",
        )


async def run_task(
    env: PolicyGuardEnv,
    client: OpenAI,
    task_name: str,
) -> Dict[str, Any]:
    """Run inference on a single task."""
    difficulty = TASKS[task_name].get("difficulty", "medium")
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    error_msg: Optional[str] = None
    
    try:
        # Reset environment
        obs = await env.reset(task=task_name, difficulty=difficulty)
        obs_dict = obs.model_dump()
        
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if obs_dict.get("done", False):
                break
            
            # Get action from LLM
            action = get_llm_decision(client, obs_dict)
            action_str = f"{action.decision.value}({action.category.value})"
            
            # Take step
            obs = await env.step(action)
            obs_dict = obs.model_dump()
            
            reward = obs_dict.get("reward", 0.0)
            done = obs_dict.get("done", False)
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            
            if done:
                break
        
        # Calculate final score
        result = grade_episode(env.state)
        score = result.get("score", 0.0)
        success = score >= SUCCESS_THRESHOLD
        
    except Exception as e:
        error_msg = str(e)[:100]
        score = 0.0
        success = False
    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return {
        "task": task_name,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


async def main() -> float:
    """Run inference across all tasks."""
    # Initialize OpenAI client
    if not API_KEY:
        print("[ERROR] No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY", flush=True)
        return 0.0
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = PolicyGuardEnv()
    
    all_results = []
    
    try:
        for task_name in TASKS.keys():
            result = await run_task(env, client, task_name)
            all_results.append(result)
        
        # Calculate overall score
        overall_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
        
        # Print summary
        print("\n" + "=" * 60, flush=True)
        print("INFERENCE SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"Overall Score: {overall_score:.4f}", flush=True)
        print(f"Tasks Completed: {len(all_results)}/{len(TASKS)}", flush=True)
        for r in all_results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['task']}: {r['score']:.4f}", flush=True)
        
        return overall_score
        
    finally:
        await env.close()


if __name__ == "__main__":
    score = asyncio.run(main())
    sys.exit(0 if score >= 0.0 else 1)
