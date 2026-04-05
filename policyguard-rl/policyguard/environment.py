"""Core environment for PolicyGuard-RL multi-agent content moderation."""

import json
import random
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

from .models import (
    ContentPost,
    AgentDecision,
    PolicyGuardAction,
    PolicyGuardObservation,
    PolicyGuardState,
    TierResult,
    ContentCategory,
    Decision,
    Tier,
)
from .grader import grade_decision


# Scenario file paths
SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

# Task configurations
TASKS = {
    "misinformation": {
        "difficulty": "easy",
        "file": "misinformation.json",
        "description": "Detect fake news, satire, and misleading claims",
    },
    "hate_speech": {
        "difficulty": "medium",
        "file": "hate_speech.json",
        "description": "Context-dependent hate speech detection",
    },
    "child_safety": {
        "difficulty": "hard",
        "file": "child_safety.json",
        "description": "Grooming patterns, age-inappropriate content",
    },
    "harassment": {
        "difficulty": "medium",
        "file": "harassment.json",
        "description": "Bullying, pile-ons, targeted attacks",
    },
    "violence": {
        "difficulty": "hard",
        "file": "violence.json",
        "description": "Threats, graphic content, news vs fiction",
    },
}

# Difficulty settings
DIFFICULTY_CONFIG = {
    "easy": {"posts_per_episode": 2, "escalation_threshold": 0.7},
    "medium": {"posts_per_episode": 3, "escalation_threshold": 0.8},
    "hard": {"posts_per_episode": 4, "escalation_threshold": 0.9},
}


class PolicyGuardEnv:
    """
    Multi-agent content moderation environment.
    
    Three-tier escalation chain:
    - Tier 1: Fast first-line review (70% of content resolved here)
    - Tier 2: Senior review for edge cases (escalated from Tier 1)
    - Tier 3: Appeals/final decisions (escalated from Tier 2)
    
    The agent plays ALL tiers, learning when to escalate vs decide.
    """
    
    def __init__(self):
        self._state: Optional[PolicyGuardState] = None
        self._scenarios: Dict[str, List[Dict]] = {}
        self._load_scenarios()
    
    def _load_scenarios(self) -> None:
        """Load all scenario files."""
        for task_name, config in TASKS.items():
            filepath = SCENARIOS_DIR / config["file"]
            if filepath.exists():
                with open(filepath, "r") as f:
                    self._scenarios[task_name] = json.load(f)
            else:
                self._scenarios[task_name] = []
    
    def _select_posts(self, task: str, count: int) -> List[ContentPost]:
        """Select posts for an episode."""
        scenarios = self._scenarios.get(task, [])
        if not scenarios:
            # Fallback to synthetic if no scenarios loaded
            return self._generate_synthetic_posts(task, count)
        
        selected = random.sample(scenarios, min(count, len(scenarios)))
        posts = []
        for s in selected:
            posts.append(ContentPost(
                id=s.get("id", str(uuid.uuid4())[:8]),
                text=s.get("text", ""),
                author_history=s.get("author_history", {}),
                engagement=s.get("engagement", {}),
                context=s.get("context", ""),
                metadata=s.get("metadata", {}),
                ground_truth=s.get("ground_truth", {}),
            ))
        return posts
    
    def _generate_synthetic_posts(self, task: str, count: int) -> List[ContentPost]:
        """Generate synthetic posts if no scenarios available."""
        posts = []
        for i in range(count):
            posts.append(ContentPost(
                id=f"synthetic_{task}_{i}",
                text=f"[Synthetic {task} content #{i}]",
                ground_truth={
                    "correct_decision": "remove" if random.random() > 0.5 else "allow",
                    "category": task,
                    "explanation": "Synthetic scenario",
                }
            ))
        return posts
    
    async def reset(
        self,
        task: Optional[str] = None,
        difficulty: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> PolicyGuardObservation:
        """
        Reset environment for a new episode.
        
        Args:
            task: Specific task (misinformation, hate_speech, etc.)
            difficulty: easy, medium, hard
            options: Additional options (can contain task/difficulty)
        """
        # Handle options dict
        if options:
            task = options.get("task", task)
            difficulty = options.get("difficulty", difficulty)
        
        # Default to random task if not specified
        if not task:
            task = random.choice(list(TASKS.keys()))
        
        # Get difficulty from task config or use provided
        if not difficulty:
            difficulty = TASKS.get(task, {}).get("difficulty", "medium")
        
        config = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
        posts_count = config["posts_per_episode"]
        
        # Select posts for episode
        posts = self._select_posts(task, posts_count)
        
        # Initialize state
        self._state = PolicyGuardState(
            episode_id=str(uuid.uuid4())[:8],
            task_name=task,
            task_difficulty=difficulty,
            posts=posts,
            posts_in_episode=len(posts),
            current_post_idx=0,
            current_tier=Tier.TIER_1,
        )
        
        return self._get_observation()
    
    async def step(self, action: PolicyGuardAction) -> PolicyGuardObservation:
        """
        Process an action from the agent.
        
        The agent decides for the current tier. If they escalate,
        they'll see the same post again at the next tier.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        
        if self._state.done:
            return self._get_observation()
        
        self._state.step_count += 1
        current_post = self._state.posts[self._state.current_post_idx]
        current_tier = self._state.current_tier
        
        # Create agent decision record
        agent_decision = AgentDecision(
            tier=current_tier,
            decision=Decision(action.decision),
            category=ContentCategory(action.category),
            confidence=action.confidence,
            reasoning=action.reasoning,
            policy_cited=action.policy_cited,
            bias_identified=action.bias_identified,
        )
        
        # Grade the decision
        reward, feedback, breakdown = grade_decision(
            post=current_post,
            decision=agent_decision,
            tier=current_tier,
            previous_decisions=self._state.tier_results,
        )
        
        # Track bias triggers
        if action.bias_identified:
            bias = action.bias_identified.lower()
            self._state.bias_triggers[bias] = self._state.bias_triggers.get(bias, 0) + 1
        
        # Record tier result
        tier_result = TierResult(
            tier=current_tier,
            decision=agent_decision,
            escalated=(action.decision == "escalate"),
        )
        self._state.tier_results.append(tier_result)
        
        # Handle escalation vs final decision
        if action.decision == "escalate":
            self._state.escalations += 1
            
            # Move to next tier
            if current_tier == Tier.TIER_1:
                self._state.current_tier = Tier.TIER_2
            elif current_tier == Tier.TIER_2:
                self._state.current_tier = Tier.TIER_3
            else:
                # Tier 3 can't escalate - treat as allow
                self._state.decisions_made += 1
                self._move_to_next_post()
        else:
            # Final decision made for this post
            self._state.decisions_made += 1
            
            # Check correctness
            gt = current_post.ground_truth or {}
            correct_decision = gt.get("correct_decision", "allow")
            if action.decision == correct_decision:
                self._state.correct_decisions += 1
            
            # Track category accuracy
            cat = action.category
            if cat not in self._state.category_accuracy:
                self._state.category_accuracy[cat] = {"correct": 0, "total": 0}
            self._state.category_accuracy[cat]["total"] += 1
            gt_cat = gt.get("category", "none")
            if cat == gt_cat:
                self._state.category_accuracy[cat]["correct"] += 1
            
            self._move_to_next_post()
        
        # Update cumulative reward
        self._state.cumulative_reward += reward
        
        return self._get_observation(feedback=feedback, reward=reward, breakdown=breakdown)
    
    def _move_to_next_post(self) -> None:
        """Move to next post in episode."""
        self._state.current_post_idx += 1
        self._state.current_tier = Tier.TIER_1  # Reset to Tier 1 for new post
        
        if self._state.current_post_idx >= self._state.posts_in_episode:
            self._state.done = True
    
    def _get_observation(
        self,
        feedback: str = "",
        reward: float = 0.0,
        breakdown: Optional[Dict[str, float]] = None,
    ) -> PolicyGuardObservation:
        """Build current observation."""
        if self._state.done:
            # Return final observation
            return PolicyGuardObservation(
                post=self._state.posts[-1] if self._state.posts else ContentPost(id="done", text=""),
                current_tier=self._state.current_tier,
                task_name=self._state.task_name,
                task_difficulty=self._state.task_difficulty,
                previous_decisions=self._state.tier_results,
                feedback=feedback or "Episode complete.",
                reward=reward,
                reward_breakdown=breakdown or {},
                step_count=self._state.step_count,
                done=True,
            )
        
        current_post = self._state.posts[self._state.current_post_idx]
        
        # Filter previous decisions to only show for current post
        current_post_decisions = [
            tr for tr in self._state.tier_results
            if tr.tier.value < self._state.current_tier.value
        ][-3:]  # Last 3 at most
        
        return PolicyGuardObservation(
            post=current_post,
            current_tier=self._state.current_tier,
            task_name=self._state.task_name,
            task_difficulty=self._state.task_difficulty,
            previous_decisions=current_post_decisions,
            feedback=feedback,
            reward=reward,
            reward_breakdown=breakdown or {},
            step_count=self._state.step_count,
            done=False,
        )
    
    @property
    def state(self) -> PolicyGuardState:
        """Return current state."""
        if self._state is None:
            return PolicyGuardState(
                episode_id="none",
                task_name="none",
                task_difficulty="none",
            )
        return self._state
    
    async def close(self) -> None:
        """Clean up resources."""
        self._state = None
    
    def get_available_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Return available tasks with metadata."""
        return {
            name: {
                **config,
                "scenario_count": len(self._scenarios.get(name, [])),
            }
            for name, config in TASKS.items()
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode."""
        if not self._state:
            return {}
        
        return {
            "episode_id": self._state.episode_id,
            "task": self._state.task_name,
            "difficulty": self._state.task_difficulty,
            "posts_processed": self._state.current_post_idx,
            "total_posts": self._state.posts_in_episode,
            "decisions_made": self._state.decisions_made,
            "escalations": self._state.escalations,
            "correct_decisions": self._state.correct_decisions,
            "accuracy": (
                self._state.correct_decisions / self._state.decisions_made
                if self._state.decisions_made > 0 else 0.0
            ),
            "cumulative_reward": self._state.cumulative_reward,
            "bias_triggers": self._state.bias_triggers,
        }


# Global state for HTTP server
_HTTP_STATE: Dict[str, Any] = {}


def sync_state_to_http(env: PolicyGuardEnv) -> None:
    """Sync environment state to HTTP-accessible global."""
    _HTTP_STATE["state"] = env.state.model_dump() if env.state else {}
    _HTTP_STATE["summary"] = env.get_episode_summary()


def get_http_state() -> Dict[str, Any]:
    """Get current state for HTTP endpoint."""
    return _HTTP_STATE.get("state", {})
