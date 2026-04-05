"""Pydantic models for PolicyGuard-RL multi-agent content moderation."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ContentCategory(str, Enum):
    """Content violation categories."""
    MISINFORMATION = "misinformation"
    HATE_SPEECH = "hate_speech"
    CHILD_SAFETY = "child_safety"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SPAM = "spam"
    NONE = "none"  # No violation


class Decision(str, Enum):
    """Moderation decisions."""
    ALLOW = "allow"
    REMOVE = "remove"
    REDUCE = "reduce"  # Reduce distribution
    LABEL = "label"  # Add warning label
    ESCALATE = "escalate"  # Send to next tier


class Tier(str, Enum):
    """Agent tiers in escalation chain."""
    TIER_1 = "tier_1"  # First-line reviewer (fast, cheap)
    TIER_2 = "tier_2"  # Senior reviewer (edge cases)
    TIER_3 = "tier_3"  # Appeals/final decision (hard cases)


class ContentPost(BaseModel):
    """A piece of content to be moderated."""
    id: str
    text: str
    author_history: Dict[str, Any] = Field(default_factory=dict)
    engagement: Dict[str, int] = Field(default_factory=lambda: {
        "likes": 0, "shares": 0, "comments": 0, "reports": 0
    })
    context: str = ""  # Additional context (thread, replies, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Ground truth (hidden from agent)
    ground_truth: Optional[Dict[str, Any]] = None


class AgentDecision(BaseModel):
    """Decision made by a single agent."""
    tier: Tier
    decision: Decision
    category: ContentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    policy_cited: str = ""  # Which policy applies
    bias_identified: str = ""  # Any bias the agent noticed in itself


class TierResult(BaseModel):
    """Result from a tier's processing."""
    tier: Tier
    decision: AgentDecision
    escalated: bool = False
    processing_time_ms: int = 0


class PolicyGuardAction(BaseModel):
    """Action submitted by the agent (for current tier)."""
    decision: Decision
    category: ContentCategory
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    reasoning: str = ""
    policy_cited: str = ""
    bias_identified: str = ""
    
    class Config:
        use_enum_values = True


class PolicyGuardObservation(BaseModel):
    """What the agent observes."""
    # Current content
    post: ContentPost
    
    # Current tier and task
    current_tier: Tier
    task_name: str
    task_difficulty: str
    
    # History from previous tiers (if escalated)
    previous_decisions: List[TierResult] = Field(default_factory=list)
    
    # Feedback (after step)
    feedback: str = ""
    reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    
    # Episode state
    step_count: int = 0
    done: bool = False
    
    class Config:
        use_enum_values = True


class PolicyGuardState(BaseModel):
    """Full episode state."""
    # Episode tracking
    episode_id: str
    task_name: str
    task_difficulty: str
    
    # Current position
    current_post_idx: int = 0
    current_tier: Tier = Tier.TIER_1
    posts_in_episode: int = 3
    
    # All posts for this episode
    posts: List[ContentPost] = Field(default_factory=list)
    
    # Decision history
    tier_results: List[TierResult] = Field(default_factory=list)
    
    # Scoring
    cumulative_reward: float = 0.0
    decisions_made: int = 0
    escalations: int = 0
    correct_decisions: int = 0
    
    # Episode state
    step_count: int = 0
    done: bool = False
    
    # Analytics
    bias_triggers: Dict[str, int] = Field(default_factory=dict)
    category_accuracy: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
