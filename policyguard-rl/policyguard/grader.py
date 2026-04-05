"""Grading logic for PolicyGuard-RL content moderation decisions."""

from typing import Dict, List, Tuple, Optional, Any
import re

from .models import (
    ContentPost,
    AgentDecision,
    TierResult,
    ContentCategory,
    Decision,
    Tier,
)


# Reward weights
WEIGHTS = {
    "correct_decision": 0.40,
    "correct_category": 0.20,
    "reasoning_quality": 0.15,
    "escalation_efficiency": 0.15,
    "bias_awareness": 0.10,
}

# Penalties
PENALTIES = {
    "false_negative": -0.50,  # Missed violation
    "false_positive": -0.30,  # Over-moderation
    "unnecessary_escalation": -0.15,
    "wrong_category": -0.10,
}

# Keywords for reasoning quality assessment
POLICY_KEYWORDS = {
    "misinformation": ["fact", "source", "verify", "claim", "evidence", "misleading", "false"],
    "hate_speech": ["protected", "group", "context", "slur", "intent", "target", "discrimination"],
    "child_safety": ["minor", "age", "grooming", "inappropriate", "protect", "safeguard", "CSAM"],
    "harassment": ["target", "pattern", "pile-on", "bully", "attack", "coordinated", "victim"],
    "violence": ["threat", "graphic", "incite", "news", "fiction", "real", "harm"],
}

# Bias types to recognize
BIAS_KEYWORDS = [
    "engagement", "viral", "popular", "follower", "verified",  # Engagement bias
    "authority", "expert", "celebrity", "influencer",  # Authority bias
    "recent", "trending", "breaking",  # Recency bias
    "similar", "pattern", "previous",  # Confirmation bias
    "report", "flagged", "complained",  # Report bias
]


def grade_decision(
    post: ContentPost,
    decision: AgentDecision,
    tier: Tier,
    previous_decisions: List[TierResult],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Grade a single moderation decision.
    
    Returns:
        Tuple of (reward, feedback, breakdown)
    """
    gt = post.ground_truth or {}
    correct_decision = gt.get("correct_decision", "allow")
    correct_category = gt.get("category", "none")
    explanation = gt.get("explanation", "")
    trap_type = gt.get("trap_type", "")
    
    breakdown = {}
    feedback_parts = []
    
    # 1. Decision correctness (40%)
    if decision.decision.value == "escalate":
        # Escalation is valid at Tier 1 and 2, not Tier 3
        if tier == Tier.TIER_3:
            breakdown["correct_decision"] = 0.0
            feedback_parts.append("Tier 3 cannot escalate - must make final decision.")
        else:
            # Partial credit for escalation (depends on difficulty)
            if _should_escalate(post, tier, correct_decision):
                breakdown["correct_decision"] = WEIGHTS["correct_decision"] * 0.7
                feedback_parts.append("Appropriate escalation to senior review.")
            else:
                breakdown["correct_decision"] = WEIGHTS["correct_decision"] * 0.3
                breakdown["unnecessary_escalation"] = PENALTIES["unnecessary_escalation"]
                feedback_parts.append("Unnecessary escalation - could have decided directly.")
    elif decision.decision.value == correct_decision:
        breakdown["correct_decision"] = WEIGHTS["correct_decision"]
        feedback_parts.append("Correct decision!")
    else:
        breakdown["correct_decision"] = 0.0
        if correct_decision == "remove" and decision.decision.value == "allow":
            breakdown["false_negative"] = PENALTIES["false_negative"]
            feedback_parts.append(f"Missed violation: content should be removed. {explanation}")
        elif correct_decision == "allow" and decision.decision.value == "remove":
            breakdown["false_positive"] = PENALTIES["false_positive"]
            feedback_parts.append(f"Over-moderation: content should be allowed. {explanation}")
        else:
            feedback_parts.append(f"Wrong decision. Expected: {correct_decision}")
    
    # 2. Category identification (20%)
    if decision.decision.value != "escalate":
        if decision.category.value == correct_category:
            breakdown["correct_category"] = WEIGHTS["correct_category"]
            feedback_parts.append("Correct category identified.")
        elif decision.category.value == "none" and correct_category == "none":
            breakdown["correct_category"] = WEIGHTS["correct_category"]
        else:
            breakdown["correct_category"] = 0.0
            breakdown["wrong_category"] = PENALTIES["wrong_category"]
            feedback_parts.append(f"Wrong category. Expected: {correct_category}")
    else:
        breakdown["correct_category"] = WEIGHTS["correct_category"] * 0.5
    
    # 3. Reasoning quality (15%)
    reasoning_score = _score_reasoning(
        decision.reasoning,
        decision.policy_cited,
        correct_category,
    )
    breakdown["reasoning_quality"] = WEIGHTS["reasoning_quality"] * reasoning_score
    if reasoning_score < 0.5:
        feedback_parts.append("Reasoning could be more thorough.")
    
    # 4. Escalation efficiency (15%)
    escalation_score = _score_escalation_efficiency(
        decision, tier, previous_decisions, correct_decision
    )
    breakdown["escalation_efficiency"] = WEIGHTS["escalation_efficiency"] * escalation_score
    
    # 5. Bias awareness (10%)
    bias_score = _score_bias_awareness(decision.bias_identified, trap_type)
    breakdown["bias_awareness"] = WEIGHTS["bias_awareness"] * bias_score
    if bias_score > 0.7 and decision.bias_identified:
        feedback_parts.append(f"Good bias awareness: identified '{decision.bias_identified}'")
    
    # Calculate total reward
    total_reward = sum(breakdown.values())
    total_reward = max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]
    
    feedback = " ".join(feedback_parts) if feedback_parts else "Decision processed."
    
    return total_reward, feedback, breakdown


def _should_escalate(post: ContentPost, tier: Tier, correct_decision: str) -> bool:
    """Determine if escalation is appropriate for this case."""
    gt = post.ground_truth or {}
    difficulty = gt.get("difficulty", "medium")
    
    # Hard cases should be escalated from Tier 1
    if tier == Tier.TIER_1 and difficulty == "hard":
        return True
    
    # Medium cases might be escalated from Tier 1
    if tier == Tier.TIER_1 and difficulty == "medium":
        return True
    
    # Easy cases shouldn't be escalated
    if difficulty == "easy":
        return False
    
    return False


def _score_reasoning(reasoning: str, policy_cited: str, category: str) -> float:
    """Score the quality of reasoning provided."""
    if not reasoning and not policy_cited:
        return 0.1
    
    score = 0.3  # Base score for any reasoning
    
    combined = f"{reasoning} {policy_cited}".lower()
    
    # Check for relevant keywords
    keywords = POLICY_KEYWORDS.get(category, [])
    keyword_matches = sum(1 for kw in keywords if kw in combined)
    score += min(0.4, keyword_matches * 0.1)
    
    # Length bonus (but not too long)
    words = len(combined.split())
    if 10 <= words <= 100:
        score += 0.2
    elif words > 5:
        score += 0.1
    
    # Policy citation bonus
    if policy_cited and len(policy_cited) > 5:
        score += 0.1
    
    return min(1.0, score)


def _score_escalation_efficiency(
    decision: AgentDecision,
    tier: Tier,
    previous_decisions: List[TierResult],
    correct_decision: str,
) -> float:
    """Score how efficiently the agent used the escalation chain."""
    # Best: Correct decision at Tier 1
    # Good: Correct escalation then correct decision
    # Bad: Unnecessary escalation
    # Worst: Wrong decision after escalation
    
    if decision.decision.value == "escalate":
        # Can't fully evaluate until final decision
        return 0.5
    
    # Count escalations to reach this decision
    escalation_count = sum(1 for tr in previous_decisions if tr.escalated)
    
    if decision.decision.value == correct_decision:
        # Correct decision - reward efficiency
        if escalation_count == 0:
            return 1.0  # Decided correctly at first tier
        elif escalation_count == 1:
            return 0.8  # One escalation was needed
        else:
            return 0.6  # Multiple escalations
    else:
        # Wrong decision after escalation is worse
        return max(0.0, 0.3 - escalation_count * 0.1)


def _score_bias_awareness(bias_identified: str, trap_type: str) -> float:
    """Score whether agent correctly identified potential biases."""
    if not bias_identified:
        if not trap_type:
            return 0.5  # No bias to identify
        return 0.2  # Missed a bias trap
    
    bias_lower = bias_identified.lower()
    
    # Check if identified bias matches trap type
    if trap_type and trap_type.lower() in bias_lower:
        return 1.0
    
    # Check for any valid bias identification
    for keyword in BIAS_KEYWORDS:
        if keyword in bias_lower:
            return 0.7
    
    return 0.4  # Some awareness shown


def grade_episode(state) -> Dict[str, Any]:
    """
    Grade an entire episode.
    
    Returns summary metrics for the episode.
    """
    if not state or not state.decisions_made:
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "efficiency": 0.0,
            "details": "No decisions made.",
        }
    
    accuracy = state.correct_decisions / state.decisions_made
    
    # Efficiency: penalize excessive escalations
    max_reasonable_escalations = state.posts_in_episode  # At most 1 per post
    escalation_ratio = state.escalations / max(1, max_reasonable_escalations)
    efficiency = max(0.0, 1.0 - max(0, escalation_ratio - 1) * 0.2)
    
    # Final score
    score = (
        accuracy * 0.6 +
        efficiency * 0.2 +
        (state.cumulative_reward / max(1, state.step_count)) * 0.2
    )
    score = max(0.0, min(1.0, score))
    
    escalation_rate = state.escalations / max(1, state.decisions_made + state.escalations)
    
    return {
        "score": score,
        "accuracy": accuracy,
        "efficiency": efficiency,
        "escalation_rate": escalation_rate,
        "total_reward": state.cumulative_reward,
        "decisions": state.decisions_made,
        "escalations": state.escalations,
        "correct": state.correct_decisions,
        "bias_triggers": state.bias_triggers,
    }
