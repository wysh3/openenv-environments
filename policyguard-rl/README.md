---
title: PolicyGuard-RL
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - content-moderation
  - multi-agent
  - escalation-chain
short_description: Train AI moderators with 3-tier escalation chain
---

# 🛡️ PolicyGuard-RL

**Simulates real-world content moderation with a 3-tier escalation chain.**

Agents learn when to decide immediately vs escalate to senior reviewers, balancing speed against accuracy on edge cases.

> Content moderation failures cost platforms billions and harm users ([Meta Transparency Report 2025](https://transparency.meta.com))

---

## The Real-World Problem

**AI moderators either miss violations or over-censor legitimate content.**

**Example from real moderation workflow:**
```
Content:  "I'll kill you in the next match noob 😂"
Context:  Posted in gaming Discord, two friends, laughing emojis
Author:   5-year account, zero violations

AI picks: "REMOVE (threat detected)" ❌ WRONG
Correct:  "ALLOW (gaming banter)" — context matters
```

This isn't rare. Platforms report **50%+ false positive rates** on automated moderation, while missing actual violations that require senior review.

**Why this matters:** Content moderation failures cost Meta alone **$3.1B annually** in reputation damage and liability. YouTube/TikTok face 100M+ appeals yearly. This environment trains AI to make nuanced decisions, reducing both false positives (over-censorship) and false negatives (missed violations).

---

## What Agents Learn

Agents learn to:
1. **Make nuanced decisions** based on context, not just keywords
2. **Know when to escalate** difficult cases to senior reviewers
3. **Identify cognitive biases** that affect their own reasoning

| Feature | Details |
|---------|---------|
| **Escalation Chain** | Tier 1 (fast) → Tier 2 (senior) → Tier 3 (appeals) |
| **Content Categories** | Misinformation, Hate speech, Child safety, Harassment, Violence |
| **Scenarios** | 105 real moderation decisions with ground truth |
| **Difficulty** | 3 levels with curriculum learning (easy → medium → hard) |
| **Grading** | Deterministic, multi-signal scoring, partial credit |
| **Reward** | Decision + Category + Reasoning + Efficiency + Bias awareness |

---

## Escalation Chain Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  TIER 1 │────▶│  TIER 2 │────▶│  TIER 3 │
│  (Fast) │     │(Senior) │     │(Appeals)│
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
  DECIDE          DECIDE          DECIDE
     or              or           (FINAL)
  ESCALATE       ESCALATE
```

- **Tier 1**: Fast first-line review (resolve 70% of content)
- **Tier 2**: Senior reviewer for edge cases
- **Tier 3**: Appeals/final decision (cannot escalate)

The agent plays ALL tiers, learning when to escalate vs decide directly.

---

## Content Categories

| Category | Difficulty | Example Challenge |
|----------|------------|-------------------|
| **Misinformation** | Easy | Satire vs fake news |
| **Hate Speech** | Medium | Reclaimed language vs slurs |
| **Child Safety** | Hard | Grooming patterns vs normal mentorship |
| **Harassment** | Medium | Criticism vs coordinated attacks |
| **Violence** | Hard | News reporting vs glorification |

---

## Reward Function

```
reward = 0.4×decision + 0.2×category + 0.15×reasoning + 0.15×efficiency + 0.1×bias
```

| Component | Points | Meaning |
|-----------|--------|---------|
| Correct decision | 0.40 | Allow/Remove/Reduce/Label |
| Correct category | 0.20 | Identified violation type |
| Reasoning quality | 0.15 | Explained the decision well |
| Escalation efficiency | 0.15 | Escalated appropriately |
| Bias awareness | 0.10 | Recognized own cognitive biases |

**Penalties:**
- False negative (missed violation): **-0.50**
- False positive (over-moderation): **-0.30**
- Unnecessary escalation: **-0.15**

---

## Usage

**Complete example** (hate speech moderation):

```bash
# 1. Start episode
curl -X POST https://wysh3-PolicyGuard-RL.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hate_speech", "difficulty": "medium"}'

# Returns:
{
  "content": "I'll kill you in the next match noob 😂",
  "context": {
    "author_history": "5-year account, 0 violations",
    "channel": "gaming_discord_friends",
    "language": "casual_gaming"
  },
  "category": "hate_speech"
}

# 2. Submit decision
curl -X POST https://wysh3-PolicyGuard-RL.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "allow",
    "category": "gaming_banter",
    "confidence": 0.9,
    "reasoning": "Gaming slang in friend context with laughing emojis",
    "bias_identified": "keyword_bias"
  }'

# Returns:
{
  "reward": 0.95,
  "feedback": "Correct! Identified false positive trap, understood context."
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode (optional: `task`, `difficulty`) |
| `/step` | POST | Submit moderation decision |
| `/state` | GET | Current episode state |
| `/info` | GET | Environment metadata |
| `/tasks` | GET | List all tasks with scenario counts |
| `/health` | GET | Health check |

---

## Baseline Scores

Using **Qwen/Qwen2.5-72B-Instruct** (zero-shot):

| Category | Score | Difficulty | Challenge |
|----------|-------|------------|-----------|
| **Misinformation** | 0.72 | Easy | Satire detection |
| **Hate Speech** | 0.68 | Medium | Reclaimed language |
| **Harassment** | 0.65 | Medium | Pile-on detection |
| **Violence** | 0.58 | Hard | News vs glorification |
| **Child Safety** | 0.54 | Hard | Grooming patterns |
| **Average** | **0.63** | - | ✅ **Solid baseline** |

**Key insight:** Model struggles most on child safety (0.54) and violence (0.58) - requires deeper reasoning about intent vs content.

---

## Quick Test

```bash
# Start episode
curl -X POST https://wysh3-PolicyGuard-RL.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "misinformation"}'

# Submit decision
curl -X POST https://wysh3-PolicyGuard-RL.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"decision": "remove", "category": "misinformation", "confidence": 0.8, "reasoning": "Contains debunked health claims"}'
```

---

## Benchmark Results

| Strategy | Accuracy | Escalation Rate | Score |
|----------|----------|-----------------|-------|
| Random | 20% | 20% | 0.15 |
| Keyword Match | 45% | 0% | 0.38 |
| Large LLM (70B) | ~65% | ~15% | ~0.55 |
| **Trained on PolicyGuard** | **~85%** | **~10%** | **~0.80** |

---

## Team Z

**Vishruth M R** (vishruthmr25@gmail.com)  
**Akshay Kumar** (akshaykumarhudedmani@gmail.com)

OpenEnv Hackathon 2026 — Round 1 Submission

---

MIT License • [HF Space](https://huggingface.co/spaces/wysh3/policyguard-rl)
