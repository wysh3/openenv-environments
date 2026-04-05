from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core import Action, Observation, State


class SREAction(Action):
    """Agent response for SRE incident triage."""
    severity: str = Field(default="", description="P1, P2, P3, P4, P5, or NONE")
    affected_service: str = Field(default="", description="Service name identified as root cause")
    is_incident: bool = Field(default=True, description="Whether this is actually an incident")
    reasoning: str = Field(default="", description="Agent's explanation of their analysis")


class SREObservation(Observation):
    """Environment state shown to agent. Inherits done, reward, metadata from Observation."""
    task_description: str = Field(default="")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    service_dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    previous_incidents: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = Field(default=0)
    max_steps: int = Field(default=1)


class SREState(State):
    """Internal environment state. Inherits episode_id, step_count from State."""
    scenario: Dict[str, Any] = Field(default_factory=dict)
    task_type: str = Field(default="")
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
    agent_action: Optional[SREAction] = Field(default=None)
    score: float = Field(default=0.0)
