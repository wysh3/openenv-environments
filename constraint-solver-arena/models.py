"""Pydantic models for ConstraintSolver Arena."""

from typing import Any, Dict, List, Optional, Union
from pydantic import Field
from openenv.core import Action, Observation, State


# ============================================================================
# CONSTRAINT TYPES
# ============================================================================

class TimeWindow:
    """Represents a time window (e.g., 9:00-12:00)."""
    def __init__(self, start: int, end: int, day: Optional[str] = None):
        self.start = start  # Minutes from midnight
        self.end = end
        self.day = day


class Constraint:
    """Base constraint class."""
    pass


# ============================================================================
# TASK 1: MEETING SCHEDULER
# ============================================================================

class MeetingSlot(Action):
    """Proposed meeting slot."""
    day: str = Field(..., description="Day of the week (Monday, Tuesday, etc.)")
    start_time: str = Field(..., description="Start time in HH:MM format")
    end_time: str = Field(..., description="End time in HH:MM format")
    room: Optional[str] = Field(default=None, description="Room assignment if required")
    reasoning: str = Field(default="", description="Explanation of why this slot works")


class MeetingProblem:
    """Meeting scheduling problem definition."""
    participants: List[Dict[str, Any]]  # [{"name": "Alice", "availability": {...}}]
    duration_minutes: int
    room_constraints: Optional[Dict[str, Any]] = None
    priority_constraints: Optional[List[str]] = None


# ============================================================================
# TASK 2: RESOURCE ALLOCATOR  
# ============================================================================

class TaskAssignment(Action):
    """Assignment of tasks to workers."""
    assignments: List[Dict[str, str]] = Field(
        ..., 
        description="List of {task_id, worker_id} assignments"
    )
    reasoning: str = Field(default="", description="Explanation of assignment logic")


class ResourceProblem:
    """Resource allocation problem definition."""
    tasks: List[Dict[str, Any]]  # [{"id": "T1", "skills_required": [...], "hours": 4}]
    workers: List[Dict[str, Any]]  # [{"id": "W1", "skills": [...], "available_hours": 8}]
    constraints: List[str]  # Additional constraints in natural language


# ============================================================================
# TASK 3: TRAVEL PLANNER
# ============================================================================

class TravelPlan(Action):
    """Proposed travel itinerary."""
    itinerary: List[Dict[str, Any]] = Field(
        ...,
        description="Ordered list of activities with times and locations"
    )
    total_cost: float = Field(..., description="Total estimated cost")
    reasoning: str = Field(default="", description="Explanation of planning logic")


class TravelProblem:
    """Travel planning problem definition."""
    destinations: List[Dict[str, Any]]  # Available activities/places
    budget: float
    time_constraints: Dict[str, Any]  # Start time, end time, durations
    dependencies: List[Dict[str, str]]  # [{"before": "A", "after": "B"}]
    preferences: Optional[List[str]] = None


# ============================================================================
# UNIFIED ACTION/OBSERVATION TYPES
# ============================================================================

class ConstraintAction(Action):
    """Unified action for all constraint tasks."""
    task_type: str = Field(..., description="meeting_scheduler, resource_allocator, or travel_planner")
    
    # Meeting Scheduler fields
    meeting_day: Optional[str] = Field(default=None, description="Day for meeting")
    meeting_start: Optional[str] = Field(default=None, description="Meeting start time HH:MM")
    meeting_end: Optional[str] = Field(default=None, description="Meeting end time HH:MM")
    meeting_room: Optional[str] = Field(default=None, description="Room assignment")
    
    # Resource Allocator fields
    assignments: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Task-to-worker assignments [{task_id, worker_id}]"
    )
    
    # Travel Planner fields
    itinerary: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Ordered travel plan with times and costs"
    )
    total_cost: Optional[float] = Field(default=None, description="Total trip cost")
    
    # Common
    reasoning: str = Field(default="", description="Agent's explanation")


class ConstraintObservation(Observation):
    """Environment observation for constraint tasks."""
    task_type: str = Field(default="", description="Type of constraint problem")
    task_description: str = Field(default="", description="Natural language problem description")
    
    # Problem data (JSON-serializable)
    problem_data: Dict[str, Any] = Field(default_factory=dict, description="Structured problem data")
    
    # Constraints explicitly listed
    hard_constraints: List[str] = Field(default_factory=list, description="Must be satisfied")
    soft_constraints: List[str] = Field(default_factory=list, description="Nice to have")
    
    # Feedback after action
    constraint_violations: List[str] = Field(default_factory=list, description="Violated constraints")
    constraints_satisfied: int = Field(default=0, description="Number of satisfied constraints")
    constraints_total: int = Field(default=0, description="Total number of constraints")
    
    step_count: int = Field(default=0)
    max_steps: int = Field(default=1)


class ConstraintState(State):
    """Internal environment state."""
    scenario: Dict[str, Any] = Field(default_factory=dict)
    task_type: str = Field(default="")
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
    agent_action: Optional[ConstraintAction] = Field(default=None)
    score: float = Field(default=0.0)
    constraint_results: Dict[str, bool] = Field(default_factory=dict)
