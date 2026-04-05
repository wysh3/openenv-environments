"""ConstraintSolver Environment implementation."""

import json
import random
from typing import Any, Optional
from pathlib import Path

from openenv.core import Environment
from .models import ConstraintAction, ConstraintObservation, ConstraintState
from .grader import ConstraintGrader


class ConstraintSolverEnv(Environment[ConstraintAction, ConstraintObservation, ConstraintState]):
    """Constraint Satisfaction Benchmark Environment."""
    
    def __init__(self, scenario_path: str = None, task_type: str = None):
        super().__init__()
        self.scenario_path = scenario_path
        self.task_type = task_type
        self.scenarios = self._load_scenarios()
        self._state: Optional[ConstraintState] = None
        self.grader = ConstraintGrader()
    
    def _load_scenarios(self) -> list:
        """Load scenarios from JSON file."""
        if self.scenario_path:
            path = Path(self.scenario_path)
        else:
            scenarios_dir = Path(__file__).parent.parent / "scenarios"
            if self.task_type:
                path = scenarios_dir / f"{self.task_type}.json"
            else:
                path = scenarios_dir / "all_tasks.json"
        
        if not path.exists():
            # Try loading from all files
            scenarios = []
            scenarios_dir = Path(__file__).parent.parent / "scenarios"
            for json_file in scenarios_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    scenarios.extend(data.get("scenarios", []))
            return scenarios
        
        with open(path, 'r') as f:
            data = json.load(f)
            return data.get("scenarios", [])
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any
    ) -> ConstraintObservation:
        """Reset environment with a new scenario."""
        if not self.scenarios:
            raise ValueError(f"No scenarios loaded")
        
        scenario_id = kwargs.get("scenario_id")
        
        if seed is not None:
            random.seed(seed)
        
        if scenario_id is not None:
            scenario = self.scenarios[scenario_id % len(self.scenarios)]
        else:
            # Filter by task type if specified
            if self.task_type:
                filtered = [s for s in self.scenarios if s.get("task_type") == self.task_type]
                scenario = random.choice(filtered) if filtered else random.choice(self.scenarios)
            else:
                scenario = random.choice(self.scenarios)
        
        self._state = ConstraintState(
            scenario=scenario,
            task_type=scenario.get("task_type", ""),
            ground_truth=scenario.get("ground_truth", {}),
            agent_action=None,
            score=0.0,
            constraint_results={},
            episode_id=episode_id,
            step_count=0
        )
        
        return ConstraintObservation(
            task_type=scenario.get("task_type", ""),
            task_description=scenario.get("task_description", ""),
            problem_data=scenario.get("problem_data", {}),
            hard_constraints=scenario.get("hard_constraints", []),
            soft_constraints=scenario.get("soft_constraints", []),
            constraint_violations=[],
            constraints_satisfied=0,
            constraints_total=len(scenario.get("hard_constraints", [])),
            step_count=0,
            max_steps=1,
            done=False,
            reward=None
        )
    
    def step(
        self,
        action: ConstraintAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any
    ) -> ConstraintObservation:
        """Process agent action and return observation with reward."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")
        
        self._state.agent_action = action
        self._state.step_count += 1
        
        # Grade the solution
        score = self.grader.grade(self._state)
        self._state.score = score
        
        # Collect violations
        violations = [
            k for k, v in self._state.constraint_results.items() 
            if v is False
        ]
        satisfied = sum(1 for v in self._state.constraint_results.values() if v is True)
        total = len(self._state.constraint_results)
        
        return ConstraintObservation(
            task_type=self._state.task_type,
            task_description=self._state.scenario.get("task_description", ""),
            problem_data=self._state.scenario.get("problem_data", {}),
            hard_constraints=self._state.scenario.get("hard_constraints", []),
            soft_constraints=self._state.scenario.get("soft_constraints", []),
            constraint_violations=violations,
            constraints_satisfied=satisfied,
            constraints_total=total,
            step_count=self._state.step_count,
            max_steps=1,
            done=True,
            reward=score
        )
    
    @property
    def state(self) -> ConstraintState:
        """Get the current environment state."""
        if self._state is None:
            return ConstraintState()
        return self._state
