import json
import random
from typing import Any, Optional
from pathlib import Path

from openenv.core import Environment
from .models import SREAction, SREObservation, SREState
from .grader import SREGrader


class SRETriageEnv(Environment[SREAction, SREObservation, SREState]):
    """SRE Incident Response Triage Environment."""
    
    def __init__(self, scenario_path: str = None, difficulty: str = "medium"):
        super().__init__()
        self.difficulty = difficulty
        self.scenario_path = scenario_path
        self.scenarios = self._load_scenarios()
        self._state: Optional[SREState] = None
        self.grader = SREGrader()
    
    def _load_scenarios(self) -> list:
        """Load scenarios from JSON file."""
        if self.scenario_path:
            path = Path(self.scenario_path)
        else:
            scenarios_dir = Path(__file__).parent.parent / "scenarios"
            path = scenarios_dir / f"{self.difficulty}.json"
        
        if not path.exists():
            return []
        
        with open(path, 'r') as f:
            data = json.load(f)
            return data.get("scenarios", [])
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any
    ) -> SREObservation:
        """Reset environment with a new scenario."""
        if not self.scenarios:
            raise ValueError(f"No scenarios loaded from {self.scenario_path or self.difficulty}")
        
        scenario_id = kwargs.get("scenario_id")
        
        if seed is not None:
            random.seed(seed)
        
        if scenario_id is not None:
            scenario = self.scenarios[scenario_id % len(self.scenarios)]
        else:
            scenario = random.choice(self.scenarios)
        
        self._state = SREState(
            scenario=scenario,
            task_type=scenario.get("task_type", "severity"),
            ground_truth=scenario.get("ground_truth", {}),
            agent_action=None,
            score=0.0,
            episode_id=episode_id,
            step_count=0
        )
        
        return SREObservation(
            task_description=scenario.get("task_description", ""),
            metrics=scenario.get("metrics", {}),
            alerts=scenario.get("alerts", []),
            logs=scenario.get("logs", []),
            service_dependencies=scenario.get("service_dependencies", {}),
            previous_incidents=scenario.get("previous_incidents", []),
            step_count=0,
            max_steps=1,
            done=False,
            reward=None
        )
    
    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any
    ) -> SREObservation:
        """Process agent action and return observation with reward."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")
        
        self._state.agent_action = action
        self._state.step_count += 1
        
        score = self.grader.grade(self._state)
        self._state.score = score
        
        return SREObservation(
            task_description=self._state.scenario.get("task_description", ""),
            metrics=self._state.scenario.get("metrics", {}),
            alerts=self._state.scenario.get("alerts", []),
            logs=self._state.scenario.get("logs", []),
            service_dependencies=self._state.scenario.get("service_dependencies", {}),
            previous_incidents=self._state.scenario.get("previous_incidents", []),
            step_count=self._state.step_count,
            max_steps=1,
            done=True,
            reward=score
        )
    
    @property
    def state(self) -> SREState:
        """Get the current environment state."""
        if self._state is None:
            return SREState()
        return self._state
