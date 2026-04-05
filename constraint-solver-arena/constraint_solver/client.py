"""Client for connecting to remote ConstraintSolver Environment server."""

from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import ConstraintAction, ConstraintObservation, ConstraintState


class ConstraintSolverClient(EnvClient[ConstraintAction, ConstraintObservation, ConstraintState]):
    """Client for connecting to remote ConstraintSolver Environment server."""
    
    def _step_payload(self, action: ConstraintAction) -> Dict[str, Any]:
        return action.model_dump()
    
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ConstraintObservation]:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=ConstraintObservation(**obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )
    
    def _parse_state(self, payload: Dict[str, Any]) -> ConstraintState:
        return ConstraintState(**payload)
