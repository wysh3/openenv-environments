from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import SREAction, SREObservation, SREState


class SRETriageClient(EnvClient[SREAction, SREObservation, SREState]):
    """Client for connecting to remote SRE Triage Environment server."""
    
    def _step_payload(self, action: SREAction) -> Dict[str, Any]:
        return action.model_dump()
    
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SREObservation]:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=SREObservation(**obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )
    
    def _parse_state(self, payload: Dict[str, Any]) -> SREState:
        return SREState(**payload)
