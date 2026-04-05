import pytest
from openenv.core.client_types import StepResult
from sre_triage_env import SRETriageClient, SREAction, SREObservation, SREState


class TestClient:
    
    def test_client_instantiation(self):
        client = SRETriageClient(base_url="http://localhost:7860")
        assert client is not None
    
    def test_step_payload_conversion(self):
        client = SRETriageClient(base_url="http://localhost:7860")
        action = SREAction(severity="P1", affected_service="database", is_incident=True)
        
        payload = client._step_payload(action)
        
        assert payload["severity"] == "P1"
        assert payload["affected_service"] == "database"
        assert payload["is_incident"] is True
    
    def test_parse_result(self):
        client = SRETriageClient(base_url="http://localhost:7860")
        
        payload = {
            "observation": {
                "task_description": "Test task",
                "metrics": {},
                "alerts": [],
                "logs": [],
                "service_dependencies": {},
                "previous_incidents": [],
                "step_count": 1,
                "max_steps": 1,
                "done": True
            },
            "reward": 0.75,
            "done": True
        }
        
        result = client._parse_result(payload)
        
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, SREObservation)
        assert result.reward == 0.75
        assert result.done is True
    
    def test_parse_state(self):
        client = SRETriageClient(base_url="http://localhost:7860")
        
        payload = {
            "scenario": {"id": "test_001"},
            "task_type": "severity",
            "ground_truth": {"severity": "P1"}
        }
        
        state = client._parse_state(payload)
        
        assert isinstance(state, SREState)
        assert state.task_type == "severity"
