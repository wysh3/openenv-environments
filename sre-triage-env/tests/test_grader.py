import pytest
from sre_triage_env import SREGrader, SREAction, SREState


class TestGrader:
    
    def test_severity_exact_match(self):
        """Test exact severity match gives full credit."""
        action = SREAction(severity="P1")
        state = SREState(
            task_type="severity",
            ground_truth={"severity": "P1"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 1.0
    
    def test_severity_close_match(self):
        """Test adjacent severity gives partial credit."""
        action = SREAction(severity="P2")
        state = SREState(
            task_type="severity",
            ground_truth={"severity": "P1"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.5
    
    def test_severity_two_off(self):
        """Test two levels off gives smaller partial credit."""
        action = SREAction(severity="P3")
        state = SREState(
            task_type="severity",
            ground_truth={"severity": "P1"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.25
    
    def test_severity_far_off(self):
        """Test completely wrong severity gives no credit."""
        action = SREAction(severity="P5")
        state = SREState(
            task_type="severity",
            ground_truth={"severity": "P1"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.0
    
    def test_root_cause_perfect(self):
        """Test perfect root cause identification."""
        action = SREAction(severity="P1", affected_service="database")
        state = SREState(
            task_type="root_cause",
            ground_truth={"severity": "P1", "affected_service": "database"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 1.0
    
    def test_root_cause_service_only(self):
        """Test correct service but wrong severity."""
        action = SREAction(severity="P2", affected_service="database")
        state = SREState(
            task_type="root_cause",
            ground_truth={"severity": "P1", "affected_service": "database"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.6
    
    def test_root_cause_severity_only(self):
        """Test correct severity but wrong service."""
        action = SREAction(severity="P1", affected_service="cache")
        state = SREState(
            task_type="root_cause",
            ground_truth={"severity": "P1", "affected_service": "database"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.2
    
    def test_root_cause_both_wrong(self):
        """Test both service and severity wrong."""
        action = SREAction(severity="P3", affected_service="cache")
        state = SREState(
            task_type="root_cause",
            ground_truth={"severity": "P1", "affected_service": "database"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.0
    
    def test_silent_failure_correct_detection_with_service(self):
        """Test correctly detecting incident and identifying service."""
        action = SREAction(is_incident=True, affected_service="payment_service")
        state = SREState(
            task_type="silent_failure",
            ground_truth={"is_incident": True, "affected_service": "payment_service"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 1.0
    
    def test_silent_failure_correct_detection_only(self):
        """Test correctly detecting incident but wrong service."""
        action = SREAction(is_incident=True, affected_service="database")
        state = SREState(
            task_type="silent_failure",
            ground_truth={"is_incident": True, "affected_service": "payment_service"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.6
    
    def test_silent_failure_no_incident_detected(self):
        """Test correctly identifying no incident."""
        action = SREAction(is_incident=False, severity="NONE")
        state = SREState(
            task_type="silent_failure",
            ground_truth={"is_incident": False}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 1.0
    
    def test_silent_failure_false_positive(self):
        """Test incorrectly identifying incident when there is none."""
        action = SREAction(is_incident=True, affected_service="database")
        state = SREState(
            task_type="silent_failure",
            ground_truth={"is_incident": False}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 0.0
    
    def test_no_action_returns_zero(self):
        """Test that no action returns zero score."""
        state = SREState(
            task_type="severity",
            ground_truth={"severity": "P1"},
            agent_action=None
        )
        
        score = SREGrader.grade(state)
        assert score == 0.0
    
    def test_case_insensitive_service(self):
        """Test service name matching is case insensitive."""
        action = SREAction(severity="P1", affected_service="DataBase")
        state = SREState(
            task_type="root_cause",
            ground_truth={"severity": "P1", "affected_service": "database"}
        )
        state.agent_action = action
        
        score = SREGrader.grade(state)
        assert score == 1.0
