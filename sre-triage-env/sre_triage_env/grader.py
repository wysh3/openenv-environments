from typing import Any, Dict
from .models import SREAction, SREState


class SREGrader:
    """Deterministic scoring for SRE triage tasks."""
    
    SEVERITY_ORDER = ["P1", "P2", "P3", "P4", "P5", "NONE"]
    
    @staticmethod
    def grade(state: SREState) -> float:
        """Score agent action against ground truth. Returns 0.0-1.0."""
        if state.agent_action is None:
            return 0.0
        
        graders = {
            "severity": SREGrader._grade_severity,
            "root_cause": SREGrader._grade_root_cause,
            "silent_failure": SREGrader._grade_silent_failure,
        }
        
        grader = graders.get(state.task_type)
        return grader(state.agent_action, state.ground_truth) if grader else 0.0
    
    @staticmethod
    def _grade_severity(action: SREAction, gt: Dict[str, Any]) -> float:
        """Full credit for exact match, partial for close."""
        correct = gt.get("severity", "")
        agent = action.severity.upper()
        
        if agent == correct:
            return 1.0
        
        try:
            dist = abs(SREGrader.SEVERITY_ORDER.index(correct) - 
                      SREGrader.SEVERITY_ORDER.index(agent))
            return {1: 0.5, 2: 0.25}.get(dist, 0.0)
        except ValueError:
            return 0.0
    
    @staticmethod
    def _grade_root_cause(action: SREAction, gt: Dict[str, Any]) -> float:
        """Rewards correct service + severity."""
        service_ok = action.affected_service.lower() == gt.get("affected_service", "").lower()
        severity_ok = action.severity.upper() == gt.get("severity", "")
        
        if service_ok and severity_ok:
            return 1.0
        return 0.6 if service_ok else (0.2 if severity_ok else 0.0)
    
    @staticmethod
    def _grade_silent_failure(action: SREAction, gt: Dict[str, Any]) -> float:
        """Focus on correctly identifying whether incident exists."""
        actual = gt.get("is_incident", True)
        detected = action.is_incident
        
        if detected != actual:
            return 0.0
        
        score = 0.6
        if actual:
            if action.affected_service.lower() == gt.get("affected_service", "").lower():
                score += 0.4
        else:
            if not action.affected_service or action.severity.upper() == "NONE":
                score += 0.4
        
        return score
