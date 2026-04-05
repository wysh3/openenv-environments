"""Tests for ConstraintSolver Arena."""
import pytest
from constraint_solver import ConstraintSolverEnv, ConstraintAction


class TestMeetingScheduler:
    """Tests for meeting scheduler task."""
    
    def test_valid_meeting(self):
        """Test a valid meeting slot."""
        env = ConstraintSolverEnv(task_type="meeting_scheduler")
        obs = env.reset(scenario_id=0)
        
        assert obs.task_type == "meeting_scheduler"
        assert len(obs.hard_constraints) > 0
        
        # Valid slot from ground truth: Monday 10:00-11:00
        action = ConstraintAction(
            task_type="meeting_scheduler",
            meeting_day="Monday",
            meeting_start="10:00",
            meeting_end="11:00",
            reasoning="All participants available"
        )
        
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward > 0.5  # Should score well
    
    def test_invalid_meeting(self):
        """Test an invalid meeting slot."""
        env = ConstraintSolverEnv(task_type="meeting_scheduler")
        env.reset(scenario_id=0)
        
        # Invalid slot - wrong duration
        action = ConstraintAction(
            task_type="meeting_scheduler",
            meeting_day="Monday",
            meeting_start="10:00",
            meeting_end="10:30",  # Only 30 min, need 60
            reasoning="Test"
        )
        
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward < 1.0  # Should not get full score


class TestResourceAllocator:
    """Tests for resource allocator task."""
    
    def test_valid_allocation(self):
        """Test a valid resource allocation."""
        env = ConstraintSolverEnv(task_type="resource_allocator")
        obs = env.reset(scenario_id=0)
        
        assert obs.task_type == "resource_allocator"
        
        # Valid assignment from ground truth
        action = ConstraintAction(
            task_type="resource_allocator",
            assignments=[
                {"task_id": "T1", "worker_id": "W1"},
                {"task_id": "T2", "worker_id": "W2"},
                {"task_id": "T3", "worker_id": "W2"}
            ],
            reasoning="Matched skills and hours"
        )
        
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward > 0.5


class TestTravelPlanner:
    """Tests for travel planner task."""
    
    def test_valid_itinerary(self):
        """Test a valid travel itinerary."""
        env = ConstraintSolverEnv(task_type="travel_planner")
        obs = env.reset(scenario_id=0)
        
        assert obs.task_type == "travel_planner"
        
        # Valid itinerary
        action = ConstraintAction(
            task_type="travel_planner",
            itinerary=[
                {"activity": "Golden Gate Bridge", "start_time": "09:00", "end_time": "10:30", "cost": 0},
                {"activity": "Fisherman's Wharf", "start_time": "11:00", "end_time": "13:00", "cost": 0},
                {"activity": "Lunch at Pier 39", "start_time": "13:00", "end_time": "14:00", "cost": 35}
            ],
            total_cost=35,
            reasoning="Followed dependencies, within budget"
        )
        
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward > 0  # Should get some score


class TestEnvironment:
    """General environment tests."""
    
    def test_reset_returns_observation(self):
        """Test reset returns valid observation."""
        env = ConstraintSolverEnv()
        obs = env.reset()
        
        assert obs.task_type in ["meeting_scheduler", "resource_allocator", "travel_planner"]
        assert obs.task_description != ""
        assert obs.done is False
    
    def test_state_property(self):
        """Test state property returns valid state."""
        env = ConstraintSolverEnv()
        env.reset()
        
        state = env.state
        assert state.task_type in ["meeting_scheduler", "resource_allocator", "travel_planner"]
    
    def test_step_without_reset_raises(self):
        """Test step without reset raises error."""
        env = ConstraintSolverEnv()
        
        action = ConstraintAction(
            task_type="meeting_scheduler",
            meeting_day="Monday",
            meeting_start="10:00",
            meeting_end="11:00"
        )
        
        with pytest.raises(RuntimeError):
            env.step(action)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
