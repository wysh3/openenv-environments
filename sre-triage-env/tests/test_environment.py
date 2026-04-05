import pytest
from sre_triage_env import SRETriageEnv, SREAction


class TestEnvironment:
    
    def test_reset_loads_scenario(self):
        env = SRETriageEnv(difficulty="easy")
        obs = env.reset(scenario_id=0)
        
        assert obs is not None
        assert obs.task_description != ""
        assert obs.step_count == 0
        assert obs.max_steps == 1
        assert obs.done is False
    
    def test_reset_with_official_params(self):
        env = SRETriageEnv(difficulty="easy")
        
        obs1 = env.reset(seed=42)
        assert obs1 is not None
        
        obs2 = env.reset(episode_id="test-episode")
        assert obs2 is not None
        
        obs3 = env.reset(scenario_id=0)
        assert obs3 is not None
    
    def test_step_returns_observation(self):
        env = SRETriageEnv(difficulty="easy")
        env.reset(scenario_id=0)
        
        action = SREAction(severity="P2", affected_service="user_service")
        obs = env.step(action)
        
        assert obs is not None
        assert isinstance(obs.reward, (int, float))
        assert 0.0 <= obs.reward <= 1.0
        assert obs.done is True
    
    def test_state_is_property(self):
        """state is a property, not a method."""
        env = SRETriageEnv(difficulty="easy")
        env.reset(scenario_id=0)
        
        state = env.state  # Property access, not method call
        assert state is not None
        assert state.scenario is not None
    
    def test_step_before_reset_raises_error(self):
        env = SRETriageEnv(difficulty="easy")
        action = SREAction(severity="P1")
        
        with pytest.raises(RuntimeError):
            env.step(action)
    
    def test_specific_scenario_selection(self):
        env = SRETriageEnv(difficulty="easy")
        obs1 = env.reset(scenario_id=0)
        obs2 = env.reset(scenario_id=0)
        
        assert obs1.task_description == obs2.task_description
    
    def test_correct_answer_high_score(self):
        env = SRETriageEnv(difficulty="easy")
        env.reset(scenario_id=0)
        
        gt = env.state.ground_truth  # Property access
        action = SREAction(
            severity=gt.get("severity", "P1"),
            affected_service=gt.get("affected_service", "")
        )
        
        obs = env.step(action)
        assert obs.reward >= 0.6
    
    def test_wrong_answer_low_score(self):
        env = SRETriageEnv(difficulty="easy")
        env.reset(scenario_id=0)
        
        action = SREAction(severity="P5", affected_service="nonexistent_service")
        obs = env.step(action)
        
        assert obs.reward < 0.5
    
    def test_different_difficulties(self):
        env_easy = SRETriageEnv(difficulty="easy")
        env_medium = SRETriageEnv(difficulty="medium")
        env_hard = SRETriageEnv(difficulty="hard")
        
        obs_easy = env_easy.reset(scenario_id=0)
        obs_medium = env_medium.reset(scenario_id=0)
        obs_hard = env_hard.reset(scenario_id=0)
        
        assert obs_easy.task_description != ""
        assert obs_medium.task_description != ""
        assert obs_hard.task_description != ""
