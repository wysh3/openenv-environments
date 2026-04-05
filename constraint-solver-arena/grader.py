"""Deterministic grader for constraint satisfaction problems."""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from .models import ConstraintState, ConstraintAction


class ConstraintGrader:
    """100% deterministic grading for constraint satisfaction problems."""
    
    def grade(self, state: ConstraintState) -> float:
        """Grade agent action against ground truth. Returns 0.0-1.0."""
        if state.agent_action is None:
            return 0.0
        
        task_type = state.task_type
        
        if task_type == "meeting_scheduler":
            return self._grade_meeting(state)
        elif task_type == "resource_allocator":
            return self._grade_resource(state)
        elif task_type == "travel_planner":
            return self._grade_travel(state)
        else:
            return 0.0
    
    def _grade_meeting(self, state: ConstraintState) -> float:
        """Grade meeting scheduler solution."""
        action = state.agent_action
        scenario = state.scenario
        ground_truth = state.ground_truth
        
        if not action.meeting_day or not action.meeting_start or not action.meeting_end:
            state.constraint_results = {"valid_format": False}
            return 0.0
        
        # Parse times
        try:
            start = self._parse_time(action.meeting_start)
            end = self._parse_time(action.meeting_end)
        except ValueError:
            state.constraint_results = {"valid_time_format": False}
            return 0.0
        
        constraints_met = 0
        total_constraints = 0
        results = {}
        
        problem = scenario.get("problem_data", {})
        participants = problem.get("participants", [])
        duration_required = problem.get("duration_minutes", 60)
        
        # Constraint 1: Duration matches requirement
        total_constraints += 1
        actual_duration = (end - start)
        if actual_duration >= duration_required:
            constraints_met += 1
            results["duration_valid"] = True
        else:
            results["duration_valid"] = False
        
        # Constraint 2: All participants available
        for p in participants:
            total_constraints += 1
            p_name = p.get("name", "Unknown")
            availability = p.get("availability", {})
            day_avail = availability.get(action.meeting_day, [])
            
            is_available = False
            for window in day_avail:
                w_start = self._parse_time(window.get("start", "00:00"))
                w_end = self._parse_time(window.get("end", "00:00"))
                if start >= w_start and end <= w_end:
                    is_available = True
                    break
            
            results[f"participant_{p_name}_available"] = is_available
            if is_available:
                constraints_met += 1
        
        # Constraint 3: Room constraints (if any)
        room_constraints = problem.get("room_constraints", {})
        if room_constraints and action.meeting_room:
            total_constraints += 1
            room_avail = room_constraints.get(action.meeting_room, {})
            day_avail = room_avail.get(action.meeting_day, [])
            
            room_ok = False
            for window in day_avail:
                w_start = self._parse_time(window.get("start", "00:00"))
                w_end = self._parse_time(window.get("end", "00:00"))
                if start >= w_start and end <= w_end:
                    room_ok = True
                    break
            
            results["room_available"] = room_ok
            if room_ok:
                constraints_met += 1
        
        state.constraint_results = results
        return constraints_met / total_constraints if total_constraints > 0 else 0.0
    
    def _grade_resource(self, state: ConstraintState) -> float:
        """Grade resource allocation solution."""
        action = state.agent_action
        scenario = state.scenario
        
        if not action.assignments:
            state.constraint_results = {"valid_format": False}
            return 0.0
        
        problem = scenario.get("problem_data", {})
        tasks = {t["id"]: t for t in problem.get("tasks", [])}
        workers = {w["id"]: w for w in problem.get("workers", [])}
        
        constraints_met = 0
        total_constraints = 0
        results = {}
        worker_hours_used = {w_id: 0 for w_id in workers}
        
        for assignment in action.assignments:
            task_id = assignment.get("task_id")
            worker_id = assignment.get("worker_id")
            
            if task_id not in tasks or worker_id not in workers:
                total_constraints += 1
                results[f"valid_assignment_{task_id}"] = False
                continue
            
            task = tasks[task_id]
            worker = workers[worker_id]
            
            # Constraint: Worker has required skills
            total_constraints += 1
            required_skills = set(task.get("skills_required", []))
            worker_skills = set(worker.get("skills", []))
            has_skills = required_skills.issubset(worker_skills)
            results[f"skills_{task_id}_{worker_id}"] = has_skills
            if has_skills:
                constraints_met += 1
            
            # Constraint: Worker has available hours
            total_constraints += 1
            task_hours = task.get("hours", 0)
            worker_hours_used[worker_id] += task_hours
            available = worker.get("available_hours", 0)
            within_hours = worker_hours_used[worker_id] <= available
            results[f"hours_{task_id}_{worker_id}"] = within_hours
            if within_hours:
                constraints_met += 1
        
        # Constraint: All tasks assigned
        total_constraints += 1
        assigned_tasks = {a.get("task_id") for a in action.assignments}
        all_assigned = set(tasks.keys()).issubset(assigned_tasks)
        results["all_tasks_assigned"] = all_assigned
        if all_assigned:
            constraints_met += 1
        
        state.constraint_results = results
        return constraints_met / total_constraints if total_constraints > 0 else 0.0
    
    def _grade_travel(self, state: ConstraintState) -> float:
        """Grade travel planner solution."""
        action = state.agent_action
        scenario = state.scenario
        
        if not action.itinerary:
            state.constraint_results = {"valid_format": False}
            return 0.0
        
        problem = scenario.get("problem_data", {})
        budget = problem.get("budget", float("inf"))
        dependencies = problem.get("dependencies", [])
        time_constraints = problem.get("time_constraints", {})
        
        constraints_met = 0
        total_constraints = 0
        results = {}
        
        # Constraint: Within budget
        total_constraints += 1
        total_cost = action.total_cost or sum(
            item.get("cost", 0) for item in action.itinerary
        )
        within_budget = total_cost <= budget
        results["within_budget"] = within_budget
        if within_budget:
            constraints_met += 1
        
        # Constraint: Dependencies respected (A before B)
        activity_order = {
            item.get("activity"): i 
            for i, item in enumerate(action.itinerary)
        }
        for dep in dependencies:
            total_constraints += 1
            before = dep.get("before")
            after = dep.get("after")
            
            if before in activity_order and after in activity_order:
                order_ok = activity_order[before] < activity_order[after]
            else:
                order_ok = False
            
            results[f"dependency_{before}_before_{after}"] = order_ok
            if order_ok:
                constraints_met += 1
        
        # Constraint: Time windows respected
        day_start = self._parse_time(time_constraints.get("day_start", "08:00"))
        day_end = self._parse_time(time_constraints.get("day_end", "22:00"))
        
        for item in action.itinerary:
            total_constraints += 1
            item_start = self._parse_time(item.get("start_time", "00:00"))
            item_end = self._parse_time(item.get("end_time", "23:59"))
            
            within_day = item_start >= day_start and item_end <= day_end
            results[f"time_window_{item.get('activity')}"] = within_day
            if within_day:
                constraints_met += 1
        
        # Constraint: No overlapping activities
        total_constraints += 1
        no_overlap = self._check_no_overlap(action.itinerary)
        results["no_overlap"] = no_overlap
        if no_overlap:
            constraints_met += 1
        
        state.constraint_results = results
        return constraints_met / total_constraints if total_constraints > 0 else 0.0
    
    def _parse_time(self, time_str: str) -> int:
        """Parse HH:MM to minutes from midnight."""
        if not time_str:
            return 0
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    
    def _check_no_overlap(self, itinerary: List[Dict[str, Any]]) -> bool:
        """Check if no activities overlap."""
        events = []
        for item in itinerary:
            start = self._parse_time(item.get("start_time", "00:00"))
            end = self._parse_time(item.get("end_time", "00:00"))
            events.append((start, end))
        
        events.sort()
        for i in range(len(events) - 1):
            if events[i][1] > events[i + 1][0]:
                return False
        return True
