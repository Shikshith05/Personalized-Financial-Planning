from dataclasses import dataclass
import copy
import math

from constants import PRIORITY_WEIGHTS
from goal_allocator import allocate_savings, goal_urgency
import reward


@dataclass
class EnvironmentObservation:
    available_money_ratio: float
    overall_goal_progress: float
    nearest_deadline_ratio: float
    highest_priority_ratio: float
    forecast_expense_ratio: float
    active_goal_count_ratio: float

    def as_list(self):
        return [
            self.available_money_ratio,
            self.overall_goal_progress,
            self.nearest_deadline_ratio,
            self.highest_priority_ratio,
            self.forecast_expense_ratio,
            self.active_goal_count_ratio,
        ]


class SimpleGoalEnv:
    def __init__(
        self,
        monthly_income: float,
        monthly_expense: float,
        goals: list,
        horizon: int,
    ):
        self.monthly_income = float(monthly_income)
        self.monthly_expense = float(monthly_expense)

        self.initial_goals = copy.deepcopy(goals)
        self.goals = copy.deepcopy(goals)
        self.current_step = 0
        self.horizon = int(horizon)

        self.observation_dim = 6
        self.action_dim = 1

    def reset(self):
        self.goals = copy.deepcopy(self.initial_goals)
        self.current_step = 0

        return self._get_obs()

    def _weighted_progress_score(self):
        if not self.goals:
            return 0.0

        total_weight = sum(
            PRIORITY_WEIGHTS.get(
                goal.priority.lower(),
                2.0,
            )
            for goal in self.goals
        )

        if total_weight <= 0:
            return 0.0

        return sum(
            PRIORITY_WEIGHTS.get(
                goal.priority.lower(),
                2.0,
            )
            * goal.progress
            for goal in self.goals
        ) / total_weight

    def step(self, action):
        raw_action = action[0] if hasattr(action, "__len__") and not isinstance(action, (str, bytes)) else action
        savings_rate = 1.0 / (1.0 + math.exp(-float(raw_action)))

        available_amount = (
            self.monthly_income
            - self.monthly_expense
        )
        savings_amount = (
            available_amount
            * savings_rate
        )

        old_goals = copy.deepcopy(self.goals)

        allocations = allocate_savings(
            savings_amount,
            self.goals,
        )

        for allocation in allocations:
            for goal in self.goals:
                if goal.name == allocation["goal"]:
                    goal.current_savings += allocation["amount"]
                    goal.current_savings = min(
                        goal.current_savings,
                        goal.target_amount,
                    )
                    break

        for goal in self.goals:
            if goal.remaining_amount <= 0:
                continue
            goal.deadline_months = max(
                goal.deadline_months - 1,
                0,
            )

        self.current_step += 1

        new_goals = copy.deepcopy(self.goals)

        old_progress = sum(
            goal.progress
            for goal in old_goals
        ) / max(len(old_goals), 1)

        new_progress = sum(
            goal.progress
            for goal in new_goals
        ) / max(len(new_goals), 1)

        total_required = sum(
            goal.remaining_amount / max(goal.deadline_months, 1)
            for goal in old_goals
            if goal.remaining_amount > 0
        )

        old_priority_score = sum(
            PRIORITY_WEIGHTS.get(
                goal.priority.lower(),
                2.0,
            )
            * goal.progress
            for goal in old_goals
        ) / max(len(old_goals), 1)

        new_priority_score = sum(
            PRIORITY_WEIGHTS.get(
                goal.priority.lower(),
                2.0,
            )
            * goal.progress
            for goal in new_goals
        ) / max(len(new_goals), 1)

        priority_reward = new_priority_score - old_priority_score

        deadline_penalty = -sum(
            1.0
            for goal in new_goals
            if goal.deadline_months == 0
            and goal.remaining_amount > 0
        )

        reward_value = reward.calculate_reward(
            old_progress=old_progress,
            new_progress=new_progress,
            savings_amount=savings_amount,
            total_required=total_required,
            priority_reward=priority_reward,
            deadline_penalty=deadline_penalty,
        )

        time_finished = self.current_step >= self.horizon
        all_goals_completed = all(
            goal.remaining_amount <= 0
            for goal in self.goals
        )

        done = (
            time_finished
            or all_goals_completed
        )

        info = {
            "savings_rate": savings_rate,
            "savings_amount": savings_amount,
            "allocations": allocations,
        }

        next_obs = self._get_obs()

        return (
            next_obs,
            reward_value,
            done,
            info,
        )

    def _get_obs(self):
        active_goals = [
            goal
            for goal in self.goals
            if goal.remaining_amount > 0
            and goal.deadline_months > 0
        ]

        available_amount = max(
            self.monthly_income - self.monthly_expense,
            0.0,
        )
        available_money_ratio = (
            available_amount
            / max(self.monthly_income, 1.0)
        )
        expense_ratio = (
            self.monthly_expense
            / max(self.monthly_income, 1.0)
        )

        if active_goals:
            most_urgent_goal = max(
                active_goals,
                key=goal_urgency,
            )

            most_urgent_remaining_ratio = (
                most_urgent_goal.remaining_amount
                / max(most_urgent_goal.target_amount, 1.0)
            )

            nearest_deadline = min(
                goal.deadline_months
                for goal in active_goals
            )

            nearest_deadline_ratio = (
                nearest_deadline
                / max(self.horizon, 1)
            )

            highest_priority_weight = max(
                PRIORITY_WEIGHTS.get(
                    goal.priority.lower(),
                    2.0,
                )
                for goal in active_goals
            )

            highest_priority_ratio = (
                highest_priority_weight / 4.0
            )

            active_goal_count_ratio = min(
                len(active_goals) / 10.0,
                1.0,
            )

        else:
            most_urgent_remaining_ratio = 0.0
            nearest_deadline_ratio = 0.0
            highest_priority_ratio = 0.0
            active_goal_count_ratio = 0.0

        observation = [
            available_money_ratio,
            most_urgent_remaining_ratio,
            nearest_deadline_ratio,
            highest_priority_ratio,
            expense_ratio,
            active_goal_count_ratio,
        ]

        return observation