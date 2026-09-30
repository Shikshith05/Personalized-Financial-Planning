"""Explainable adaptive savings planner, backed by explainable_rl_planner.

The planner modules are plain scripts that import each other by bare name
(`from agent import ...`), so they are imported once here with their folder
at the front of sys.path. sms_parsing also ships a `train` module; importing
the planner first keeps the right one cached.
"""

import copy
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import POLICY_MODEL_CANDIDATES, PRIORITIES, RL_PLANNER_DIR

sys.dont_write_bytecode = True

_IMPORT_ERROR: Optional[str] = None
try:
    sys.path.insert(0, str(RL_PLANNER_DIR))
    import train as rl_train
    import explainer as rl_explainer
    import financial_planner as rl_planner
    from agent import PolicyNetwork
    from env import SimpleGoalEnv
    from goal_allocator import allocate_savings
    from goals import Goal

    if Path(rl_train.__file__).resolve().parent != RL_PLANNER_DIR.resolve():
        raise ImportError('a different `train` module was imported before the RL planner')
except Exception as exc:  # pragma: no cover - depends on the local install
    _IMPORT_ERROR = f'{type(exc).__name__}: {exc}'
finally:
    if str(RL_PLANNER_DIR) in sys.path:
        sys.path.remove(str(RL_PLANNER_DIR))

MAX_GOALS = 8
MAX_DEADLINE_MONTHS = 120


class PlannerError(ValueError):
    """Invalid planner input; the message is safe to show to the user."""


def find_policy_path() -> Optional[Path]:
    return next((path for path in POLICY_MODEL_CANDIDATES if path.exists()), None)


def train_policy(scenarios: int = 80, episodes_per_scenario: int = 25, seed: int = 42, log=print):
    """Train the savings policy with the planner's own REINFORCE loop.

    explainable_rl_planner/train.py trains on the single scenario typed in
    at the prompt. The app has to serve any user, so the same train_model()
    is run across randomly drawn incomes, expense burdens and goal sets.
    """
    import contextlib
    import io

    import torch

    if _IMPORT_ERROR:
        raise RuntimeError(f'RL planner could not be imported: {_IMPORT_ERROR}')

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = PolicyNetwork(state_dim=9)
    recent: List[float] = []

    for index in range(1, scenarios + 1):
        income = float(rng.uniform(20000, 200000))
        expense = income * float(rng.uniform(0.45, 0.9))
        available = income - expense

        goals = []
        for goal_index in range(int(rng.integers(1, 5))):
            deadline = int(rng.integers(3, 25))
            target = available * deadline * float(rng.uniform(0.15, 0.6))
            goals.append(Goal(
                name=f'goal_{goal_index}',
                target_amount=target,
                current_savings=target * float(rng.uniform(0.0, 0.4)),
                deadline_months=deadline,
                priority=str(rng.choice(PRIORITIES)),
            ))

        env = SimpleGoalEnv(
            monthly_income=income, monthly_expense=expense, goals=goals,
            horizon=max(goal.deadline_months for goal in goals),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            history = rl_train.train_model(env=env, model=model, episodes=episodes_per_scenario)
        recent.append(float(np.mean(history)))

        if index % 20 == 0:
            log(f'  scenario {index:3d}/{scenarios} | mean episode reward {np.mean(recent[-20:]):.3f}')

    model.eval()
    return model


class SavingsPlannerService:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else find_policy_path()
        self.model = None
        self.error: Optional[str] = _IMPORT_ERROR
        self._lock = threading.Lock()

        if self.error is None:
            if self.model_path is None or not self.model_path.exists():
                self.error = 'Savings policy not trained yet. Run: python -m app.backend.build_artifacts'
            else:
                try:
                    self.model = rl_planner.load_model(str(self.model_path), state_dim=9)
                except Exception as exc:
                    self.error = f'Could not load the savings policy: {exc}'

    @property
    def ready(self) -> bool:
        return self.model is not None

    def status(self) -> Dict[str, Any]:
        return {
            'ready': self.ready,
            'message': self.error or 'REINFORCE policy loaded',
            'model': 'Policy network (9 state features -> savings rate), blended with saving behaviour and deadlines',
        }

    # -- input handling ----------------------------------------------------

    @staticmethod
    def _parse_goals(raw_goals: List[dict]) -> list:
        if not raw_goals:
            raise PlannerError('Add at least one goal to plan for.')
        if len(raw_goals) > MAX_GOALS:
            raise PlannerError(f'At most {MAX_GOALS} goals are supported.')

        goals, seen = [], set()
        for index, raw in enumerate(raw_goals, start=1):
            name = str(raw.get('name') or '').strip() or f'Goal {index}'
            # Allocations are matched to goals by name, so names must be unique.
            base, suffix = name, 2
            while name.lower() in seen:
                name = f'{base} ({suffix})'
                suffix += 1
            seen.add(name.lower())

            try:
                target = float(raw.get('target_amount'))
                saved = float(raw.get('current_savings') or 0.0)
                deadline = int(float(raw.get('deadline_months')))
            except (TypeError, ValueError):
                raise PlannerError(f'"{name}": target, saved amount and deadline must be numbers.')

            priority = str(raw.get('priority') or 'medium').lower()
            if not np.isfinite(target) or target <= 0:
                raise PlannerError(f'"{name}": target amount must be greater than zero.')
            if not np.isfinite(saved) or saved < 0:
                raise PlannerError(f'"{name}": amount already saved cannot be negative.')
            if not 1 <= deadline <= MAX_DEADLINE_MONTHS:
                raise PlannerError(f'"{name}": deadline must be between 1 and {MAX_DEADLINE_MONTHS} months.')
            if priority not in PRIORITIES:
                raise PlannerError(f'"{name}": priority must be one of {", ".join(PRIORITIES)}.')

            goals.append(Goal(name=name, target_amount=target, current_savings=min(saved, target),
                              deadline_months=deadline, priority=priority))
        return goals

    # -- planning ----------------------------------------------------------

    def plan(self, monthly_income: float, monthly_expense: float, goals: List[dict],
             actual_savings: Optional[List[float]] = None) -> Dict[str, Any]:
        """Recommend this month's savings and explain it.

        `actual_savings` lists what the user really saved in each earlier
        month. The months are replayed through the environment so the
        recommendation adapts to that behaviour; the call itself is stateless.
        """
        if not self.ready:
            return {'status': 'unavailable', 'message': self.error}

        try:
            monthly_income = float(monthly_income)
            monthly_expense = float(monthly_expense)
        except (TypeError, ValueError):
            raise PlannerError('Monthly income and expense must be numbers.')
        if not np.isfinite(monthly_income) or monthly_income <= 0:
            raise PlannerError('Monthly income must be greater than zero.')
        if not np.isfinite(monthly_expense) or monthly_expense < 0:
            raise PlannerError('Monthly expense cannot be negative.')

        parsed_goals = self._parse_goals(goals)
        available = monthly_income - monthly_expense

        if available <= 0:
            return {
                'status': 'no_surplus',
                'message': (
                    'Forecast spending is higher than income, so there is nothing left to '
                    'set aside this month. Reduce the forecast expense to get a savings plan.'
                ),
                'monthly_income': monthly_income,
                'monthly_expense': monthly_expense,
                'available_amount': available,
            }

        with self._lock:
            return self._plan(monthly_income, monthly_expense, parsed_goals, actual_savings or [])

    def _plan(self, monthly_income, monthly_expense, goals, actual_savings) -> Dict[str, Any]:
        initial_goals = copy.deepcopy(goals)
        env = SimpleGoalEnv(
            monthly_income=monthly_income, monthly_expense=monthly_expense, goals=goals,
            horizon=max(goal.deadline_months for goal in goals),
        )
        state = env.reset()
        available = monthly_income - monthly_expense

        history = []
        done = False
        for saved in actual_savings:
            if done:
                break
            try:
                saved = float(saved)
            except (TypeError, ValueError):
                raise PlannerError('Actual savings must be numbers.')
            if not np.isfinite(saved) or saved < 0:
                raise PlannerError('Actual savings cannot be negative.')

            action, recommended, _ = rl_planner.recommend_savings(self.model, env, state)
            state, _, done, info = env.step(action, actual_savings=saved)
            history.append({
                'month': env.current_step,
                'recommended': float(recommended),
                'actual': float(info['actual_savings']),
                'gap': float(recommended - info['actual_savings']),
            })

        all_completed = all(goal.remaining_amount <= 0 for goal in env.goals)
        result: Dict[str, Any] = {
            'status': 'ok',
            'month': env.current_step + 1,
            'horizon': env.horizon,
            'monthly_income': monthly_income,
            'monthly_expense': monthly_expense,
            'available_amount': available,
            'history': history,
            'finished': done,
            'all_goals_completed': all_completed,
            'initial_goals': [self._goal_dict(goal) for goal in initial_goals],
        }

        if done:
            result['goals'] = [self._goal_dict(goal) for goal in env.goals]
            return result

        _, recommended, breakdown = rl_planner.recommend_savings(self.model, env, state)
        _, influences = rl_explainer.calculate_feature_influence(self.model, state)
        allocations = allocate_savings(recommended, env.goals)
        previews = {item['goal']: item for item in rl_explainer.preview_goal_progress(allocations, env.goals)}
        urgency = {item['goal']: item for item in allocations}

        goal_rows = []
        for goal in env.goals:
            row = self._goal_dict(goal)
            preview = previews.get(goal.name)
            row.update({
                'allocation': float(preview['contribution']) if preview else 0.0,
                'allocation_share': float(urgency[goal.name]['share']) if goal.name in urgency else 0.0,
                'progress_after': float(preview['progress_after']) if preview else row['progress'],
                'remaining_after': float(preview['remaining_after']) if preview else row['remaining'],
            })
            goal_rows.append(row)

        state_values = {name: float(value) for name, value in zip(rl_explainer.FEATURE_NAMES, state)}
        drivers = [
            {
                'feature': item['feature'],
                'description': rl_explainer.FEATURE_DESCRIPTIONS.get(item['feature'], ''),
                'influence_pp': float(item['influence']) * 100,
                'state_value': state_values.get(item['feature']),
            }
            for item in influences
        ]

        result.update({
            'recommended_savings': float(recommended),
            'recommended_rate': float(breakdown['recommended_rate']),
            'policy_rate': float(breakdown['policy_rate']),
            'blend': [
                {key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                 for key, value in component.items()}
                for component in breakdown['components']
            ],
            'blended_total': float(breakdown['blended_total']),
            'was_capped': bool(breakdown['was_capped']),
            'drivers': drivers,
            'goals': goal_rows,
            'allocations': [
                {'goal': item['goal'], 'amount': float(item['amount']), 'share': float(item['share'])}
                for item in allocations
            ],
            'explanation_text': rl_explainer.format_recommendation_explanation(breakdown, influences),
            'projection': self._project(env, state),
        })
        result['counterfactual'] = self.what_if(
            available, recommended, result['allocations'],
            max(result['recommended_rate'] - 0.20, 0.0) * available,
        )
        return result

    @staticmethod
    def _goal_dict(goal) -> Dict[str, Any]:
        return {
            'name': goal.name,
            'target_amount': float(goal.target_amount),
            'current_savings': float(goal.current_savings),
            'remaining': float(goal.remaining_amount),
            'deadline_months': int(goal.deadline_months),
            'priority': goal.priority,
            'progress': float(goal.progress) * 100,
            'required_per_month': float(goal.remaining_amount / max(goal.deadline_months, 1)),
        }

    def _project(self, env, state) -> Dict[str, Any]:
        """Simulate the rest of the plan assuming every recommendation is followed."""
        sim = copy.deepcopy(env)
        start_step = sim.current_step
        deadlines = {goal.name: goal.deadline_months for goal in sim.goals}
        completed_in = {goal.name: 0 for goal in sim.goals if goal.remaining_amount <= 0}

        months = [{
            'month': start_step,
            'savings': None,
            'progress': {goal.name: float(goal.progress) * 100 for goal in sim.goals},
        }]
        done = False
        while not done:
            action, recommended, _ = rl_planner.recommend_savings(self.model, sim, state)
            state, _, done, _ = sim.step(action, actual_savings=recommended)
            step = sim.current_step - start_step
            for goal in sim.goals:
                if goal.remaining_amount <= 0 and goal.name not in completed_in:
                    completed_in[goal.name] = step
            months.append({
                'month': sim.current_step,
                'savings': float(recommended),
                'progress': {goal.name: float(goal.progress) * 100 for goal in sim.goals},
            })

        outcomes = []
        for goal in sim.goals:
            months_needed = completed_in.get(goal.name)
            deadline = deadlines[goal.name]
            if months_needed is None:
                status = 'short'
            elif months_needed <= max(deadline, 0) or months_needed == 0:
                status = 'on_track'
            else:
                status = 'late'
            outcomes.append({
                'goal': goal.name,
                'status': status,
                'months_needed': months_needed,
                'deadline_months': deadline,
                'final_progress': float(goal.progress) * 100,
                'shortfall': float(goal.remaining_amount),
            })
        return {'months': months, 'outcomes': outcomes}

    def what_if(self, available_amount: float, recommended_savings: float,
                allocations: List[dict], alternative_savings: float) -> Dict[str, Any]:
        """What saving a different (smaller) amount would cost each goal."""
        if not self.ready:
            return {'status': 'unavailable', 'message': self.error}
        try:
            available_amount = float(available_amount)
            recommended_savings = float(recommended_savings)
            alternative_savings = float(alternative_savings)
        except (TypeError, ValueError):
            raise PlannerError('What-if amounts must be numbers.')
        if available_amount <= 0:
            raise PlannerError('There is no money left after expenses to compare against.')

        recommended_rate = min(max(recommended_savings / available_amount, 0.0), 1.0)
        alternative_rate = min(max(alternative_savings / available_amount, 0.0), 1.0)
        counterfactual = rl_explainer.calculate_counterfactual(
            available_amount=available_amount,
            recommended_rate=recommended_rate,
            allocations=[{'goal': item['goal'], 'amount': float(item['amount'])} for item in allocations],
            alternative_rate=alternative_rate,
        )
        return {
            'status': 'ok',
            'recommended_rate': float(counterfactual['recommended_rate']),
            'alternative_rate': float(counterfactual['alternative_rate']),
            'recommended_savings': float(counterfactual['recommended_savings']),
            'alternative_savings': float(counterfactual['alternative_savings']),
            'savings_difference': float(counterfactual['savings_difference']),
            'goal_impacts': [
                {key: (float(value) if key != 'goal' else value) for key, value in impact.items()}
                for impact in counterfactual['goal_impacts']
            ],
        }
