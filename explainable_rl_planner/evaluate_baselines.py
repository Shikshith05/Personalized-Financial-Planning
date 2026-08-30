"""Compare the proposed adaptive RL planner against rule-based baselines.

The comparison isolates the *savings-amount* decision: every strategy uses
the same priority-weighted goal allocation (via env.step), so the only thing
that differs is how much each strategy recommends saving per month. Each
simulated user has a fixed monthly saving *capacity* drawn from a behaviour
type; the same capacity sequence is applied to every strategy in a scenario,
so the comparison is apples-to-apples.

Metrics (averaged over N scenarios):
  - Goal Completion Rate : mean fraction of goals fully funded by the horizon
  - Avg. Goal Progress   : mean per-goal progress (%)
  - Cumulative Reward    : mean summed per-step env reward
  - Savings Consistency  : how closely actual savings track the recommendation
                           (100 = user could always follow the recommendation)
"""

import copy

import numpy as np
import torch

from agent import PolicyNetwork
from env import SimpleGoalEnv
from goals import Goal
import financial_planner as fp

MODEL_PATH = "savings_policy_model.pt"
NUM_SCENARIOS = 500
SEED = 12345


BEHAVIOUR_TYPES = {
    "low":          {"base": (0.15, 0.40), "noise": 0.10, "p": 0.30},
    "inconsistent": {"base": (0.30, 0.65), "noise": 0.25, "p": 0.40},
    "good":         {"base": (0.60, 0.95), "noise": 0.10, "p": 0.30},
}

PRIORITIES = ["low", "medium", "high", "critical"]


def make_scenario(rng):
    """Generate one random but realistic planning scenario."""
    monthly_income = float(rng.integers(30000, 80000))
    expense_ratio = float(rng.uniform(0.40, 0.70))
    monthly_expense = round(monthly_income * expense_ratio, 2)

    num_goals = int(rng.integers(1, 4))
    goals = []
    for i in range(num_goals):
        target = float(rng.integers(20000, 150000))
        current = round(target * float(rng.uniform(0.0, 0.10)), 2)
        deadline = int(rng.integers(4, 19))
        priority = PRIORITIES[int(rng.integers(0, 4))]
        goals.append(
            Goal(
                name=f"Goal{i + 1}",
                target_amount=target,
                current_savings=current,
                deadline_months=deadline,
                priority=priority,
            )
        )

    horizon = max(goal.deadline_months for goal in goals)

    # Fixed monthly saving-capacity sequence for this simulated user.
    keys = list(BEHAVIOUR_TYPES.keys())
    probs = [BEHAVIOUR_TYPES[k]["p"] for k in keys]
    behaviour = keys[int(rng.choice(len(keys), p=probs))]
    spec = BEHAVIOUR_TYPES[behaviour]
    base_capacity = float(rng.uniform(*spec["base"]))

    capacity_sequence = []
    for _ in range(horizon):
        noisy = base_capacity + float(rng.normal(0.0, spec["noise"]))
        capacity_sequence.append(float(np.clip(noisy, 0.05, 1.0)))

    return {
        "monthly_income": monthly_income,
        "monthly_expense": monthly_expense,
        "goals": goals,
        "horizon": horizon,
        "capacity_sequence": capacity_sequence,
    }


def rate_to_action(rate):
    rate = float(np.clip(rate, 0.001, 0.999))
    return np.array([np.log(rate / (1.0 - rate))], dtype=np.float32)


# --- Strategies: each returns a recommended savings amount for the month ----

def strat_fixed_rate(env, state, model):
    available = max(env.monthly_income - env.monthly_expense, 0.0)
    return 0.30 * available


def strat_proportional(env, state, model):
    """Needs-based: save the total monthly amount required for all goals."""
    available = max(env.monthly_income - env.monthly_expense, 0.0)
    required = 0.0
    for goal in env.goals:
        remaining = max(goal.target_amount - goal.current_savings, 0.0)
        if remaining > 0:
            required += remaining / max(goal.deadline_months, 1)
    return min(required, available)


def strat_deadline_greedy(env, state, model):
    """Save only the nearest-deadline unmet goal's monthly requirement."""
    available = max(env.monthly_income - env.monthly_expense, 0.0)
    active = [g for g in env.goals if g.remaining_amount > 0]
    if not active:
        return 0.0
    nearest = min(active, key=lambda g: g.deadline_months)
    required = nearest.remaining_amount / max(nearest.deadline_months, 1)
    return min(required, available)


def strat_rl(env, state, model):
    _, recommended, _ = fp.recommend_savings(model, env, state)
    return recommended


STRATEGIES = {
    "Fixed-rate saving (30%)": strat_fixed_rate,
    "Proportional (needs-based)": strat_proportional,
    "Deadline-greedy": strat_deadline_greedy,
    "RL policy (proposed)": strat_rl,
}


def run_episode(strategy_fn, scenario, model):
    env = SimpleGoalEnv(
        monthly_income=scenario["monthly_income"],
        monthly_expense=scenario["monthly_expense"],
        goals=copy.deepcopy(scenario["goals"]),
        horizon=scenario["horizon"],
    )
    state = env.reset()
    capacity_sequence = scenario["capacity_sequence"]

    total_reward = 0.0
    gap_ratios = []
    month = 0
    done = False

    while not done:
        available = max(env.monthly_income - env.monthly_expense, 0.0)
        recommended = float(strategy_fn(env, state, model))
        recommended = float(np.clip(recommended, 0.0, available))

        capacity = capacity_sequence[min(month, len(capacity_sequence) - 1)]
        user_max = capacity * available
        actual = min(recommended, user_max)

        action = rate_to_action(recommended / available if available > 0 else 0.0)
        state, reward, done, info = env.step(action, actual_savings=actual)

        total_reward += reward
        if available > 0:
            gap_ratios.append(abs(recommended - actual) / available)
        month += 1

    goals = env.goals
    completed = sum(1 for g in goals if g.current_savings >= g.target_amount)
    completion_rate = completed / len(goals)
    avg_progress = np.mean(
        [min(g.current_savings / max(g.target_amount, 1.0), 1.0) for g in goals]
    )
    consistency = 1.0 - (np.mean(gap_ratios) if gap_ratios else 0.0)

    return {
        "completion_rate": completion_rate,
        "avg_progress": float(avg_progress),
        "reward": total_reward,
        "consistency": float(consistency),
    }


def main():
    np.random.seed(SEED)
    master_rng = np.random.default_rng(SEED)

    model = PolicyNetwork(state_dim=9)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Build all scenarios once so every strategy faces the identical set.
    scenarios = [make_scenario(master_rng) for _ in range(NUM_SCENARIOS)]

    results = {name: [] for name in STRATEGIES}
    for name, fn in STRATEGIES.items():
        for scenario in scenarios:
            results[name].append(run_episode(fn, scenario, model))

    print(f"\nEvaluated over {NUM_SCENARIOS} simulated scenarios (seed={SEED})\n")
    header = (
        f"{'Method':<28}"
        f"{'GoalCompletion%':>17}"
        f"{'AvgProgress%':>14}"
        f"{'CumReward':>12}"
        f"{'Consistency%':>14}"
    )
    print(header)
    print("-" * len(header))

    for name in STRATEGIES:
        rows = results[name]
        gc = np.mean([r["completion_rate"] for r in rows]) * 100
        ap = np.mean([r["avg_progress"] for r in rows]) * 100
        cr = np.mean([r["reward"] for r in rows])
        cs = np.mean([r["consistency"] for r in rows]) * 100
        print(
            f"{name:<28}{gc:>17.2f}{ap:>14.2f}{cr:>12.2f}{cs:>14.2f}"
        )


if __name__ == "__main__":
    main()
