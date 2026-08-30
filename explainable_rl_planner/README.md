# Explainable Adaptive RL Financial Planner

A Reinforcement Learning financial planner that adapts its monthly savings
recommendations to the user's real saving behaviour **and explains every
recommendation it makes.**

This project merges two ideas into one model:

1. **Adaptive RL** — the planner takes the amount the user *actually saved*
   each month as input and adjusts the next recommendation accordingly.
2. **Explainable AI** — for every month it tells the user *why* it
   recommended that amount and how the recommendation moves each goal
   forward.

## How it adapts

- The environment state includes 3 behaviour features: the user's previous
  actual savings, their average savings history, and the gap between the
  last recommendation and what was actually saved.
- Training simulates realistic user types (low saver, inconsistent saver,
  good saver) so the policy learns to handle non-compliant users.
- The reward penalizes recommendations the user does not follow, teaching
  the model to suggest achievable amounts.
- Each month's recommendation blends the RL policy output, the user's
  demonstrated saving capacity, and the amount required to meet goal
  deadlines.
- Overdue goals stay active with boosted urgency instead of being dropped.

## How it explains (Explainable AI)

Every month the planner prints, before you enter what you saved:

- **The blend breakdown** — the exact three parts that produced the number
  (the RL policy's suggestion, your recent saving behaviour, and the amount
  needed for deadlines), each with its weight and rupee contribution.
- **Feature attribution** — which inputs (available money, nearest deadline,
  expense burden, saving history, ...) pushed the RL policy's savings rate up
  or down, and by how many percentage points.
- **Goal progress preview** — how the recommended amount would be split
  across your goals and the progress each goal would reach.
- **Counterfactual** — what saving a smaller amount would cost your goals
  this month ("what if you saved less?").

The explanation logic lives in [`explainer.py`](explainer.py).

## Features

- Month-by-month interactive recommendations
- Adapts to the user's actual savings each month
- Priority-aware goal allocation
- Plain-language explanation of every recommendation
- Feature attribution and counterfactual reasoning
- Stops when goals are completed or the deadline is reached

## Run

Train the policy once (creates `savings_policy_model.pt`):

```bash
python train.py
```

Then run the interactive monthly planner:

```bash
python financial_planner.py
```

> Note: `savings_policy_model.pt` is git-ignored, so after cloning the repo
> run `python train.py` once before `python financial_planner.py`.

## Requirements

- Python 3
- numpy
- torch
