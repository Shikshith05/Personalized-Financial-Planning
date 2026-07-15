# Adaptive RL Financial Planner

A Reinforcement Learning financial planner that adapts its monthly savings
recommendations to the user's real saving behaviour.

Unlike a static plan, the model takes the amount the user *actually saved*
each month as input and adjusts the next recommendation accordingly.

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

## Features

- Month-by-month interactive recommendations
- Adapts to the user's actual savings each month
- Priority-aware goal allocation
- Recommendation-gap tracking
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
