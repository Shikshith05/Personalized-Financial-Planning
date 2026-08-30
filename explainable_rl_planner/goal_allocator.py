from constants import PRIORITY_WEIGHTS


def goal_urgency(goal) -> float:
    priority_weight = PRIORITY_WEIGHTS.get(
        goal.priority.lower(),
        2.0,
    )

    remaining_ratio = (
        goal.remaining_amount
        / max(goal.target_amount, 1.0)
    )

    months_left = max(
        goal.deadline_months,
        1,
    )

    # Overdue goals get extra urgency instead of being dropped.
    overdue_multiplier = (
        1.5
        if goal.deadline_months <= 0
        else 1.0
    )

    return (
        priority_weight
        * remaining_ratio
        * (1 / months_left**1.5)
        * overdue_multiplier
    )


def allocate_savings(
    savings_amount: float,
    goals: list,
) -> list[dict]:
    active_goals = [
        goal
        for goal in goals
        if goal.remaining_amount > 0
    ]

    if savings_amount <= 0 or not active_goals:
        return []

    urgencies = [
        goal_urgency(goal)
        for goal in active_goals
    ]
    total_urgency = sum(urgencies)

    allocations = []

    for goal, urgency in zip(active_goals, urgencies):
        share = (
            urgency / total_urgency
            if total_urgency > 0
            else 1.0 / len(active_goals)
        )

        amount = savings_amount * share

        amount = min(
            amount,
            goal.remaining_amount,
        )

        allocations.append(
            {
                "goal": goal.name,
                "urgency": urgency,
                "share": share,
                "amount": amount,
            }
        )

    return allocations