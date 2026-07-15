import numpy as np
import torch


FEATURE_NAMES = [
    "Available money",
    "Urgent goal remaining",
    "Nearest deadline",
    "Highest priority",
    "Expense burden",
    "Number of active goals",

    "Previous actual savings",
    "Average savings history",
    "Recommendation gap",
]


def get_policy_rate(model, state):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    )

    with torch.no_grad():
        raw_action = model(
            state_tensor
        ).item()

    return float(
        1.0 / (
            1.0 + np.exp(-raw_action)
        )
    )


def calculate_feature_influence(
    model,
    state,
    baseline=None,
):
    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if baseline is None:
        baseline = np.array(
            [
                0.5,  # Available money
                0.5,  # Urgent goal remaining
                0.5,  # Nearest deadline
                0.5,  # Highest priority
                0.5,  # Expense burden
                0.5,  # Active goals

                0.5,  # Previous actual savings
                0.5,  # Average savings history
                0.0,  # Recommendation gap
            ],
            dtype=np.float32,
        )

    original_rate = get_policy_rate(
        model,
        state,
    )

    influences = []

    for index, feature_name in enumerate(
        FEATURE_NAMES
    ):
        modified_state = state.copy()

        modified_state[index] = (
            baseline[index]
        )

        modified_rate = get_policy_rate(
            model,
            modified_state,
        )

        influence = (
            original_rate
            - modified_rate
        )

        influences.append(
            {
                "feature": feature_name,
                "influence": influence,
                "modified_rate": modified_rate,
            }
        )

    influences.sort(
        key=lambda item: abs(
            item["influence"]
        ),
        reverse=True,
    )

    return original_rate, influences


def calculate_counterfactual(
    available_amount,
    recommended_rate,
    allocations,
    alternative_rate=0.50,
):
    recommended_savings = (
        available_amount * recommended_rate
    )

    alternative_rate = min(
        max(alternative_rate, 0.0),
        recommended_rate,
    )

    alternative_savings = (
        available_amount * alternative_rate
    )

    savings_difference = (
        recommended_savings - alternative_savings
    )

    allocation_results = []

    total_allocated = sum(
        allocation["amount"]
        for allocation in allocations
    )

    for allocation in allocations:
        if total_allocated > 0:
            goal_share = (
                allocation["amount"]
                / total_allocated
            )
        else:
            goal_share = 0.0

        recommended_goal_amount = (
            recommended_savings * goal_share
        )

        alternative_goal_amount = (
            alternative_savings * goal_share
        )

        contribution_loss = (
            recommended_goal_amount
            - alternative_goal_amount
        )

        allocation_results.append(
            {
                "goal": allocation["goal"],
                "share": goal_share,
                "recommended_amount": recommended_goal_amount,
                "alternative_amount": alternative_goal_amount,
                "contribution_loss": contribution_loss,
            }
        )

    return {
        "recommended_rate": recommended_rate,
        "alternative_rate": alternative_rate,
        "recommended_savings": recommended_savings,
        "alternative_savings": alternative_savings,
        "savings_difference": savings_difference,
        "goal_impacts": allocation_results,
    }


def print_counterfactual(counterfactual):
    print("\nWhat if you saved less?")

    print(
        f"If you saved "
        f"{counterfactual['alternative_rate'] * 100:.2f}% "
        f"instead of "
        f"{counterfactual['recommended_rate'] * 100:.2f}%, "
        f"you would save "
        f"Rs.{counterfactual['alternative_savings']:,.2f} "
        f"instead of "
        f"Rs.{counterfactual['recommended_savings']:,.2f}."
    )

    print(
        f"That would provide "
        f"Rs.{counterfactual['savings_difference']:,.2f} "
        f"less for your goals this month."
    )

    for impact in counterfactual["goal_impacts"]:
        print(
            f"- {impact['goal']} would receive "
            f"Rs.{impact['contribution_loss']:,.2f} less."
        )


def explain_decision(
    state,
    savings_rate,
    savings_amount,
    available_amount,
    allocations,
    goals,
):
    feature_names = [
        "available_money_ratio",
        "most_urgent_remaining_ratio",
        "nearest_deadline_ratio",
        "highest_priority_ratio",
        "expense_ratio",
        "active_goal_count_ratio",
    ]

    feature_values = dict(
        zip(feature_names, state)
    )

    reasons = []

    if feature_values["most_urgent_remaining_ratio"] > 0.75:
        reasons.append(
            "The most urgent goal still has a large amount remaining."
        )

    if feature_values["nearest_deadline_ratio"] < 0.5:
        reasons.append(
            "The nearest goal deadline is approaching."
        )

    if feature_values["highest_priority_ratio"] >= 0.75:
        reasons.append(
            "At least one active goal has high or critical priority."
        )

    if feature_values["expense_ratio"] > 0.6:
        reasons.append(
            "A large share of income is already used for monthly expenses."
        )

    if feature_values["active_goal_count_ratio"] >= 0.2:
        reasons.append(
            "Savings must be distributed across multiple active goals."
        )

    allocation_explanations = []

    goal_lookup = {
        goal.name: goal
        for goal in goals
    }

    for allocation in allocations:
        goal_name = allocation["goal"]
        contribution = allocation["amount"]

        goal = goal_lookup.get(goal_name)

        if goal is None:
            continue

        remaining_before = max(
            goal.target_amount
            - goal.current_savings
            + contribution,
            0.0,
        )

        remaining_after = max(
            goal.target_amount
            - goal.current_savings,
            0.0,
        )

        progress_after = (
            goal.current_savings
            / max(goal.target_amount, 1.0)
        ) * 100

        allocation_explanations.append(
            {
                "goal": goal_name,
                "contribution": contribution,
                "remaining_before": remaining_before,
                "remaining_after": remaining_after,
                "progress_after": progress_after,
                "priority": goal.priority,
                "deadline_months": goal.deadline_months,
            }
        )

    lower_rate = max(
        savings_rate - 0.20,
        0.0,
    )

    lower_savings = (
        available_amount
        * lower_rate
    )

    difference = (
        savings_amount
        - lower_savings
    )

    return {
        "savings_rate": savings_rate,
        "savings_amount": savings_amount,
        "reasons": reasons,
        "goal_impacts": allocation_explanations,
        "counterfactual": {
            "alternative_rate": lower_rate,
            "alternative_savings": lower_savings,
            "monthly_shortfall": difference,
        },
    }


def format_explanation(explanation):
    lines = []

    lines.append(
        f"The model recommends saving "
        f"{explanation['savings_rate'] * 100:.2f}% "
        f"of the available money, equal to "
        f"Rs.{explanation['savings_amount']:,.2f}."
    )

    if explanation["reasons"]:
        lines.append("\nMain reasons:")

        for reason in explanation["reasons"]:
            lines.append(
                f"- {reason}"
            )

    lines.append("\nHow this helps your goals:")

    for impact in explanation["goal_impacts"]:
        lines.append(
            f"- {impact['goal']}: "
            f"Rs.{impact['contribution']:,.2f} is allocated. "
            f"Progress becomes {impact['progress_after']:.2f}%. "
            f"Remaining amount is "
            f"Rs.{impact['remaining_after']:,.2f}. "
            f"Priority: {impact['priority']}; "
            f"{impact['deadline_months']} months remaining."
        )

    counterfactual = explanation["counterfactual"]

    lines.append(
        "\nWhat if you saved less?"
    )

    lines.append(
        f"Saving only "
        f"{counterfactual['alternative_rate'] * 100:.2f}% "
        f"would provide "
        f"Rs.{counterfactual['alternative_savings']:,.2f}, "
        f"which is "
        f"Rs.{counterfactual['monthly_shortfall']:,.2f} less "
        f"for your goals this month."
    )

    return "\n".join(lines)
