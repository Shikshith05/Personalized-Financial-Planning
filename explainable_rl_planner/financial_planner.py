import numpy as np
import torch

from agent import PolicyNetwork
from env import SimpleGoalEnv
from goals import Goal
from goal_allocator import allocate_savings
from explainer import (
    calculate_feature_influence,
    calculate_counterfactual,
    build_recommendation_breakdown,
    format_recommendation_explanation,
    preview_goal_progress,
    format_goal_progress_preview,
)
from train import (
    get_float,
    get_goal_name,
    get_int,
    get_priority,
)

MODEL_PATH = "savings_policy_model.pt"


def load_model(model_path: str, state_dim: int = 9) -> PolicyNetwork:
    model = PolicyNetwork(state_dim=state_dim)

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_recommended_action(
    model: PolicyNetwork,
    state: np.ndarray,
) -> np.ndarray:
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        raw_action = model(state_tensor)

    return np.array(
        [raw_action.item()],
        dtype=np.float32,
    )


def get_actual_savings():
    while True:
        try:
            actual_savings = float(
                input(
                    "Enter actual amount saved "
                    "this month: Rs."
                )
            )

            if actual_savings < 0:
                print("Savings cannot be negative.")
                continue

            return actual_savings

        except ValueError:
            print("Please enter a valid number.")


def build_environment():
    print("Enter details to create your plan\n")

    monthly_income = get_float("Monthly income: Rs.")
    monthly_expense = get_float("Monthly expense: Rs.")

    number_of_goals = get_int("Number of goals: ")

    goals = []

    for index in range(number_of_goals):
        print(f"\n--- Goal {index + 1} ---")

        goals.append(
            Goal(
                name=get_goal_name(),
                target_amount=get_float("Target amount: Rs."),
                current_savings=get_float("Current savings: Rs."),
                deadline_months=get_int("Deadline in months: "),
                priority=get_priority(),
            )
        )

    horizon = max(
        goal.deadline_months
        for goal in goals
    )

    return SimpleGoalEnv(
        monthly_income=monthly_income,
        monthly_expense=monthly_expense,
        goals=goals,
        horizon=horizon,
    )


def recommend_savings(model, env, state):
    """Return (action, recommended_savings, breakdown).

    The recommendation blends three parts so the number is both
    goal-aware and realistic. The returned ``breakdown`` records the
    value and weight of every part so the choice can be explained.
    """
    action = get_recommended_action(model, state)

    base_rate = float(
        1 / (1 + np.exp(-action[0]))
    )

    available_amount = max(
        env.monthly_income - env.monthly_expense,
        0.0,
    )

    base_recommendation = (
        base_rate * available_amount
    )

    # Amount still required for all incomplete goals
    required_monthly_savings = 0.0

    for goal in env.goals:
        remaining_amount = max(
            goal.target_amount - goal.current_savings,
            0.0,
        )

        if remaining_amount > 0:
            months_left = max(
                goal.deadline_months,
                1,
            )

            required_monthly_savings += (
                remaining_amount / months_left
            )

    required_monthly_savings = min(
        required_monthly_savings,
        available_amount,
    )

    # Learn the user's realistic saving capacity
    if env.monthly_savings_history:
        recent_savings = (
            env.monthly_savings_history[-3:]
        )

        user_average_savings = (
            sum(recent_savings)
            / len(recent_savings)
        )

        # Dynamic recommendation:
        # 20% RL + 50% user behaviour + 30% deadline requirement
        weights = {
            "policy": 0.20,
            "behaviour": 0.50,
            "deadline": 0.30,
        }

        recommended_savings = (
            weights["policy"] * base_recommendation
            + weights["behaviour"] * user_average_savings
            + weights["deadline"] * required_monthly_savings
        )

    else:
        # First month has no user history
        weights = {
            "policy": 0.40,
            "deadline": 0.60,
        }

        user_average_savings = None

        recommended_savings = (
            weights["policy"] * base_recommendation
            + weights["deadline"] * required_monthly_savings
        )

    recommended_savings = float(
        np.clip(
            recommended_savings,
            0.0,
            available_amount,
        )
    )

    breakdown = build_recommendation_breakdown(
        available_amount=available_amount,
        recommended_savings=recommended_savings,
        policy_rate=base_rate,
        policy_amount=base_recommendation,
        required_monthly_savings=required_monthly_savings,
        user_average_savings=user_average_savings,
        weights=weights,
    )

    # Convert the blended recommendation back into an RL action
    # so env.step() reconstructs the exact amount shown to the
    # user.
    if available_amount > 0:
        adjusted_rate = (
            recommended_savings
            / available_amount
        )
    else:
        adjusted_rate = 0.0

    # Avoid log(0) and division by zero
    adjusted_rate = float(
        np.clip(
            adjusted_rate,
            0.001,
            0.999,
        )
    )

    adjusted_raw_action = np.log(
        adjusted_rate
        / (1.0 - adjusted_rate)
    )

    adjusted_action = np.array(
        [adjusted_raw_action],
        dtype=np.float32,
    )

    return adjusted_action, recommended_savings, breakdown


def explain_recommendation(model, env, state, recommended_savings, breakdown):
    """Print, in plain language, why this month's amount was chosen."""
    available_amount = breakdown["available_amount"]

    # Feature attribution: which inputs drove the RL policy part.
    _, policy_influences = calculate_feature_influence(
        model,
        state,
    )

    print("\n----- Why this recommendation? -----")
    print(
        format_recommendation_explanation(
            breakdown,
            policy_influences,
        )
    )

    # How the recommended amount would be split across goals, and the
    # progress each goal would reach if the user follows the plan.
    recommended_allocations = allocate_savings(
        recommended_savings,
        env.goals,
    )

    previews = preview_goal_progress(
        recommended_allocations,
        env.goals,
    )

    print(format_goal_progress_preview(previews))

    # Counterfactual: what a smaller saving would cost the goals.
    if recommended_allocations and available_amount > 0:
        recommended_rate = recommended_savings / available_amount

        counterfactual = calculate_counterfactual(
            available_amount=available_amount,
            recommended_rate=recommended_rate,
            allocations=recommended_allocations,
            alternative_rate=max(recommended_rate - 0.20, 0.0),
        )

        print("\nWhat if you saved less?")
        print(
            f"Saving {counterfactual['alternative_rate'] * 100:.1f}% "
            f"instead of {counterfactual['recommended_rate'] * 100:.1f}% "
            f"would set aside Rs.{counterfactual['alternative_savings']:,.2f} "
            f"rather than Rs.{counterfactual['recommended_savings']:,.2f} "
            f"- Rs.{counterfactual['savings_difference']:,.2f} less "
            f"toward your goals this month."
        )

        for impact in counterfactual["goal_impacts"]:
            if impact["contribution_loss"] > 0:
                print(
                    f"- {impact['goal']} would receive "
                    f"Rs.{impact['contribution_loss']:,.2f} less."
                )

    print("------------------------------------")


def main():
    env = build_environment()

    model = load_model(
        MODEL_PATH,
        state_dim=env.observation_dim,
    )

    state = env.reset()
    done = False

    while not done:
        month_number = env.current_step + 1

        action, recommended_savings, breakdown = recommend_savings(
            model,
            env,
            state,
        )

        print(f"\n==================== Month {month_number} ====================")
        print(
            f"Recommended savings: "
            f"Rs.{recommended_savings:,.2f}"
        )

        # Explainable AI: justify the recommendation before the user acts.
        explain_recommendation(
            model,
            env,
            state,
            recommended_savings,
            breakdown,
        )

        actual_savings = get_actual_savings()

        state, reward, done, info = env.step(
            action,
            actual_savings=actual_savings,
        )

        print(
            f"\nRecorded savings: "
            f"Rs.{info['actual_savings']:,.2f}"
        )

        print(
            f"Recommendation gap: "
            f"Rs.{info['savings_gap']:,.2f} "
            f"(recommended minus actual)"
        )

        print("Goal progress after this month:")

        for goal in env.goals:
            progress = min(
                goal.current_savings
                / max(goal.target_amount, 1.0),
                1.0,
            )

            print(
                f"- {goal.name}: "
                f"{progress * 100:.2f}% "
                f"(Rs.{goal.current_savings:,.2f} / "
                f"Rs.{goal.target_amount:,.2f})"
            )

        all_goals_completed = all(
            goal.current_savings
            >= goal.target_amount
            for goal in env.goals
        )

        if all_goals_completed:
            print(
                "\nAll financial goals have been completed."
            )
            break

    print("\nPlanning period completed.")

    for goal in env.goals:
        print(
            f"{goal.name}: "
            f"Rs.{goal.current_savings:,.2f} "
            f"/ Rs.{goal.target_amount:,.2f}"
        )


if __name__ == "__main__":
    main()
