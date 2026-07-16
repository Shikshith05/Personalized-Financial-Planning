import copy
import numpy as np
import torch
from torch.distributions import Normal

from agent import PolicyNetwork
from env import SimpleGoalEnv
from explainer import (
    calculate_feature_influence,
    calculate_counterfactual,
    print_counterfactual,
)
from goals import Goal


def get_float(prompt):
    while True:
        try:
            value = (
                input(prompt)
                .replace("Rs.", "")
                .replace("₹", "")
                .replace(",", "")
                .strip()
            )
            return float(value)
        except ValueError:
            print("Please enter a valid number.")


def get_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Please enter a valid whole number.")


def get_goal_name():
    while True:
        name = input("Goal name: ").strip()

        if name:
            return name

        print("Goal name cannot be empty.")


def get_priority():
    valid = {"low", "medium", "high", "critical"}

    while True:
        priority = input(
            "Priority (low/medium/high/critical): "
        ).strip().lower()

        if priority in valid:
            return priority

        print("Enter low, medium, high, or critical.")


def calculate_returns(rewards, gamma=0.99):
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    returns = torch.tensor(
        returns,
        dtype=torch.float32,
    )

    if len(returns) > 1:
        returns = (
            returns - returns.mean()
        ) / (returns.std() + 1e-8)

    return returns


def train_model(
    env,
    model,
    episodes=1000,
    learning_rate=0.001,
    gamma=0.99,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    reward_history = []

    for episode in range(1, episodes + 1):
        state = env.reset()

        done = False
        episode_rewards = []
        log_probabilities = []

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
            )

            # Network predicts the mean raw action.
            action_mean = model(state_tensor).squeeze()

            # Exploration around the predicted action.
            distribution = Normal(
                action_mean,
                torch.tensor(0.5),
            )

            raw_action = distribution.sample()
            log_probability = distribution.log_prob(
                raw_action
            )

            next_state, reward, done, info = env.step(
                np.array(
                    [raw_action.item()],
                    dtype=np.float32,
                )
            )

            log_probabilities.append(log_probability)
            episode_rewards.append(float(reward))

            state = next_state

        discounted_returns = calculate_returns(
            episode_rewards,
            gamma,
        )

        log_probabilities = torch.stack(
            log_probabilities
        )

        loss = -(
            log_probabilities
            * discounted_returns
        ).sum()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_reward = sum(episode_rewards)
        reward_history.append(total_reward)

        if episode % 100 == 0:
            recent_average = np.mean(
                reward_history[-100:]
            )

            print(
                f"Episode {episode:4d} | "
                f"Average reward: {recent_average:.4f}"
            )

    return reward_history


def evaluate_model(env, model):
    state = env.reset()
    done = False

    total_reward = 0.0
    month = 1

    print("\n===== TRAINED MODEL RESULT =====")

    with torch.no_grad():
        while not done:
            state_before_action = state.copy()

            state_tensor = torch.tensor(
                state_before_action,
                dtype=torch.float32,
            )

            raw_action = model(state_tensor).item()

            next_state, reward, done, info = env.step(
                np.array(
                    [raw_action],
                    dtype=np.float32,
                )
            )

            print(f"\nMonth {month}")
            print(
                f"Savings rate: "
                f"{info['savings_rate'] * 100:.2f}%"
            )
            print(
                f"Savings amount: "
                f"Rs.{info['savings_amount']:.2f}"
            )
            print(f"Reward: {reward:.4f}")

            print("Goal allocations:")

            for allocation in info["allocations"]:
                print(
                    f"  {allocation['goal']}: "
                    f"Rs.{allocation['amount']:.2f}"
                )

            for goal in env.goals:

                progress = (
                    goal.current_savings
                    / goal.target_amount
                ) * 100

                print(
                    f"{goal.name}: "
                    f"{progress:.2f}% completed"
                )

            counterfactual = calculate_counterfactual(
                available_amount=(
                    env.monthly_income
                    - env.monthly_expense
                ),
                recommended_rate=info["savings_rate"],
                allocations=info["allocations"],
                alternative_rate=max(
                    info["savings_rate"] - 0.20,
                    0.0,
                ),
            )

            print_counterfactual(counterfactual)

            original_rate, influences = (
                calculate_feature_influence(
                    model,
                    state_before_action,
                )
            )

            print("\nTop factors influencing this decision:")

            for item in influences[:3]:
                direction = (
                    "increased"
                    if item["influence"] > 0
                    else "decreased"
                )

                print(
                    f"- {item['feature']} "
                    f"{direction} the savings rate by "
                    f"{abs(item['influence']) * 100:.2f} "
                    f"percentage points."
                )

            total_reward += reward
            state = next_state
            month += 1

    print(
        f"\nEvaluation total reward: "
        f"{total_reward:.4f}"
    )

    print("\n===== FINAL GOAL SUMMARY =====")

    completed_goals = 0

    for goal in env.goals:
        progress_percent = (
            goal.current_savings
            / max(goal.target_amount, 1.0)
        ) * 100

        remaining = max(
            goal.target_amount - goal.current_savings,
            0.0
        )

        completed = goal.current_savings >= goal.target_amount

        if completed:
            completed_goals += 1

        print(f"\nGoal: {goal.name or 'Unnamed Goal'}")
        print(f"Target amount: Rs.{goal.target_amount:,.2f}")
        print(f"Final savings: Rs.{goal.current_savings:,.2f}")
        print(f"Remaining amount: Rs.{remaining:,.2f}")
        print(f"Progress: {progress_percent:.2f}%")
        print(f"Completed: {'Yes' if completed else 'No'}")

    goal_success_rate = (
        completed_goals / len(env.goals) * 100
        if env.goals
        else 0.0
    )

    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"Goals completed: {completed_goals}/{len(env.goals)}")
    print(f"Goal success rate: {goal_success_rate:.2f}%")
    print(f"Total evaluation reward: {total_reward:.4f}")


def main():
    monthly_income = get_float(
        "Monthly income: Rs."
    )

    monthly_expense = get_float(
        "Monthly expense: Rs."
    )

    number_of_goals = get_int(
        "Number of goals: "
    )

    goals = []

    for index in range(number_of_goals):
        print(f"\n--- Goal {index + 1} ---")

        goals.append(
            Goal(
                name=get_goal_name(),
                target_amount=get_float(
                    "Target amount: Rs."
                ),
                current_savings=get_float(
                    "Current savings: Rs."
                ),
                deadline_months=get_int(
                    "Deadline in months: "
                ),
                priority=get_priority(),
            )
        )

    horizon = max(
        goal.deadline_months
        for goal in goals
    )

    env = SimpleGoalEnv(
        monthly_income=monthly_income,
        monthly_expense=monthly_expense,
        goals=copy.deepcopy(goals),
        horizon=horizon,
    )

    model = PolicyNetwork(
        state_dim=env.observation_dim
    )

    print("\nTraining started...")

    train_model(
        env=env,
        model=model,
        episodes=1000,
    )

    torch.save(
        model.state_dict(),
        "savings_policy_model.pt",
    )

    print(
        "\nTraining complete. "
        "Model saved as savings_policy_model.pt"
    )

    evaluate_model(env, model)


if __name__ == "__main__":
    main()
