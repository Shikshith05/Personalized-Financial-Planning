from goals import Goal
from env import SimpleGoalEnv
import numpy as np

VALID_PRIORITIES = {"low", "medium", "high", "critical"}


def read_positive_float(message: str) -> float:
    while True:
        try:
            value = float(input(message))

            if value > 0:
                return value

            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")


def read_positive_int(message: str) -> int:
    while True:
        try:
            value = int(input(message))

            if value > 0:
                return value

            print("Please enter a positive whole number.")
        except ValueError:
            print("Please enter a valid whole number.")


def read_priority() -> str:
    while True:
        priority = input(
            "Priority (low/medium/high/critical): "
        ).strip().lower()

        if priority in VALID_PRIORITIES:
            return priority

        print(
            "Invalid priority. Enter low, medium, high, or critical."
        )


def collect_goals() -> list[Goal]:
    number_of_goals = read_positive_int(
        "\nHow many goals do you want to enter? "
    )

    goals: list[Goal] = []

    for index in range(number_of_goals):
        print(f"\n--- Goal {index + 1} ---")

        name = input("Goal name: ").strip()

        while not name:
            print("Goal name cannot be empty.")
            name = input("Goal name: ").strip()

        target_amount = read_positive_float(
            "Target amount: Rs."
        )

        current_savings = read_positive_float(
            "Current savings: Rs."
        )

        while current_savings > target_amount:
            print(
                "Current savings cannot exceed the target. "
                "It will be limited to the target amount."
            )
            current_savings = target_amount

        deadline_months = read_positive_int(
            "Deadline in months: "
        )

        priority = read_priority()

        goal = Goal(
            name=name,
            target_amount=target_amount,
            current_savings=current_savings,
            deadline_months=deadline_months,
            priority=priority,
        )

        goals.append(goal)

    return goals


def display_goals(goals: list[Goal]) -> None:
    print("\n" + "=" * 65)
    print("GOALS ENTERED")
    print("=" * 65)

    for index, goal in enumerate(goals, start=1):
        print(f"\nGoal {index}: {goal.name}")
        print(f"Target:          Rs.{goal.target_amount:,.2f}")
        print(f"Already saved:   Rs.{goal.current_savings:,.2f}")
        print(f"Remaining:       Rs.{goal.remaining_amount:,.2f}")
        print(f"Deadline:        {goal.deadline_months} months")
        print(f"Priority:        {goal.priority.title()}")
        print(f"Progress:        {goal.progress * 100:.2f}%")


def main() -> None:
    print("=" * 65)
    print("SIMPLE MULTI-GOAL SAVINGS RL SYSTEM")
    print("=" * 65)

    monthly_income = read_positive_float(
        "\nEnter monthly income: Rs."
    )

    monthly_expense = read_positive_float(
        "Enter expected monthly expense: Rs."
    )

    while monthly_expense > monthly_income:
        print(
            "\nWarning: expenses are greater than income."
        )

        choice = input(
            "Do you still want to continue? (yes/no): "
        ).strip().lower()

        if choice == "yes":
            break

        monthly_expense = read_positive_float(
            "Enter expected monthly expense again: Rs."
        )

    goals = collect_goals()

    display_goals(goals)

    available_amount = max(
        monthly_income - monthly_expense,
        0.0,
    )

    print("\n" + "=" * 65)
    print("MONTHLY FINANCIAL SUMMARY")
    print("=" * 65)
    print(f"Monthly income:   Rs.{monthly_income:,.2f}")
    print(f"Monthly expense:  Rs.{monthly_expense:,.2f}")
    print(f"Available amount: Rs.{available_amount:,.2f}")

    horizon = max(
        goal.deadline_months
        for goal in goals
    )

    env = SimpleGoalEnv(
        monthly_income=monthly_income,
        monthly_expense=monthly_expense,
        goals=goals,
        horizon=horizon,
    )

    observation, info = env.reset()

    print("\nEnvironment created successfully.")
    print(f"Observation size: {len(observation)}")
    print(f"Training horizon: {horizon} months")
    print(f"Initial observation:\n{observation}")

    action = np.array([1.0], dtype=np.float32)

    obs, step_reward, done, info = env.step(action)

    print("\n===== STEP OUTPUT =====")
    print("Observation:", obs)
    print("Reward:", step_reward)
    print("Done:", done)
    print("Info:", info)


if __name__ == "__main__":
    main()