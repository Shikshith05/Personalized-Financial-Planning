def calculate_reward(
    old_progress,
    new_progress,
    savings_amount,
    total_required,
    priority_reward,
    deadline_penalty,
):
    progress_reward = float(new_progress - old_progress)

    if total_required > 0:
        schedule_reward = float(
            (savings_amount - total_required) / total_required
        )
    else:
        schedule_reward = 0.0

    reward = float(
        progress_reward
        + schedule_reward
        + priority_reward
        + deadline_penalty
    )

    return reward