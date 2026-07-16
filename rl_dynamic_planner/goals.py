from dataclasses import dataclass


@dataclass
class Goal:
    name: str
    target_amount: float
    current_savings: float
    deadline_months: int
    priority: str

    @property
    def remaining_amount(self) -> float:
        return max(
            self.target_amount - self.current_savings,
            0.0,
        )

    @property
    def remaining_ratio(self) -> float:
        return (
            self.remaining_amount
            / max(self.target_amount, 1.0)
        )

    @property
    def progress(self) -> float:
        return (
            min(
                self.current_savings
                / max(self.target_amount, 1.0),
                1.0,
            )
        )