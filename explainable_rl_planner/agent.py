import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim=6):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    model = PolicyNetwork()

    sample_state = torch.tensor(
        [0.7, 0.95, 0.91, 0.75, 0.3, 0.2],
        dtype=torch.float32
    )

    output = model(sample_state)

    print(output)
