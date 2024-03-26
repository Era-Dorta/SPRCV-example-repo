import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    my_network = Net()

    for step in range(2):
        print(f"Step {step}")
        x = torch.rand((4, 3, 32, 32))
        y = my_network(x)
        print(f"input {x[0,0,0,0]}, output {y[0,0]}\n")

    print("Done!")

if __name__ == "__main__":
    main()