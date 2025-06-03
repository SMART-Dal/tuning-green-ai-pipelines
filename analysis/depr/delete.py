

from codecarbon import EmissionsTracker



tracker = EmissionsTracker()
tracker.start()
# loop
for i in range(10):
    print(i)

    # write a dummmy code that uses gpu in reality
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 100)
            self.fc2 = nn.Linear(100, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()

emissions: float = tracker.stop()
print(emissions)