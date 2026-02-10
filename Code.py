import torch
import torch.nn as nn
import torch.optim as optim
import pickle


class DogClassifier(nn.Module):
    def __init__(self, input_size):
        super(DogClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def load_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def main():
    data = load_data("data/data_dog_nondog.pickle")

    X_train = torch.tensor(data["X_train"], dtype=torch.float32).T
    Y_train = torch.tensor(data["Y_train"], dtype=torch.float32).T

    X_test = torch.tensor(data["X_test"], dtype=torch.float32).T
    Y_test = torch.tensor(data["Y_test"], dtype=torch.float32).T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DogClassifier(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, Y_train = X_train.to(device), Y_train.to(device)

    epochs = 1000
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    with torch.no_grad():
        train_preds = (model(X_train) > 0.5).float()
        train_acc = (train_preds == Y_train).float().mean()

        X_test, Y_test = X_test.to(device), Y_test.to(device)
        test_preds = (model(X_test) > 0.5).float()
        test_acc = (test_preds == Y_test).float().mean()

    print(f"Train Accuracy: {train_acc.item() * 100:.2f}%")
    print(f"Test Accuracy: {test_acc.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
