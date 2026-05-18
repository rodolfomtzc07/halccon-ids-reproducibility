import torch
import torch.nn as nn
import torch.optim as optim
import json

print("Starting CNN training...")

# ------------------
# LOAD DATA
# ------------------
base = "02_data/processed/litnet_10pct/catboost_label"

X_train = torch.load(f"{base}/X_train.pt")
y_train = torch.load(f"{base}/y_train.pt")

X_val = torch.load(f"{base}/X_val.pt")
y_val = torch.load(f"{base}/y_val.pt")

num_features = X_train.shape[1]
num_classes = len(torch.unique(y_train))


# ------------------
# MODEL
# ------------------
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        conv_out = input_dim // 4  # por dos poolings

        self.fc = nn.Sequential(
            nn.Linear(32 * conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D(num_features, num_classes).to(device)


# ------------------
# TRAIN SETUP
# ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

batch_size = 4096
epochs = 5

max_val_acc = 0


# ------------------
# TRAIN LOOP
# ------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        xb = X_train[i:i+batch_size].to(device)
        yb = y_train[i:i+batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ------------------
    # VALIDATION
    # ------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            xb = X_val[i:i+batch_size].to(device)
            yb = y_val[i:i+batch_size].to(device)

            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total

    if acc > max_val_acc:
        max_val_acc = acc

    print(f"Val Accuracy: {acc:.4f}")

print("Training complete")

# ------------------
# SAVE
# ------------------
variant = base.split("/")[-1]

torch.save(model.state_dict(), f"06_experiments/cnn_{variant}.pt")

results = {
    "variant": variant,
    "model": "cnn_basic",
    "final_val_accuracy": acc,
    "best_val_accuracy": max_val_acc
}

with open(f"06_experiments/results_cnn_{variant}.json", "w") as f:
    json.dump(results, f, indent=4)

print("Model and results saved")