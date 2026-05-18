import torch
import torch.nn as nn
import torch.optim as optim

print("Starting training...")

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
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(num_features, num_classes).to(device)


# ------------------
# TRAIN SETUP
# ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

batch_size = 4096
epochs = 5


# ------------------
# TRAIN LOOP
# ------------------
max_val_acc = 0
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
    print(f"Val Accuracy: {acc:.4f}")
    if acc > max_val_acc:
        max_val_acc = acc
print("Training complete")

torch.save(model.state_dict(), "06_experiments/mlp_catboost_label.pt")
print("Model saved")
import json

best_acc = max_val_acc  # debes guardarlo durante entrenamiento

results = {
    "variant": base.split("/")[-1],
    "final_val_accuracy": acc,
    "best_val_accuracy": best_acc
}

with open(f"06_experiments/results_{results['variant']}.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved")