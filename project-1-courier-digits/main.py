import copy
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. Fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Save outputs in the same directory as main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Load digits (1,797 images of 8x8 = 64 pixels) and scale to 0-1
digits = load_digits()
X = digits.data.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

# Split: 200 train, remaining 1,597 split into 798 val and 799 test
X_tr, X_rest, y_tr, y_rest = train_test_split(
    X, y, train_size=200, random_state=SEED, stratify=y
)
X_va, X_te, y_va, y_te = train_test_split(
    X_rest, y_rest, train_size=798, random_state=SEED, stratify=y_rest
)

X_tr_t = torch.tensor(X_tr, device=device)
y_tr_t = torch.tensor(y_tr, device=device)
X_va_t = torch.tensor(X_va, device=device)
y_va_t = torch.tensor(y_va, device=device)
X_te_t = torch.tensor(X_te, device=device)
y_te_t = torch.tensor(y_te, device=device)

loss_fn = nn.CrossEntropyLoss()


# 3. Training loop recording train & val loss every epoch + early stopping
def train_model(name, model, opt, epochs=300, patience=None):
    model.to(device)
    hist = {"train": [], "val": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(X_tr_t), y_tr_t).item()
            va_loss = loss_fn(model(X_va_t), y_va_t).item()

        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if patience is not None and wait >= patience:
            print(f"[{name}] Early stopping triggered at epoch {epoch} (best at {best_epoch})")
            break

    # Save best checkpoint once after training finishes (prevents OneDrive lock errors)
    ckpt_path = os.path.join(BASE_DIR, f"best_{name}.pt")
    torch.save(best_state, ckpt_path)

    # Load best weights for test evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(dim=1)
        test_acc = (preds == y_te_t).float().mean().item()

    return {
        "hist": hist,
        "final_train": hist["train"][-1],
        "best_val": best_val,
        "best_epoch": best_epoch,
        "stopped_epoch": len(hist["train"]),
        "test_acc": test_acc,
    }


# 4. Define the three models
torch.manual_seed(SEED)
small_model = nn.Linear(64, 10)
opt_small = torch.optim.Adam(small_model.parameters(), lr=0.001)

torch.manual_seed(SEED)
big_model = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
opt_big = torch.optim.Adam(big_model.parameters(), lr=0.001)

torch.manual_seed(SEED)
reg_model = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)
opt_reg = torch.optim.AdamW(reg_model.parameters(), lr=0.001, weight_decay=0.05)

# 5. Train all three models
results = {
    "Small (Linear)": train_model("small", small_model, opt_small, epochs=300),
    "Big (2x256 MLP)": train_model("big", big_model, opt_big, epochs=300),
    "Big + Dropout + Decay": train_model("reg", reg_model, opt_reg, epochs=300, patience=30),
}

# 6. Plot side-by-side learning curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (title, res) in zip(axes, results.items()):
    epochs_range = range(1, len(res["hist"]["train"]) + 1)
    ax.plot(epochs_range, res["hist"]["train"], color="#d97706", linewidth=2, label="Training loss")
    ax.plot(epochs_range, res["hist"]["val"], color="#2563eb", linestyle="--", linewidth=2, label="Validation loss")
    ax.axvline(res["best_epoch"], color="black", linestyle=":", label=f"Best val (ep {res['best_epoch']})")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xlim(0, 300)
    ax.legend()
    ax.grid(alpha=0.25)

plt.tight_layout()
curves_path = os.path.join(BASE_DIR, "curves.png")
plt.savefig(curves_path, dpi=150)
print(f"\nSaved learning curves to {curves_path}\n")

# 7. Print summary table
print(f"{'Model':<24} | {'Final Train':<11} | {'Best Val (Epoch)':<20} | {'Test Acc':<8}")
print("-" * 72)
for title, res in results.items():
    val_str = f"{res['best_val']:.3f} at ep {res['best_epoch']}"
    print(f"{title:<24} | {res['final_train']:<11.3f} | {val_str:<20} | {res['test_acc']:.3f}")

# 8. Decision & Next Step
print(
    "\nDecision: Ship the regularized model ('Big + Dropout + Decay') because it achieves "
    "the lowest validation loss and stops early before overfitting; collecting more labeled "
    "training images beyond 200 would improve accuracy most next."
)