import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def prepare_data(seed=42, scale=True):
    """Load breast cancer dataset (malignant = 1), split 60/20/20, and optionally scale on train only."""
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    # In sklearn load_breast_cancer, 0 = malignant, 1 = benign. Flip so malignant = 1.
    y = (data.target == 0).astype(np.int64)
    feature_names = list(data.feature_names)

    # 60% train, 40% temp
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=y
    )
    # Split 40% temp in half -> 20% val, 20% test
    X_va, X_te, y_va, y_te = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

    return {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32),
        "y_tr": torch.tensor(y_tr, dtype=torch.long),
        "X_va": torch.tensor(X_va, dtype=torch.float32),
        "y_va": torch.tensor(y_va, dtype=torch.long),
        "X_te": torch.tensor(X_te, dtype=torch.float32),
        "y_te": torch.tensor(y_te, dtype=torch.long),
        "X_te_raw": X_te,
        "y_te_raw": y_te,
        "feature_names": feature_names,
        "scaler": scaler,
    }


class ScreeningMLP(nn.Module):
    def __init__(self, in_features, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def diagnose(hist):
    """Diagnosis rule:
    - final training loss > 0.3 -> underfitting
    - final validation loss > 1.3 * best validation loss -> overfitting
    - else -> healthy
    """
    final_train = hist["train"][-1]
    final_val = hist["val"][-1]
    best_val = min(hist["val"])

    if final_train > 0.30:
        return "underfitting"
    elif final_val > 1.30 * best_val:
        return "overfitting"
    else:
        return "healthy"


def evaluate_split(model, X_t, y_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).numpy()
        y_true = y_t.numpy()

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    # cm = [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    total_cancers = int(tp + fn)
    cancers_found = int(tp)
    false_alarms = int(fp)
    accuracy = float((tp + tn) / len(y_true))
    recall = float(tp / total_cancers) if total_cancers > 0 else 0.0

    return {
        "accuracy": round(accuracy, 3),
        "cancers_found": cancers_found,
        "total_cancers": total_cancers,
        "false_alarms": false_alarms,
        "recall": round(recall, 3),
    }


def train(cfg, data, on_epoch=None):
    torch.manual_seed(cfg["seed"])
    model = ScreeningMLP(
        in_features=data["X_tr"].shape[1],
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    loss_fn = nn.CrossEntropyLoss()

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_va, y_va = data["X_va"], data["y_va"]

    hist = {"train": [], "val": []}
    best_val = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(X_tr), y_tr).item()
            va_loss = loss_fn(model(X_va), y_va).item()

        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if on_epoch:
            on_epoch(epoch, hist)

        if cfg["patience"] > 0 and wait >= cfg["patience"]:
            break

    model.load_state_dict(best_state)
    verdict = diagnose(hist)
    val_metrics = evaluate_split(model, X_va, y_va)

    summary = {
        "epochs_run": len(hist["train"]),
        "best_val_loss": round(best_val, 3),
        "best_epoch": best_epoch,
        "diagnosis": verdict,
        "val_metrics": val_metrics,
    }
    return model, hist, summary