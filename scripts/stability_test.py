import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
from model import SnoreCNN

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 10
NUM_RUNS = 30

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data(split):
    data = torch.load(os.path.join(DATA_DIR, f"{split}.pt"))
    return TensorDataset(data["features"], data["labels"].float())


def run_once(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    train_set = load_data("train")
    val_set = load_data("val")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = SnoreCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                loss = criterion(model(features), labels)
                val_loss += loss.item() * features.size(0)
        val_loss /= len(val_set)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    test_data = torch.load(os.path.join(DATA_DIR, "test.pt"))
    features = test_data["features"].to(device)
    labels = test_data["labels"].numpy()

    with torch.no_grad():
        preds = (torch.sigmoid(model(features)) > 0.5).cpu().numpy().astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    print(f"Running {NUM_RUNS} trials on {device}\n")
    print(f"{'Run':>4s} | {'Acc':>6s} | {'Recall':>6s} | {'F1':>6s}")
    print("-" * 35)

    results = []
    for i in range(NUM_RUNS):
        r = run_once(seed=i)
        results.append(r)
        print(f"{i+1:4d} | {r['accuracy']:>6.4f} | {r['recall']:>6.4f} | {r['f1']:>6.4f}")

    accs = [r["accuracy"] for r in results]
    recs = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    print("-" * 35)
    print(f"{'Mean':>4s} | {np.mean(accs):>6.4f} | {np.mean(recs):>6.4f} | {np.mean(f1s):>6.4f}")
    print(f"{'Std':>4s} | {np.std(accs):>6.4f} | {np.std(recs):>6.4f} | {np.std(f1s):>6.4f}")
    print(f"{'Min':>4s} | {np.min(accs):>6.4f} | {np.min(recs):>6.4f} | {np.min(f1s):>6.4f}")
    print(f"{'Max':>4s} | {np.max(accs):>6.4f} | {np.max(recs):>6.4f} | {np.max(f1s):>6.4f}")


if __name__ == "__main__":
    main()
