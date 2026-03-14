import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import SnoreCNN, CONFIGS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 10
SEED = 42


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)


def load_data(split):
    data = torch.load(os.path.join(DATA_DIR, f"{split}.pt"))
    return TensorDataset(data["features"], data["labels"].float())


def train_and_evaluate(channels):
    set_seed()

    train_set = load_data("train")
    val_set = load_data("val")
    test_set = load_data("test")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = SnoreCNN(channels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

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
            best_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Evaluate on test set
    model.load_state_dict(best_state)
    model.eval()
    test_data = torch.load(os.path.join(DATA_DIR, "test.pt"))
    features = test_data["features"].to(device)
    labels = test_data["labels"].numpy()

    with torch.no_grad():
        preds = (torch.sigmoid(model(features)) > 0.5).cpu().numpy().astype(int)

    params = sum(p.numel() for p in model.parameters())

    return {
        "params": params,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    print(f"{'Model':10s} | {'Params':>6s} | {'Epoch':>5s} | {'Val Loss':>8s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s}")
    print("-" * 75)

    for name, channels in CONFIGS.items():
        print(f"Training {name}...", end=" ", flush=True)
        result = train_and_evaluate(channels)
        print(f"\r{name:10s} | {result['params']:>6,} | {result['best_epoch']:>5d} | {result['val_loss']:>8.4f} | {result['accuracy']:>6.4f} | {result['precision']:>6.4f} | {result['recall']:>6.4f} | {result['f1']:>6.4f}")


if __name__ == "__main__":
    main()
