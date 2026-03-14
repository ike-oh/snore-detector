import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from model import SnoreCNN

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    # Load model
    model = SnoreCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device))
    model.eval()

    # Load test data
    data = torch.load(os.path.join(DATA_DIR, "test.pt"))
    features = data["features"].to(device)
    labels = data["labels"].numpy()

    # Predict
    with torch.no_grad():
        outputs = model(features)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)

    # Metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print("=== Test Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Snoring", "Snoring"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    print("\n[Saved] confusion_matrix.png")


if __name__ == "__main__":
    main()
