import os
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")

def main():
    # Compute mean/std from train only
    train_data = torch.load(os.path.join(FEATURE_DIR, "train.pt"))
    train_features = train_data["features"]

    mean = train_features.mean()
    std = train_features.std()
    print(f"Train mean: {mean:.4f}")
    print(f"Train std:  {std:.4f}")

    # Save stats for inference
    torch.save({"mean": mean, "std": std},
               os.path.join(FEATURE_DIR, "norm_stats.pt"))
    print("[Saved] norm_stats.pt")

    # Normalize all splits
    for split in ["train", "val", "test"]:
        data = torch.load(os.path.join(FEATURE_DIR, f"{split}.pt"))
        features = (data["features"] - mean) / std

        torch.save({"features": features, "labels": data["labels"]},
                   os.path.join(FEATURE_DIR, f"{split}.pt"))
        print(f"[Normalized] {split}.pt")


if __name__ == "__main__":
    main()
