import os
import shutil
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEED = 42
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

random.seed(SEED)


def main():
    for label in ["0", "1"]:
        src_dir = os.path.join(DATA_DIR, label)
        files = sorted([f for f in os.listdir(src_dir) if f.endswith(".wav")])
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLIT_RATIO["train"])
        n_val = int(n * SPLIT_RATIO["val"])

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        class_name = "Non-Snoring" if label == "0" else "Snoring"
        print(f"\n[{class_name}]")

        for split_name, split_files in splits.items():
            dst_dir = os.path.join(DATA_DIR, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)

            for f in split_files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

            print(f"  {split_name}: {len(split_files)} files")

    print("\nDone. You can now safely remove data/0/ and data/1/ if desired.")


if __name__ == "__main__":
    main()
