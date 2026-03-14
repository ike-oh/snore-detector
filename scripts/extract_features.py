import os
import torchaudio
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")

SAMPLE_RATE = 48000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TARGET_LENGTH = 48000  # 1 second

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()


def extract(split):
    """Extract Mel Spectrogram features for a given split"""
    split_dir = os.path.join(DATA_DIR, split)
    features = []
    labels = []

    for label in ["0", "1"]:
        folder = os.path.join(split_dir, label)
        files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
        print(f"  [{split}] Class {label}: {len(files)} files")

        for f in files:
            wav, sr = torchaudio.load(os.path.join(folder, f))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # Pad or trim to fixed length
            if wav.shape[1] > TARGET_LENGTH:
                wav = wav[:, :TARGET_LENGTH]
            elif wav.shape[1] < TARGET_LENGTH:
                wav = torch.nn.functional.pad(wav, (0, TARGET_LENGTH - wav.shape[1]))
            mel = amp_to_db(mel_transform(wav))  # (1, n_mels, time)
            features.append(mel)
            labels.append(int(label))

    features = torch.stack(features)  # (N, 1, n_mels, time)
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels


def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\nExtracting {split}...")
        features, labels = extract(split)
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")

        torch.save({"features": features, "labels": labels},
                    os.path.join(FEATURE_DIR, f"{split}.pt"))
        print(f"  [Saved] {split}.pt")


if __name__ == "__main__":
    main()
