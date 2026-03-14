import os
import random
import torchaudio
import torch
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
AUG_DIR = TRAIN_DIR
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def add_noise(waveform, snr_db=None):
    """Add gaussian noise with random SNR"""
    if snr_db is None:
        snr_db = random.uniform(10, 30)
    noise = torch.randn_like(waveform)
    signal_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    scale = (signal_power / (noise_power * snr_linear)).sqrt()
    return waveform + scale * noise


def pitch_shift(waveform, sr, n_steps=None):
    """Shift pitch by resampling (no sox dependency)"""
    if n_steps is None:
        n_steps = random.uniform(-2, 2)
    rate = 2 ** (n_steps / 12)
    resampled = torchaudio.functional.resample(waveform, sr, int(sr * rate))
    target_len = waveform.shape[1]
    if resampled.shape[1] > target_len:
        resampled = resampled[:, :target_len]
    else:
        resampled = torch.nn.functional.pad(resampled, (0, target_len - resampled.shape[1]))
    return resampled


def time_shift(waveform, shift_max=0.2):
    """Shift waveform left or right by a fraction of its length"""
    shift = int(random.uniform(-shift_max, shift_max) * waveform.shape[1])
    return torch.roll(waveform, shifts=shift, dims=1)


def volume_scale(waveform, gain_range=(0.5, 1.5)):
    """Scale volume randomly"""
    gain = random.uniform(*gain_range)
    return torch.clamp(waveform * gain, -1.0, 1.0)


AUGMENT_FNS = {
    "noise": lambda wav, sr: add_noise(wav),
    "pitch": lambda wav, sr: pitch_shift(wav, sr),
    "shift": lambda wav, sr: time_shift(wav),
    "volume": lambda wav, sr: volume_scale(wav),
}


def main():
    # Get sample rate from first file
    sample_file = os.listdir(os.path.join(TRAIN_DIR, "1"))[0]
    _, sr = torchaudio.load(os.path.join(TRAIN_DIR, "1", sample_file))

    total = 0
    for label in ["0", "1"]:
        src_dir = os.path.join(TRAIN_DIR, label)
        dst_dir = os.path.join(AUG_DIR, label)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(src_dir) if f.endswith(".wav")])
        class_name = "Non-Snoring" if label == "0" else "Snoring"
        print(f"\n[{class_name}] Augmenting {len(files)} train files...")

        for f in files:
            filepath = os.path.join(src_dir, f)
            base = os.path.splitext(f)[0]
            waveform, _ = torchaudio.load(filepath)

            for aug_name, aug_fn in AUGMENT_FNS.items():
                augmented = aug_fn(waveform, sr)
                out_path = os.path.join(dst_dir, f"{base}_{aug_name}.wav")
                torchaudio.save(out_path, augmented, sr)
                total += 1

        aug_count = len(os.listdir(dst_dir))
        print(f"  Generated: {aug_count} files")

    train_count = sum(
        len(os.listdir(os.path.join(TRAIN_DIR, l)))
        for l in ["0", "1"]
    )
    print(f"\nTrain original: {train_count}")
    print(f"Train augmented: {total}")
    print(f"Train total: {train_count + total}")


if __name__ == "__main__":
    main()
