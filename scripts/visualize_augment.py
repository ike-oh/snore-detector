import os
import torchaudio
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
AUG_DIR = os.path.join(DATA_DIR, "augmented")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL = "1"
AUG_NAMES = ["noise", "pitch", "shift", "volume"]

# Pick first train file
sample_file = sorted(os.listdir(os.path.join(TRAIN_DIR, LABEL)))[0]
sample_id = os.path.splitext(sample_file)[0]

orig_path = os.path.join(TRAIN_DIR, LABEL, sample_file)
orig_wav, sr = torchaudio.load(orig_path)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_mels=64, n_fft=1024, hop_length=256
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

# --- Waveform comparison ---
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
fig.suptitle(f"Waveform: Original vs Augmented (Snoring {sample_id})", fontsize=14)

axes[0].plot(orig_wav[0].numpy(), linewidth=0.5)
axes[0].set_title("Original")
axes[0].set_ylim(-1, 1)

for i, aug in enumerate(AUG_NAMES):
    aug_path = os.path.join(AUG_DIR, LABEL, f"{sample_id}_{aug}.wav")
    wav, _ = torchaudio.load(aug_path)
    axes[i + 1].plot(wav[0].numpy(), linewidth=0.5)
    axes[i + 1].set_title(aug.capitalize())
    axes[i + 1].set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "augment_waveform.png"), dpi=150)
print("[Saved] augment_waveform.png")

# --- Mel Spectrogram comparison ---
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
fig.suptitle(f"Mel Spectrogram: Original vs Augmented (Snoring {sample_id})", fontsize=14)

mel_db = amp_to_db(mel_transform(orig_wav))
axes[0].imshow(mel_db[0].numpy(), aspect="auto", origin="lower", cmap="viridis")
axes[0].set_title("Original")

for i, aug in enumerate(AUG_NAMES):
    aug_path = os.path.join(AUG_DIR, LABEL, f"{sample_id}_{aug}.wav")
    wav, _ = torchaudio.load(aug_path)
    mel_db = amp_to_db(mel_transform(wav))
    axes[i + 1].imshow(mel_db[0].numpy(), aspect="auto", origin="lower", cmap="viridis")
    axes[i + 1].set_title(aug.capitalize())

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "augment_mel_spectrogram.png"), dpi=150)
print("[Saved] augment_mel_spectrogram.png")
