# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: snore-detector
#     language: python
#     name: snore-detector
# ---

# %% [markdown]
# ## 1. Basic Dataset Info

# %%
import os
import torchaudio
import matplotlib.pyplot as plt
import torch

DATA_DIR = os.path.join(os.getcwd(), "..", "data")

# %%
for label in ["0", "1"]:
    folder = os.path.join(DATA_DIR, label)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"Class {label} ({'Non-Snoring' if label == '0' else 'Snoring'}): {len(files)} files")

sample_path = os.path.join(DATA_DIR, "1", "1_0.wav")
waveform, sr = torchaudio.load(sample_path)
print(f"\nSample rate: {sr}")
print(f"Channels: {waveform.shape[0]}")
print(f"Length: {waveform.shape[1]} samples ({waveform.shape[1]/sr:.2f}s)")
print(f"dtype: {waveform.dtype}")

# %% [markdown]
# ## 2. Waveform Comparison

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
fig.suptitle("Waveform Comparison", fontsize=14)

for i in range(3):
    for row, label in enumerate(["1", "0"]):
        path = os.path.join(DATA_DIR, label, f"{label}_{i}.wav")
        wav, _ = torchaudio.load(path)
        axes[row][i].plot(wav[0].numpy(), linewidth=0.5)
        axes[row][i].set_title(f"{'Snoring' if label == '1' else 'Non-Snoring'} #{i}")
        axes[row][i].set_ylim(-1, 1)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Mel Spectrogram Comparison

# %%
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_mels=64, n_fft=1024, hop_length=256
)

fig, axes = plt.subplots(2, 3, figsize=(15, 6))
fig.suptitle("Mel Spectrogram Comparison", fontsize=14)

for i in range(3):
    for row, label in enumerate(["1", "0"]):
        path = os.path.join(DATA_DIR, label, f"{label}_{i}.wav")
        wav, _ = torchaudio.load(path)
        mel = mel_transform(wav)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        axes[row][i].imshow(mel_db[0].numpy(), aspect="auto", origin="lower", cmap="viridis")
        axes[row][i].set_title(f"{'Snoring' if label == '1' else 'Non-Snoring'} #{i}")
        axes[row][i].set_ylabel("Mel Bin")
        axes[row][i].set_xlabel("Frame")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Overall Sample Statistics

# %%
for label in ["0", "1"]:
    folder = os.path.join(DATA_DIR, label)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
    durations = []
    amplitudes = []
    for f in files:
        wav, file_sr = torchaudio.load(os.path.join(folder, f))
        durations.append(wav.shape[1] / file_sr)
        amplitudes.append(wav.abs().mean().item())

    name = "Non-Snoring" if label == "0" else "Snoring"
    print(f"\n[{name}]")
    print(f"  Duration range: {min(durations):.2f}~{max(durations):.2f}s")
    print(f"  Mean amplitude: {sum(amplitudes)/len(amplitudes):.4f}")
    print(f"  Amplitude range: {min(amplitudes):.4f}~{max(amplitudes):.4f}")
