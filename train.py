
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SmallLanguageModel


# ---------------- CONFIGURATION ----------------

CONFIG = {
    "batch_size": 8,
    "block_size": 128,
    "epochs": 3,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "eval_interval": 1,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path("checkpoints").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

torch.manual_seed(42)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True

print("Training device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------- DATA PREPARATION ----------------

def create_sequences(tokens, block_size):
    seq_len = block_size + 1
    num_sequences = len(tokens) // seq_len

    tokens = tokens[:num_sequences * seq_len]
    sequences = tokens.view(num_sequences, seq_len)

    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]

    return inputs, targets


train_tokens = torch.load(
    "tokenizer/train_tokens.pt",
    map_location="cpu",
    weights_only=True
)

val_tokens = torch.load(
    "tokenizer/val_tokens.pt",
    map_location="cpu",
    weights_only=True
)

X_train, Y_train = create_sequences(
    train_tokens, CONFIG["block_size"]
)

X_val, Y_val = create_sequences(
    val_tokens, CONFIG["block_size"]
)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print(f"Training sequences: {len(train_dataset):,}")
print(f"Validation sequences: {len(val_dataset):,}")
print(f"Training batches per epoch: {len(train_loader):,}")


# ---------------- MODEL ----------------

model = SmallLanguageModel(
    vocab_size=50257,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    ff_dim=512,
    max_seq_len=128,
    dropout=0.1
).to(DEVICE)

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

criterion = nn.CrossEntropyLoss()

use_amp = DEVICE.type == "cuda"
scaler = GradScaler("cuda", enabled=use_amp)


# ---------------- VALIDATION ----------------

@torch.no_grad()
def evaluate():
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in val_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

        token_count = y.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    model.train()
    return total_loss / total_tokens


# ---------------- TRAINING ----------------

history = {
    "train_loss": [],
    "val_loss": [],
    "perplexity": []
}

best_val_loss = float("inf")

for epoch in range(CONFIG["epochs"]):

    model.train()

    running_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}"
    )

    for x, y in progress:

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            logits = model(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            CONFIG["grad_clip"]
        )

        scaler.step(optimizer)
        scaler.update()

        token_count = y.numel()
        running_loss += loss.item() * token_count
        total_tokens += token_count

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / total_tokens
    val_loss = evaluate()

    # Clamp exponent to avoid numerical overflow.
    perplexity = torch.exp(
        torch.tensor(val_loss, dtype=torch.float64)
    ).item()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["perplexity"].append(perplexity)

    elapsed = time.time() - start_time

    print(f"\nEpoch {epoch + 1} completed")
    print(f"Training loss:   {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Perplexity:      {perplexity:.2f}")
    print(f"Time:            {elapsed:.1f} seconds")

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": CONFIG,
        "history": history
    }

    torch.save(
        checkpoint,
        "checkpoints/latest.pt"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            checkpoint,
            "checkpoints/best_model.pt"
        )

        print("Best checkpoint updated!")

    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=4)


# ---------------- LOSS PLOT ----------------

plt.figure(figsize=(9, 5))

epochs = range(1, len(history["train_loss"]) + 1)

plt.plot(
    epochs, history["train_loss"],
    marker="o", label="Training Loss"
)

plt.plot(
    epochs, history["val_loss"],
    marker="o", label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training and Validation Loss")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/loss_plot.png", dpi=300)
plt.show()

print("\nTraining completed.")
print("Best validation loss:", best_val_loss)
print("Results saved in outputs/")
