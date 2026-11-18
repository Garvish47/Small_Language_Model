import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path("tokenizer")
DATA_DIR.mkdir(exist_ok=True)

print("Loading WikiText-2...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

print("Loading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_split(split):
    all_tokens = []

    for text in tqdm(split["text"], desc="Tokenizing"):
        if text.strip():
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)
            all_tokens.append(tokenizer.eos_token_id)

    return torch.tensor(all_tokens, dtype=torch.long)

train_tokens = tokenize_split(dataset["train"])
val_tokens = tokenize_split(dataset["validation"])

torch.save(train_tokens, DATA_DIR / "train_tokens.pt")
torch.save(val_tokens, DATA_DIR / "val_tokens.pt")

print("\nDataset preparation complete.")
print(f"Training tokens:   {len(train_tokens):,}")
print(f"Validation tokens: {len(val_tokens):,}")
print(f"Vocabulary size:   {tokenizer.vocab_size:,}")
print(f"Tokenizer:         {tokenizer.name_or_path}")
print(f"Train size:        {train_tokens.numel() * 8 / 1024**2:.2f} MB")
print(f"Validation size:   {val_tokens.numel() * 8 / 1024**2:.2f} MB")
