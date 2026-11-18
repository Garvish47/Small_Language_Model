import torch
from transformers import AutoTokenizer
from model import SmallLanguageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Reconstruct model
checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint["config"]

model = SmallLanguageModel(
    vocab_size=50257,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    ff_dim=512,
    max_seq_len=128,
    dropout=0.1
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
):
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    for _ in range(max_new_tokens):

        # Keep only the latest context window
        context = input_ids[:, -128:]

        logits = model(context)

        # Select next-token logits
        logits = logits[:, -1, :]

        # Temperature scaling
        logits = logits / temperature

        # Top-k sampling
        values, _ = torch.topk(logits, top_k)
        threshold = values[:, -1].unsqueeze(-1)

        logits[logits < threshold] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities, num_samples=1
        )

        input_ids = torch.cat(
            [input_ids, next_token], dim=1
        )

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True
    )


prompts = [
    "The history of science",
    "Artificial intelligence is",
    "In the modern world,"
]

with open("outputs/generated_samples.txt", "w") as f:

    for prompt in prompts:

        print("\n" + "=" * 60)
        print("PROMPT:", prompt)

        generated = generate(prompt)

        print(generated)

        f.write("=" * 60 + "\n")
        f.write(generated + "\n\n")

print("\nGeneration completed.")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Saved outputs/generated_samples.txt")
