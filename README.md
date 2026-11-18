# Small Language Model from Scratch

A compact decoder-only Transformer language model implemented in PyTorch and trained from random initialization on WikiText-2 using **causal language modeling (next-token prediction)**.

> This is an educational experiment in understanding the mechanics of a small language model—not a pretrained GPT model, and not an attempt to build a production-ready text generator.

## What I built

- A decoder-only Transformer with causal self-attention.
- Learned token and positional embeddings.
- Multi-head attention, residual connections, LayerNorm, and GELU feed-forward blocks.
- A tied token-embedding / language-model output matrix.
- A training pipeline with AdamW, FP16 mixed precision, gradient clipping, validation, checkpointing, and loss tracking.
- Autoregressive text generation using temperature scaling and top-k sampling.

The model weights were initialized and trained from scratch. I used the existing GPT-2 tokenizer, so **the tokenizer is pretrained; the language model is not**.

## Architecture

| Component | Configuration |
|---|---:|
| Architecture | Decoder-only Transformer |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Feed-forward dimension | 512 |
| Context length | 128 tokens |
| Dropout | 0.1 |
| Vocabulary size | 50,257 |
| Parameters | **6,846,080 (~6.85M)** |
| Attention mask | Causal |
| Output weights | Tied with token embeddings |

Each input position is trained to predict the next token. For a token sequence

`[t0, t1, t2, ..., t128]`

the model receives `[t0, ..., t127]` and learns to predict `[t1, ..., t128]`.

## Dataset and tokenization

- **Dataset:** [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-2-raw-v1`.
- **Tokenizer:** GPT-2 tokenizer from Hugging Face.
- **Vocabulary:** 50,257 tokens.
- **Training split:** 2,415,651 tokens.
- **Validation split:** 249,750 tokens.
- **Preprocessing:** Empty/whitespace-only lines were skipped, each non-empty line was tokenized, and an EOS token was appended. The resulting token stream was divided into fixed-length next-token prediction examples.

The corpus is intentionally small enough for a compact training experiment. It is not representative of the data scale needed for a capable general-purpose language model.

## Training setup

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU |
| Objective | Causal language modeling |
| Batch size | 8 |
| Sequence length | 128 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | FP16 automatic mixed precision |
| Approx. time | 50 seconds / epoch |

## Results

Training and validation loss both decreased over the three-epoch run. Validation perplexity also improved, indicating that the model became better at predicting held-out next tokens.

![Training and validation loss across three epochs](assets/loss_curve.png)

| Epoch | Training loss | Validation loss | Validation perplexity |
|---:|---:|---:|---:|
| 1 | 6.4139 | 5.9413 | 380.44 |
| 2 | 5.5569 | 5.6408 | 281.68 |
| 3 | 5.1845 | 5.5025 | 245.29 |

Perplexity is calculated as `exp(validation cross-entropy loss)`.

## Sample generation

The model generates text autoregressively from a prompt. These are representative samples from the three-epoch checkpoint; they are included as generated, not edited examples.

**Prompt:** `The history of science`

```text
The history of science fiction . The music has been " , based on the band
of the song 's original band 's musical , and song " is a song to a song
the song .
```

**Prompt:** `Artificial intelligence is`

```text
Artificial intelligence is the first of the game of the game .
= = = Death is = = = =
In the United States , the game was given to the ball of the game ...
```

The outputs show that the model learned some surface-level patterns from WikiText-2, including punctuation and WikiText-style headings. They also show substantial repetition, topic drift, and weak sentence coherence.

## What I learned

- How token IDs and shifted targets are constructed for next-token prediction.
- How causal self-attention prevents a position from attending to future tokens.
- How Transformer blocks combine attention, residual paths, normalization, and feed-forward layers.
- How to implement a training/validation loop and monitor loss and perplexity.
- Why a falling loss does not automatically imply coherent text generation.

## Limitations and next experiments

This run is deliberately small: 6.85M parameters, a small corpus, context length 128, and only three epochs. The generated text is therefore expected to be weak.

Possible next experiments:

- Train for more epochs and compare validation loss.
- Add a learning-rate scheduler and warm-up.
- Compare different context lengths and model sizes.
- Track gradient norms and inspect attention behaviour.
- Compare sampling strategies and repetition penalties.
- Evaluate on a larger or domain-specific corpus.

## Run it locally

### 1. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets matplotlib tqdm numpy
```

Install a PyTorch build compatible with your CUDA driver/GPU if you want GPU acceleration.

### 2. Prepare the dataset

```bash
python data.py
```

This downloads WikiText-2 and the GPT-2 tokenizer, then saves token tensors under `tokenizer/`.

### 3. Train the model

```bash
python train.py
```

Training checkpoints are written to `checkpoints/`, while the training history and loss plot are written to `outputs/`.

### 4. Generate text

```bash
python evaluate.py
```

Generated samples are saved to `outputs/generated_samples.txt`.

## Repository structure

```text
.
├── model.py
├── data.py
├── train.py
├── evaluate.py
├── tokenizer/
├── checkpoints/
├── outputs/
│   ├── history.json
│   ├── loss_plot.png
│   └── generated_samples.txt
├── assets/
│   └── loss_curve.png
└── README.md
```

> Large generated artifacts such as model checkpoints and downloaded token tensors may be kept out of Git and recreated by running the scripts. If you choose to publish checkpoints, consider using Git LFS or a model-artifact hosting service.

## Acknowledgements

- [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
