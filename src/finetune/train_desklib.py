"""Fine-tune desklib's top layers to predict the OpAI-Bench edit version (0 = human ... 8 = last AI edit)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from baseline.desklib import MODEL_ID, DesklibAIDetectionModel
from baseline.eval_desklib import auroc
from data.opai_bench import version_loader

N_CLASSES = 9
TRAIN_LAYERS = 4
TRAIN_PER_CLASS, DEV_PER_CLASS = 2000, 300
MAX_LEN, BATCH_SIZE = 512, 8
LR_LAYERS, LR_HEAD = 2e-5, 1e-3
ROOT = Path(__file__).parent.parent.parent
NAME = f"desklib-top{TRAIN_LAYERS}_opai-bench_version{N_CLASSES}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    model = DesklibAIDetectionModel.from_pretrained(MODEL_ID)
    model.classifier = nn.Linear(model.config.hidden_size, N_CLASSES)
    model.requires_grad_(False)
    # whole model won't fit into my GPU
    model.model.encoder.layer[-TRAIN_LAYERS:].requires_grad_(True)
    model.classifier.requires_grad_(True)
    return model.to(DEVICE)


def forward(model, tokenizer, texts):
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        return model(enc["input_ids"], enc["attention_mask"]).float()


def train(model, tokenizer):
    optimizer = torch.optim.AdamW([
        {"params": model.model.encoder.layer[-TRAIN_LAYERS:].parameters(), "lr": LR_LAYERS},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ])
    model.train()
    for step, (_, texts, labels) in enumerate(version_loader("train", TRAIN_PER_CLASS, BATCH_SIZE, shuffle=True), 1):
        loss = nn.functional.cross_entropy(forward(model, tokenizer, texts), labels.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.3f}")


@torch.no_grad()
def evaluate(model, tokenizer):
    model.eval()
    y, p = [], []
    for _, texts, labels in version_loader("dev", DEV_PER_CLASS, BATCH_SIZE):
        p.append(torch.softmax(forward(model, tokenizer, texts), dim=-1).cpu())
        y += labels.tolist()
    y, p = np.array(y), torch.cat(p).numpy()
    pred = p.argmax(axis=1)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)  # rows = true, cols = predicted
    np.add.at(cm, (y, pred), 1)
    f1 = 2 * cm.diagonal() / (cm.sum(axis=0) + cm.sum(axis=1))
    return {
        "model": f"{MODEL_ID}, top {TRAIN_LAYERS} layers + Linear(1024, {N_CLASSES}) fine-tuned",
        "task": f"predict edit version v0 (human) ... v8; train {TRAIN_PER_CLASS}/class, dev {DEV_PER_CLASS}/class",
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "f1_per_version": f1.round(4).tolist(),
        "mean_abs_version_error": float(np.abs(pred - y).mean()),
        "auroc_human_vs_any_ai": float(auroc((y > 0).astype(int), 1 - p[:, 0])),
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion(m, out):
    cm = np.array(m["confusion_matrix"])
    rows = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, f"{rows[i, j]:.0%}", ha="center", va="center", fontsize=9,
                color="white" if rows[i, j] > 0.5 else "black")
    ticks = [f"v{k}" for k in range(N_CLASSES)]
    ax.set_xticks(range(N_CLASSES), ticks)
    ax.set_yticks(range(N_CLASSES), ticks)
    ax.set_xlabel("Predicted version")
    ax.set_ylabel("True version (v0 = human)")
    ax.set_title(f"Confusion matrix (row %), accuracy = {m['accuracy']:.3f}, macro F1 = {m['macro_f1']:.3f}\n"
                 f"desklib top-{TRAIN_LAYERS} fine-tuned on OpAI-Bench edit version, dev split", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = build_model()
    train(model, tokenizer)
    metrics = evaluate(model, tokenizer)

    for d in ("data", "graphs", "checkpoints"):
        (ROOT / d).mkdir(exist_ok=True)
    trained = {k: v for k, v in model.named_parameters() if v.requires_grad}
    torch.save(trained, ROOT / "checkpoints" / f"{NAME}.pt")
    (ROOT / "data" / f"{NAME}_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(metrics, ROOT / "graphs" / f"{NAME}_confusion_matrix.png")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
