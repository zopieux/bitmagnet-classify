"""Incrementally fine-tune the E5 classifier from the deployed FP32 checkpoint."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from train.dataset import extract_clean_dataset
from train.finetune import MODEL_ID, E5Classifier, TorrentDataset
from train.preprocess import sanitize_title_for_model


def read_fname(path: str) -> list[tuple[str, int]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [(row["title"], 1 if row["label"] == "nsfw" else 0) for row in rows]


def dedupe(data: list[tuple[str, int]]) -> list[tuple[str, int]]:
    seen = set()
    out = []
    for title, label in data:
        if title in seen:
            continue
        seen.add(title)
        out.append((title, label))
    return out


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(batch["label"].numpy())
    return np.array(probs), np.array(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-train", default="train/fname_train.tsv")
    parser.add_argument("--holdout", default="train/fname_holdout.tsv")
    parser.add_argument("--output", default="scratch/e5_nsfw_retrained")
    parser.add_argument("--auto-positives", type=int, default=40000)
    parser.add_argument("--auto-negatives", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--include-auto-test", action="store_true")
    parser.add_argument(
        "--init-from", default="scratch/e5_nsfw_source/pytorch_model.bin"
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--manual-only", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manual_train = read_fname(args.manual_train)
    holdout = read_fname(args.holdout)
    excluded = {title for title, _ in manual_train} | {title for title, _ in holdout}

    if args.manual_only:
        data = dedupe(manual_train)
    else:
        auto_train, auto_test = extract_clean_dataset(
            num_positives=args.auto_positives,
            num_negatives=args.auto_negatives,
        )
        auto_source = auto_train + auto_test if args.include_auto_test else auto_train
        auto = [(title, label) for title, label in auto_source if title not in excluded]
        data = dedupe(auto + manual_train)
    random.Random(args.seed).shuffle(data)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})"
    )
    print(
        f"Train rows: {len(data):,} (nsfw={sum(y for _, y in data):,}, "
        f"clean={sum(1 - y for _, y in data):,}); holdout rows: {len(holdout):,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ds = TorrentDataset(
        [sanitize_title_for_model(t) for t, _ in data],
        [y for _, y in data],
        tokenizer,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model = E5Classifier(MODEL_ID)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"Initialized from {args.init_from}")
    model = model.to(device)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch}/{args.epochs} - loss={total_loss / len(train_loader):.4f}"
        )

    model.eval()
    torch.save(model.state_dict(), output / "pytorch_model.bin")
    tokenizer.save_pretrained(output)

    holdout_ds = TorrentDataset(
        [sanitize_title_for_model(t) for t, _ in holdout],
        [y for _, y in holdout],
        tokenizer,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )
    probs, targets = evaluate(model, holdout_loader, device)
    auc = roc_auc_score(targets, probs)
    print(f"Holdout ROC-AUC: {auc:.6f}")
    for threshold in (0.5, 0.7, 0.8, 0.85):
        preds = (probs >= threshold).astype(int)
        print(f"Threshold {threshold:.2f}")
        print(
            classification_report(
                targets,
                preds,
                target_names=["clean", "nsfw"],
                digits=4,
                zero_division=0,
            )
        )

    with open(output / "holdout_scores.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["title", "label", "score", "pred@0.70"])
        for (title, label), prob in zip(holdout, probs):
            writer.writerow(
                [title, "nsfw" if label else "clean", f"{prob:.9f}", int(prob >= 0.70)]
            )

    manifest = {
        "manual_train": args.manual_train,
        "holdout": args.holdout,
        "auto_positives": args.auto_positives,
        "auto_negatives": args.auto_negatives,
        "include_auto_test": args.include_auto_test,
        "init_from": args.init_from,
        "freeze_encoder": args.freeze_encoder,
        "manual_only": args.manual_only,
        "train_rows": len(data),
        "train_nsfw": sum(y for _, y in data),
        "train_clean": sum(1 - y for _, y in data),
        "holdout_rows": len(holdout),
        "holdout_nsfw": sum(y for _, y in holdout),
        "holdout_clean": sum(1 - y for _, y in holdout),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "holdout_roc_auc": float(auc),
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Saved model and tokenizer to {output}")


if __name__ == "__main__":
    main()
