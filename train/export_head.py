"""Extract the classifier head from a fine-tuned PyTorch checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from train.preprocess import sanitize_title_for_model

PROBES = [
    "Myra Moans - Legitimate Business (07.02.2025)_480p.mp4",
    "Kimmy Saddle - Kimmy Saddle Fun Camera Two (23.06.2026)_720p.mp4",
    "SophiaParker-HiddenDesires",
    "[Tushy] Sophia Burns",
    "[BelAmi] Pillow Talk 3",
    "[Blacked] Avery Cristy",
    "[HorribleSubs] Kill la Kill - 02 [720p]",
    "[YTS.MX] The Matrix 1999 1080p BluRay",
    "Sex and the City S01E01 1080p BluRay",
    "The Big Bang Theory S10E01 720p HDTV",
]


def extract_head(model_dir: Path, source_dir: Path, output: Path) -> None:
    if not (source_dir / "pytorch_model.bin").is_file():
        source_dir = model_dir
    state = torch.load(source_dir / "pytorch_model.bin", map_location="cpu")
    weights = {
        "W1": state["classifier.1.weight"].numpy().astype(np.float32),
        "b1": state["classifier.1.bias"].numpy().astype(np.float32),
        "W2": state["classifier.4.weight"].numpy().astype(np.float32),
        "b2": state["classifier.4.bias"].numpy().astype(np.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **weights)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    providers = ["CPUExecutionProvider"]
    backbone = ort.InferenceSession(str(model_dir / "backbone.onnx"), providers=providers)
    full = ort.InferenceSession(str(source_dir / "model.onnx"), providers=providers)
    texts = [f"query: {sanitize_title_for_model(title)}" for title in PROBES]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="np",
    )
    feeds = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    hidden = backbone.run(None, feeds)[0]
    mask = feeds["attention_mask"].astype(np.float32)[:, :, None]
    pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    head_hidden = np.maximum(0, pooled @ weights["W1"].T + weights["b1"])
    logits = (head_hidden @ weights["W2"].T + weights["b2"]).squeeze(-1)
    combined = 1.0 / (1.0 + np.exp(-logits))
    expected = full.run(None, feeds)[0]
    max_diff = float(np.abs(combined - expected).max())
    print(f"Saved {output}")
    print(f"Max FP32 backbone+head vs full model difference: {max_diff:.8f}")
    if max_diff > 1e-4:
        raise SystemExit("Extracted head does not match the full ONNX classifier")
    for title, probability in zip(PROBES, combined):
        print(f"  {probability:.4f}  {sanitize_title_for_model(title)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="scratch/e5_nsfw_finetuned")
    parser.add_argument("--source-dir", default="scratch/e5_nsfw_source")
    parser.add_argument("--output", default="train/output/head_weights.npz")
    args = parser.parse_args()
    extract_head(Path(args.model_dir), Path(args.source_dir), Path(args.output))


if __name__ == "__main__":
    main()
