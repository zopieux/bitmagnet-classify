"""Export the fine-tuned E5 NSFW classifier and its encoder to FP32 ONNX.

Creates model.onnx (full classifier) and backbone.onnx (encoder only) taking
input_ids and attention_mask.
"""

import os
import shutil

import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoTokenizer

from train.finetune import MODEL_ID, OUTPUT_DIR, E5Classifier

MODEL_DIR = "scratch/e5_nsfw_finetuned"
ONNX_FP32 = os.path.join(OUTPUT_DIR, "model.onnx")
ONNX_BACKBONE = os.path.join(MODEL_DIR, "backbone.onnx")


class BackboneWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]


class FullClassifierWrapper(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # E5 mean pooling
        token_embeddings = outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        logits = self.classifier(pooled).squeeze(-1)
        return torch.sigmoid(logits)


def export_fp32():
    print("1. Loading fine-tuned PyTorch checkpoint...")
    model = E5Classifier(MODEL_ID)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "pytorch_model.bin"), map_location="cpu")
    )
    model.eval()

    wrapper = FullClassifierWrapper(model.encoder, model.classifier)
    wrapper.eval()

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    dummy = tokenizer("query: test title", return_tensors="pt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"2. Exporting to {ONNX_FP32}...")
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        ONNX_FP32,
        input_names=["input_ids", "attention_mask"],
        output_names=["nsfw_prob"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "nsfw_prob": {0: "batch_size"},
        },
        opset_version=17,
        external_data=True,
    )

    print(f"3. Exporting encoder to {ONNX_BACKBONE}...")
    backbone = BackboneWrapper(model.encoder)
    backbone.eval()
    torch.onnx.export(
        backbone,
        (dummy["input_ids"], dummy["attention_mask"]),
        ONNX_BACKBONE,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "last_hidden_state": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
        external_data=True,
    )

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(MODEL_DIR, name))

    fp32_mb = os.path.getsize(ONNX_FP32) / (1024 * 1024)
    backbone_mb = os.path.getsize(ONNX_BACKBONE) / (1024 * 1024)
    print(f"Size: classifier = {fp32_mb:.1f} MB, backbone = {backbone_mb:.1f} MB")

    print("\n4. Verifying ONNX FP32 with ONNXRuntime...")
    sess = ort.InferenceSession(ONNX_FP32, providers=["CPUExecutionProvider"])

    test_queries = [
        ("query: Brazzers Exxtra 1080p XXX", True),
        ("query: The Big Bang Theory S10E01", False),
        (
            "query: [Airandou] I hired a maid, and it turned out she was incredibly sensitive",
            True,
        ),
        ("query: 5-Minute Chair Yoga for Weight Loss", False),
    ]

    for q, exp in test_queries:
        enc = tokenizer([q], return_tensors="np")
        ort_in = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        prob = float(sess.run(None, ort_in)[0][0])
        status = "PASS" if (prob >= 0.5) == exp else "FAIL"
        print(f"  [{status}] prob={prob:.4f} (exp={exp}) :: {q}")

    print("\nONNX export and verification complete!")


if __name__ == "__main__":
    export_fp32()
