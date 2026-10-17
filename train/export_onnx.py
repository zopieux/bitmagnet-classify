"""Export fine-tuned E5 NSFW classifier to standalone ONNX and quantize to dynamic INT8.

Creates a single model.onnx and model_int8.onnx taking input_ids and attention_mask
and returning nsfw_probability directly.
"""

import os

import onnxruntime as ort
import torch
from finetune import MODEL_ID, OUTPUT_DIR, E5Classifier
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoTokenizer

ONNX_DIR = "scratch/e5_nsfw_finetuned"
ONNX_FP32 = os.path.join(ONNX_DIR, "model.onnx")
ONNX_INT8 = os.path.join(ONNX_DIR, "model_int8.onnx")


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


def export_and_quantize():
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
    )

    print(f"3. Quantizing to {ONNX_INT8}...")
    quantize_dynamic(
        model_input=ONNX_FP32,
        model_output=ONNX_INT8,
        weight_type=QuantType.QInt8,
    )

    fp32_mb = os.path.getsize(ONNX_FP32) / (1024 * 1024)
    int8_mb = os.path.getsize(ONNX_INT8) / (1024 * 1024)
    print(f"Size: FP32 = {fp32_mb:.1f} MB -> INT8 = {int8_mb:.1f} MB")

    print("\n4. Verifying ONNX INT8 with ONNXRuntime...")
    sess = ort.InferenceSession(ONNX_INT8, providers=["CPUExecutionProvider"])

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
    export_and_quantize()
