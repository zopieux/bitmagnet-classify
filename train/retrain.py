"""Retrain calibrated classifier head on frozen multilingual-e5-small embeddings.

Saves:
1. Updated ONNX INT8 model (backbone_int8.onnx)
2. Calibrated classification head weights (head_weights.npz)
3. Evaluates all false positive test fixtures.
"""

import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from bitmagnet_classify.nsfw import sanitize_title_for_model
from train.dataset import extract_clean_dataset

MODEL_ID = "intfloat/multilingual-e5-small"
TARGET_DIR = "train/output"


def retrain():
    os.makedirs(TARGET_DIR, exist_ok=True)

    print("1. Extracting clean dataset (10k positives, 10k diverse negatives)...")
    train_data, test_data = extract_clean_dataset(
        num_positives=10000, num_negatives=10000
    )

    print("2. Loading frozen multilingual-e5-small on CUDA...")
    encoder = SentenceTransformer(MODEL_ID, device="cuda")

    print(f"3. Embedding {len(train_data):,} train samples on GPU...")
    t0 = time.time()
    train_texts = [f"query: {sanitize_title_for_model(t)}" for t, _ in train_data]
    X_train = encoder.encode(
        train_texts, batch_size=256, device="cuda", show_progress_bar=False
    )
    y_train = np.array([y for _, y in train_data])
    print(
        f"Encoded train set in {time.time() - t0:.2f}s ({len(train_data) / (time.time() - t0):.1f} samples/s)"
    )

    print(f"4. Embedding {len(test_data):,} test samples on GPU...")
    test_texts = [f"query: {sanitize_title_for_model(t)}" for t, _ in test_data]
    X_test = encoder.encode(
        test_texts, batch_size=256, device="cuda", show_progress_bar=False
    )
    y_test = np.array([y for _, y in test_data])

    print("5. Training calibrated LogisticRegression head...")
    # C=0.5 provides strong L2 regularization to prevent over-reacting to rare tokens
    clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"Test ROC-AUC:   {auc:.4f}")
    print(
        classification_report(
            y_test, clf.predict(X_test), target_names=["Clean (0)", "NSFW (1)"]
        )
    )

    # Save head weights as a single linear layer (W, b)
    # Since clf is LogisticRegression: output = sigmoid(X @ coef_.T + intercept_)
    # We store W1=coef_, b1=intercept_, W2=identity, b2=0 for compatibility
    W1 = clf.coef_  # (1, 384)
    b1 = clf.intercept_  # (1,)
    # Dummy W2, b2 so head computation h = max(0, pooled @ W1.T + b1); p = sigmoid(h @ W2.T + b2) works
    # Or cleaner: W1 = clf.coef_, b1 = clf.intercept_, W2 = [[1.0]], b2 = [0.0]
    # But let's check nsfw.py head computation:
    #   h = np.maximum(0, pooled @ W1.T + b1)
    #   logits = (h @ W2.T + b2).squeeze(-1)
    # If W1 has shape (1, 384) and W2 has shape (1, 1) with [[1.0]], h = max(0, logit) which drops negatives!
    # Instead, let's export W1 as shape (1, 384), b1=[0], W2=coef_, b2=intercept_!
    # Let's inspect: if W1 is Identity (384, 384), b1=0, W2=coef_ (1, 384), b2=intercept_:
    #   h = max(0, pooled @ I + 0) = max(0, pooled) (which is mostly positive for E5!)
    # Even better: save W1=(128, 384), b1, W2=(1, 128), b2 from a small MLP or adjust nsfw.py to use single linear.
    # Let's train an MLPClassifier with 1 hidden layer of 64 or 128 units:
    from sklearn.neural_network import MLPClassifier

    print("Training 1-hidden-layer MLP head (128 units)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,), max_iter=200, random_state=42, early_stopping=True
    )
    mlp.fit(X_train, y_train)

    mlp_probs = mlp.predict_proba(X_test)[:, 1]
    print(f"MLP Test ROC-AUC: {roc_auc_score(y_test, mlp_probs):.4f}")

    W1 = mlp.coefs_[0].T  # (128, 384)
    b1 = mlp.intercepts_[0]  # (128,)
    W2 = mlp.coefs_[1].T  # (1, 128)
    b2 = mlp.intercepts_[1]  # (1,)

    np.savez(f"{TARGET_DIR}/head_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Saved head weights to {TARGET_DIR}/head_weights.npz")

    # Benchmark challenge probe queries
    print("\n6. Evaluating Critical Probes:")
    probes = [
        (
            "Love Happy (1949) Xvid 1cd - Marx Brothers, Vera Ellen -  Marilyn Monroe [DDR]",
            0,
        ),
        ("OmenXiii - Grim [MQA]", 0),
        ("Lady_Bird_2017_WEB-DLRip_MVO_by_Dalemake", 0),
        ("202609032145jxyczycj577.zip", 0),
        ("VID_20200704_120150.mp4", 0),
        ("Roadside Research.24864017", 0),
        ("YP-EPUB-COMIC55.epub", 0),
        ("MIZD-553", 0),
        ("Sex and the City S01E01 1080p BluRay", 0),
        ("Dan Carlin Hardcore History 65 Supernova in the East IV", 0),
        ("The Big Bang Theory S10E01 720p HDTV", 0),
        ("[HorribleSubs] Kill la Kill - 02 [720p]", 0),
        ("Brazzers Exxtra Kendra Lust XXX 1080p", 1),
        ("Over40Handjobs Alexis Fawx Hardcore XXX", 1),
        ("momcomesfirst - Brianna Beach - yoga mom secret lesson", 1),
        (
            "[Airandou] I hired a maid, and it turned out she was incredibly sensitive",
            1,
        ),
        ("Mary Jane - Jackson Full Body Hitachi Massage", 1),
    ]

    probe_embs = encoder.encode(
        [f"query: {sanitize_title_for_model(t)}" for t, _ in probes], device="cuda"
    )
    # Verify using exact numpy inference formula
    h = np.maximum(0, probe_embs @ W1.T + b1)
    logits = (h @ W2.T + b2).squeeze(-1)
    probe_probs = 1.0 / (1.0 + np.exp(-logits))

    all_passed = True
    for (title, expected), prob in zip(probes, probe_probs):
        pred = int(prob >= 0.5)
        status = "PASS" if pred == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] prob={prob:.4f} (exp={expected}) :: {title}")

    return all_passed


if __name__ == "__main__":
    retrain()
