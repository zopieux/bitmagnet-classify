"""End-to-end fine-tuning script for multilingual-e5-small on torrent titles.

Fine-tunes the transformer representation with a classification head on RTX 4090,
saves the optimized PyTorch model, and evaluates accuracy on challenge probes.
"""

import os
import time

import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from train.dataset import extract_clean_dataset
from train.preprocess import sanitize_title_for_model

MODEL_ID = "intfloat/multilingual-e5-small"
OUTPUT_DIR = "scratch/e5_nsfw_source"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 3
LR = 2e-5


class TorrentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"query: {self.texts[idx]}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class E5Classifier(nn.Module):
    def __init__(self, backbone_id):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_id)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over attention mask
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def train_and_evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(
        f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )

    print("1. Loading dataset...")
    train_data, test_data = extract_clean_dataset(num_positives=40000, num_negatives=80000)
    train_texts = [sanitize_title_for_model(t) for t, _ in train_data]
    train_labels = [y for _, y in train_data]
    test_texts = [sanitize_title_for_model(t) for t, _ in test_data]
    test_labels = [y for _, y in test_data]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ds = TorrentDataset(train_texts, train_labels, tokenizer)
    test_ds = TorrentDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)} ({len(train_ds)} items)")
    print(f"Test batches:  {len(test_loader)} ({len(test_ds)} items)")

    model = E5Classifier(MODEL_ID).to(DEVICE)

    # Optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
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
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    criterion = nn.BCEWithLogitsLoss()

    print("\n2. Fine-tuning E5-Small...")
    t0_all = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0_epoch = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dt_epoch = time.time() - t0_epoch
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f} ({dt_epoch:.1f}s)")

    print(f"\nTraining completed in {time.time() - t0_all:.1f}s")

    # Evaluation
    print("\n3. Evaluating on Test Set...")
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"]

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            all_targets.extend(labels.numpy())

    auc = roc_auc_score(all_targets, all_probs)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(
        classification_report(
            all_targets, all_preds, target_names=["Clean (0)", "NSFW (1)"]
        )
    )

    # Test challenge probes
    print("4. Challenge Probes Evaluation:")
    eval_probes = [
        ("Sex and the City S01E01 1080p BluRay", 0),
        ("Sex Education S03E01 720p WEB-DL", 0),
        ("Dan Carlin Hardcore History 65 Supernova in the East IV", 0),
        ("The Big Bang Theory S10E01 720p HDTV", 0),
        ("[HorribleSubs] Kill la Kill - 02 [720p]", 0),
        ("Odna vojna 2009", 0),
        ("archive sea fighting in greece 1897", 0),
        ("[Beatrice-Raws] Sword Art Online II 14.5", 0),
        ("Brazzers Exxtra Kendra Lust XXX 1080p", 1),
        ("Over40Handjobs Alexis Fawx Hardcore XXX", 1),
        ("Starbangers 9 Jeanna Fine DVDRip", 1),
        ("Heyzo 2511 HD", 1),
        ("strapon slaves com submissive male", 1),
    ]

    for title, expected in eval_probes:
        clean = sanitize_title_for_model(title)
        enc = tokenizer(f"query: {clean}", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logit = model(enc["input_ids"], enc["attention_mask"])
            prob = float(torch.sigmoid(logit)[0])
        pred = int(prob >= 0.5)
        status = "PASS" if pred == expected else "FAIL"
        print(f"  [{status}] prob={prob:.4f} (exp={expected}) :: {title}")

    # Save model and tokenizer
    print(f"\n5. Saving fine-tuned weights to {OUTPUT_DIR}...")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    train_and_evaluate()
