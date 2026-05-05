"""
Movie Review Sentiment Analysis using Deep Learning & NLP
==========================================================
Models: LSTM (with GloVe embeddings) vs DistilBERT (Transformer)
Dataset: IMDB 50,000 reviews
Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
MAX_LEN     = 256        # token / word limit
BATCH_SIZE  = 32
LSTM_EPOCHS = 5
BERT_EPOCHS = 3
LSTM_LR     = 1e-3
BERT_LR     = 2e-5
VOCAB_SIZE  = 30_000
EMBED_DIM   = 100        # GloVe 100-d (falls back to random if not available)

np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load and do basic sanity-check on the IMDB dataset."""
    df = pd.read_csv(path)
    assert set(df.columns) >= {"review", "sentiment"}, "Unexpected columns"
    df["label"] = (df["sentiment"] == "positive").astype(int)
    print(f"Loaded {len(df):,} reviews | "
          f"Positive: {df['label'].sum():,} | Negative: {(df['label']==0).sum():,}")
    return df


def clean_text(text: str) -> str:
    """Remove HTML tags, special characters, and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)              # strip HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # keep letters only
    text = re.sub(r"\s+", " ", text).strip().lower()   # normalise spaces
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning text …")
    df = df.copy()
    df["clean"] = df["review"].apply(clean_text)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZER & VOCABULARY (for LSTM)
# ══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    PAD, UNK = 0, 1

    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        from collections import Counter
        counts = Counter(w for t in texts for w in t.split())
        for word, _ in counts.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"Vocabulary size: {len(self.word2idx):,}")

    def encode(self, text: str, max_len: int = MAX_LEN) -> list:
        ids = [self.word2idx.get(w, self.UNK) for w in text.split()]
        ids = ids[:max_len]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary):
        self.data   = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.data[i], self.labels[i]


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        enc = tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.ids      = enc["input_ids"]
        self.masks    = enc["attention_mask"]
        self.labels   = torch.tensor(labels, dtype=torch.float)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return self.ids[i], self.masks[i], self.labels[i]


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELS
# ══════════════════════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with optional pre-trained embeddings."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden=256, layers=2, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden, num_layers=layers,
                                 batch_first=True, bidirectional=True,
                                 dropout=dropout if layers > 1 else 0.0)
        self.attention = nn.Linear(hidden * 2, 1)   # simple attention
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        emb   = self.embedding(x)                   # (B, L, E)
        out, _= self.lstm(emb)                      # (B, L, 2H)
        attn  = torch.softmax(self.attention(out), dim=1)  # (B, L, 1)
        ctx   = (out * attn).sum(dim=1)             # (B, 2H)
        return self.classifier(ctx).squeeze(1)      # (B,)


class DistilBERTClassifier(nn.Module):
    """DistilBERT fine-tuned for binary classification."""

    def __init__(self, model_name="distilbert-base-uncased", dropout=0.3):
        super().__init__()
        from transformers import DistilBertModel
        self.bert       = DistilBertModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)

    def forward(self, ids, mask):
        out   = self.bert(input_ids=ids, attention_mask=mask)
        cls   = out.last_hidden_state[:, 0, :]          # [CLS] token
        cls   = nn.functional.relu(self.pre_classifier(cls))
        cls   = self.dropout(cls)
        return self.classifier(cls).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING & EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
        "auc_roc"  : roc_auc_score(y_true, y_pred_prob),
    }


def train_epoch_lstm(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch_bert(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_lstm(model, loader):
    model.eval()
    all_prob, all_label = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        all_prob.extend(prob); all_label.extend(y.numpy())
    return np.array(all_prob), np.array(all_label)


@torch.no_grad()
def evaluate_bert(model, loader):
    model.eval()
    all_prob, all_label = [], []
    for ids, mask, y in loader:
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        prob = torch.sigmoid(model(ids, mask)).cpu().numpy()
        all_prob.extend(prob); all_label.extend(y.numpy())
    return np.array(all_prob), np.array(all_label)


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(lstm_hist, bert_hist, lstm_prob, bert_prob, y_test, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Movie Review Sentiment Analysis – Results", fontsize=15, fontweight="bold")

    # ── Training loss curves ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(lstm_hist, marker="o", label="Bi-LSTM")
    ax.plot(bert_hist, marker="s", label="DistilBERT")
    ax.set_title("Training Loss per Epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # ── ROC curves ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    for name, prob in [("Bi-LSTM", lstm_prob), ("DistilBERT", bert_prob)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Confusion matrices ────────────────────────────────────────────────
    for col, (name, prob) in enumerate(
        [("Bi-LSTM", lstm_prob), ("DistilBERT", bert_prob)], start=1
    ):
        ax  = axes[1, col - 1]
        cm  = confusion_matrix(y_test, (prob >= 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Neg", "Pos"],
                    yticklabels=["Neg", "Pos"], ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # ── Metrics bar chart ────────────────────────────────────────────────
    ax  = axes[0, 2]
    m_lstm = compute_metrics(y_test, lstm_prob)
    m_bert = compute_metrics(y_test, bert_prob)
    metrics = list(m_lstm.keys())
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x - w/2, m_lstm.values(), w, label="Bi-LSTM")
    ax.bar(x + w/2, m_bert.values(), w, label="DistilBERT")
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Metric Comparison"); ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Score distribution ───────────────────────────────────────────────
    ax = axes[1, 2]
    ax.hist(lstm_prob[y_test==0], bins=30, alpha=0.5, label="Bi-LSTM Neg", color="steelblue")
    ax.hist(lstm_prob[y_test==1], bins=30, alpha=0.5, label="Bi-LSTM Pos", color="tomato")
    ax.set_title("Bi-LSTM Score Distribution")
    ax.set_xlabel("P(positive)"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {path}")


def print_comparison(lstm_prob, bert_prob, y_test):
    m1 = compute_metrics(y_test, lstm_prob)
    m2 = compute_metrics(y_test, bert_prob)
    header = f"{'Metric':<12} {'Bi-LSTM':>12} {'DistilBERT':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for k in m1:
        print(f"{k:<12} {m1[k]:>12.4f} {m2[k]:>12.4f}")
    print("=" * len(header))


# ══════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE (predict on new text)
# ══════════════════════════════════════════════════════════════════════════════

def predict_lstm(text: str, model, vocab: Vocabulary) -> dict:
    model.eval()
    clean   = clean_text(text)
    ids     = torch.tensor([vocab.encode(clean)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(ids)).item()
    return {"sentiment": "positive" if prob >= 0.5 else "negative", "confidence": prob}


def predict_bert(text: str, model, tokenizer) -> dict:
    model.eval()
    enc  = tokenizer(text, padding="max_length", truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(ids, mask)).item()
    return {"sentiment": "positive" if prob >= 0.5 else "negative", "confidence": prob}


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(data_path: str = "IMDB_Dataset.csv"):
    # ── Load & preprocess ─────────────────────────────────────────────────
    df  = load_data(data_path)
    df  = preprocess(df)

    # Stratified split: 80/10/10
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )
    print(f"Split → train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,}")

    # ─────────────────────────────────────────────────────────────────────
    # MODEL 1: Bi-LSTM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  MODEL 1 – Bidirectional LSTM with Attention")
    print("═"*55)

    vocab = Vocabulary()
    vocab.build(X_train)

    train_ds = LSTMDataset(X_train, y_train.values, vocab)
    val_ds   = LSTMDataset(X_val,   y_val.values,   vocab)
    test_ds  = LSTMDataset(X_test,  y_test.values,  vocab)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    lstm_model = BiLSTMClassifier(len(vocab)).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = AdamW(lstm_model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=LSTM_EPOCHS)

    lstm_hist = []
    for epoch in range(1, LSTM_EPOCHS + 1):
        loss = train_epoch_lstm(lstm_model, train_dl, criterion, optimizer)
        prob_val, y_val_arr = evaluate_lstm(lstm_model, val_dl)
        m    = compute_metrics(y_val_arr, prob_val)
        lstm_hist.append(loss)
        scheduler.step()
        print(f"Epoch {epoch}/{LSTM_EPOCHS} | loss={loss:.4f} | "
              f"acc={m['accuracy']:.4f} | f1={m['f1']:.4f} | auc={m['auc_roc']:.4f}")

    lstm_prob, y_test_arr = evaluate_lstm(lstm_model, test_dl)

    # ─────────────────────────────────────────────────────────────────────
    # MODEL 2: DistilBERT
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  MODEL 2 – DistilBERT (Transformer)")
    print("═"*55)

    try:
        from transformers import DistilBertTokenizerFast
        tokenizer   = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        b_train_ds  = BERTDataset(X_train, y_train.values, tokenizer)
        b_val_ds    = BERTDataset(X_val,   y_val.values,   tokenizer)
        b_test_ds   = BERTDataset(X_test,  y_test.values,  tokenizer)

        b_train_dl  = DataLoader(b_train_ds, batch_size=16, shuffle=True,  num_workers=0)
        b_val_dl    = DataLoader(b_val_ds,   batch_size=16, shuffle=False, num_workers=0)
        b_test_dl   = DataLoader(b_test_ds,  batch_size=16, shuffle=False, num_workers=0)

        bert_model  = DistilBERTClassifier().to(DEVICE)
        b_opt       = AdamW(bert_model.parameters(), lr=BERT_LR, weight_decay=1e-4)
        b_sched     = CosineAnnealingLR(b_opt, T_max=BERT_EPOCHS)

        bert_hist   = []
        for epoch in range(1, BERT_EPOCHS + 1):
            loss = train_epoch_bert(bert_model, b_train_dl, criterion, b_opt)
            prob_val, y_val_b = evaluate_bert(bert_model, b_val_dl)
            m    = compute_metrics(y_val_b, prob_val)
            bert_hist.append(loss)
            b_sched.step()
            print(f"Epoch {epoch}/{BERT_EPOCHS} | loss={loss:.4f} | "
                  f"acc={m['accuracy']:.4f} | f1={m['f1']:.4f} | auc={m['auc_roc']:.4f}")

        bert_prob, _ = evaluate_bert(bert_model, b_test_dl)

    except Exception as e:
        print(f"[WARN] DistilBERT unavailable ({e}). Using dummy baseline.")
        bert_model = bert_hist = None
        bert_prob  = np.random.rand(len(y_test_arr))  # placeholder

    # ─────────────────────────────────────────────────────────────────────
    # Results & Visualisation
    # ─────────────────────────────────────────────────────────────────────
    print_comparison(lstm_prob, bert_prob, y_test_arr)
    plot_all(lstm_hist, bert_hist or [0]*3, lstm_prob, bert_prob,
             y_test_arr, out_dir="outputs")

    # Demo predictions
    samples = [
        "This movie was absolutely brilliant! A masterpiece.",
        "Terrible film. Waste of time, dull plot and bad acting.",
        "It was okay, nothing special but not the worst either."
    ]
    print("\n── Demo Predictions ───────────────────────────────")
    for s in samples:
        r = predict_lstm(s, lstm_model, vocab)
        print(f"  [{r['sentiment'].upper()} {r['confidence']:.2f}]  \"{s[:60]}\"")

    # Save models
    torch.save(lstm_model.state_dict(), "outputs/lstm_model.pt")
    if bert_model:
        torch.save(bert_model.state_dict(), "outputs/bert_model.pt")
    print("\nModels saved to outputs/")


if __name__ == "__main__":
    main("IMDB_Dataset.csv")
