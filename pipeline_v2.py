"""
Improved Pipeline for Hierarchical Multi-Label Text Classification (v2)

Root-cause fixes for low accuracy:

[A] Silver Label Generation (CRITICAL)
  Bug 1: 77.6% of keywords contained underscores ('blood_pressure_monitors') and
         never matched natural review text ('blood pressure monitor'). Fixed by
         converting underscores to spaces for all keywords.
  Bug 2: Single-word generic keywords ('relaxation', 'food') matched incidentally
         in reviews about unrelated products, creating massive false positives.
         Fixed by using only MULTI-WORD keywords from the keyword file.
  Fix 3: Multi-word keywords get 3x score weight; they are far more discriminative.
  Fix 4: Class names added as high-value anchor keywords (2x weight if multi-word).
  Fix 5: Minimum score threshold of 8.0 filters out weak/noisy matches.

[B] Training
  Fix 6: Linear LR warmup (10% of steps) + decay — prevents early divergence.
  Fix 7: Gradient clipping (max_norm=1.0) — stabilises fine-tuning.

[C] Self-Training
  Fix 8: Confidence threshold lowered 0.8 → 0.5 (was allowing only 582 pseudo-
          labels in round 1; lowering yields far more training signal).

[D] Prediction
  Fix 9: Hierarchy-aware post-processing — predicted leaf labels are propagated to
          ancestor classes to ensure hierarchical consistency.
  Fix 10: Dynamic label count (2 or 3) based on confidence spread.
"""

import os
import re
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# =============================================================================
# [1] Configuration
# =============================================================================
CONFIG = {
    'model_name': 'bert-base-uncased',
    'batch_size': 32,
    'max_len': 256,                # Increased from 128; reviews can be long
    'lr': 2e-5,
    'epochs_init': 3,
    'epochs_st': 2,
    'st_iterations': 3,
    'confidence_threshold': 0.5,   # Lowered from 0.8 — more pseudo-labels accepted
    'seed': 42,
    'warmup_ratio': 0.1,           # NEW: 10% warmup steps
    'max_grad_norm': 1.0,          # NEW: gradient clipping
    'silver_min_score': 8.0,       # Minimum score to assign a silver label
    'mw_keyword_weight': 3.0,      # Score multiplier for multi-word file keywords
    'mw_classname_weight': 2.0,    # Score multiplier for multi-word class names
    'sw_classname_weight': 1.0,    # Score multiplier for single-word class names
    'max_silver_labels': 3,        # Max leaf classes per document
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG['seed'])
print(f"Running on device: {CONFIG['device']}")

# =============================================================================
# [2] File Paths
# =============================================================================
ROOT = Path("Amazon_products")
TRAIN_CORPUS_PATH = ROOT / "train" / "train_corpus.txt"
TEST_CORPUS_PATH  = ROOT / "test"  / "test_corpus.txt"
CLASSES_PATH      = ROOT / "classes.txt"
HIERARCHY_PATH    = ROOT / "class_hierarchy.txt"
KEYWORDS_PATH     = ROOT / "class_related_keywords.txt"
SUBMISSION_PATH   = "submission_v2.csv"

# =============================================================================
# [3] Data Loading
# =============================================================================
def load_classes(path):
    id2name, name2id = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cid, cname = int(parts[0]), parts[1]
                id2name[cid] = cname
                name2id[cname] = cid
    return id2name, name2id


def load_hierarchy(path):
    parents = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                p, c = int(parts[0]), int(parts[1])
                parents[c].append(p)
    return parents


def load_keywords_raw(path):
    class_keywords = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                cname, kw_str = line.strip().split(':', 1)
                class_keywords[cname] = [k.strip() for k in kw_str.split(',')]
    return class_keywords


def load_corpus(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                data.append((int(parts[0]), parts[1]))
    data.sort(key=lambda x: x[0])
    return data


# =============================================================================
# [4] Improved Keyword Index
# =============================================================================
def build_keyword_index(class_keywords_raw, name2id):
    """
    Build a normalized, IDF-weighted keyword index.

    Key design choices:
    - Only MULTI-WORD keywords from the file are included.
      Single-word keywords like 'relaxation', 'food', 'organic' match too many
      unrelated reviews and were the primary source of false positive silver labels.
    - Class names are added as anchors (multi-word names get 2x weight; single-word
      names get 1x but won't pass the silver_min_score=8 threshold on their own).
    - Both singular and plural forms are added for each keyword.
    """
    # cid -> {multi-word file keywords}
    mw_file_kws: dict[int, list[str]] = {}
    # cid -> (class_name_str, weight)
    cn_kws: dict[int, tuple[str, float]] = {}

    all_kw_for_idf: Counter = Counter()

    for cname, kws in class_keywords_raw.items():
        if cname not in name2id:
            continue
        cid = name2id[cname]

        # -- Class name keyword --
        cn = cname.replace('_', ' ').lower()
        cn_weight = (CONFIG['mw_classname_weight'] if len(cn.split()) >= 2
                     else CONFIG['sw_classname_weight'])
        cn_kws[cid] = (cn, cn_weight)
        all_kw_for_idf[cn] += 1

        # -- Multi-word file keywords only --
        mw_set: set[str] = set()
        for kw in kws:
            kn = kw.replace('_', ' ').lower().strip()
            if len(kn.split()) >= 2:
                mw_set.add(kn)
                # Add both singular and plural variants
                mw_set.add(kn[:-1] if kn.endswith('s') else kn + 's')

        if mw_set:
            mw_file_kws[cid] = sorted(mw_set, key=len, reverse=True)
            for kw in mw_set:
                all_kw_for_idf[kw] += 1

    N = len(name2id)
    keyword_idf = {kw: math.log(N / max(df, 1)) for kw, df in all_kw_for_idf.items()}

    # Compile regex patterns
    compiled_mw: dict[int, re.Pattern] = {}
    for cid, kws in mw_file_kws.items():
        compiled_mw[cid] = re.compile(
            r'\b(' + '|'.join(map(re.escape, kws)) + r')\b'
        )

    compiled_cn: dict[int, tuple[re.Pattern, float]] = {}
    for cid, (cn, weight) in cn_kws.items():
        cn_s = cn[:-1] if cn.endswith('s') else cn + 's'
        pattern = re.compile(r'\b(?:' + re.escape(cn) + r'|' + re.escape(cn_s) + r')\b')
        compiled_cn[cid] = (pattern, weight, keyword_idf.get(cn, 0))

    print(f"Keyword index: {sum(len(v) for v in mw_file_kws.values())} multi-word keywords "
          f"across {len(mw_file_kws)} classes")
    return compiled_mw, compiled_cn, keyword_idf


# =============================================================================
# [5] Silver Label Generation
# =============================================================================
def propagate_to_ancestors(class_ids, parents_map):
    """Expand a set of class IDs to include all ancestors in the hierarchy."""
    all_labels = set(class_ids)
    queue = list(class_ids)
    while queue:
        curr = queue.pop()
        for p in parents_map.get(curr, []):
            if p not in all_labels:
                all_labels.add(p)
                queue.append(p)
    return all_labels


def score_document(text_lower, compiled_mw, compiled_cn, keyword_idf):
    """Compute per-class IDF-weighted keyword scores for a single document."""
    scores: dict[int, float] = {}

    for cid, pattern in compiled_mw.items():
        matches = set(pattern.findall(text_lower))
        if matches:
            sc = sum(keyword_idf.get(m, 0) * CONFIG['mw_keyword_weight'] for m in matches)
            if sc > 0:
                scores[cid] = scores.get(cid, 0.0) + sc

    for cid, (pattern, weight, idf) in compiled_cn.items():
        if pattern.search(text_lower):
            scores[cid] = scores.get(cid, 0.0) + idf * weight

    return scores


def generate_silver_labels(corpus, compiled_mw, compiled_cn, keyword_idf,
                            parents_map, num_classes):
    """
    Generate silver labels for the training corpus.

    A document is labeled only when at least one class achieves
    silver_min_score (default 8.0). This threshold requires at minimum
    one multi-word keyword match (typical score ~18) or a multi-word class-name
    match (~12.5), effectively filtering single-word incidental matches.
    """
    min_score = CONFIG['silver_min_score']
    max_leaves = CONFIG['max_silver_labels']

    labeled_data, unlabeled_data = [], []

    for did, text in tqdm(corpus, desc="Generating Silver Labels"):
        text_lower = text.lower()
        class_scores = score_document(text_lower, compiled_mw, compiled_cn, keyword_idf)

        # Filter to classes meeting the minimum score
        strong = {c: s for c, s in class_scores.items() if s >= min_score}

        if strong:
            # Take top leaf-level classes by score, then propagate upward
            top_classes = sorted(strong, key=lambda c: -strong[c])[:max_leaves]
            all_labels = propagate_to_ancestors(top_classes, parents_map)

            label_vec = np.zeros(num_classes, dtype=np.float32)
            for cid in all_labels:
                if cid < num_classes:
                    label_vec[cid] = 1.0
            labeled_data.append((text, label_vec))
        else:
            unlabeled_data.append((text, np.zeros(num_classes, dtype=np.float32)))

    print(f"Labeled (seed): {len(labeled_data):,} / Unlabeled: {len(unlabeled_data):,}")
    return labeled_data, unlabeled_data


# =============================================================================
# [6] Model & Dataset
# =============================================================================
class TaxoClassModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out[0][:, 0, :]
        return self.classifier(self.dropout(cls))


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels = self.data[idx]
        enc = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float),
        }


# =============================================================================
# [7] Training Loop (with scheduler + gradient clipping)
# =============================================================================
def make_optimizer_scheduler(model, num_training_steps):
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    num_warmup = int(CONFIG['warmup_ratio'] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, desc):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=desc):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# =============================================================================
# [8] Hierarchy-Aware Prediction
# =============================================================================
def predict_with_hierarchy(model, loader, device, num_classes, parents_map):
    """
    Predict multi-labels and propagate to ancestor classes.

    For each sample:
    1. Get top-3 class probabilities from the model.
    2. Dynamically choose 2 or 3 labels (use 2 if the 3rd is much weaker than the 2nd).
    3. Propagate the selected classes up through the hierarchy.
    4. Return the top-3 highest-scoring labels from the propagated set.
    """
    model.eval()
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask=mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            for p_vec in probs:
                sorted_idx = np.argsort(p_vec)[::-1]
                top3_probs = p_vec[sorted_idx[:3]]

                # Dynamic label count: if 3rd prob is <40% of 2nd, use 2 labels
                if top3_probs[2] < top3_probs[1] * 0.4:
                    leaf_preds = list(sorted_idx[:2])
                else:
                    leaf_preds = list(sorted_idx[:3])

                # Propagate to ancestors for hierarchical consistency
                full_labels = propagate_to_ancestors(leaf_preds, parents_map)

                # From propagated set, return top-3 by model confidence
                full_sorted = sorted(
                    full_labels,
                    key=lambda c: p_vec[c] if c < num_classes else 0.0,
                    reverse=True,
                )
                all_labels.append(sorted(full_sorted[:3]))

    return all_labels


# =============================================================================
# [9] Main Pipeline
# =============================================================================
def run():
    print("=" * 60)
    print("[1/7] Loading resources...")
    id2name, name2id = load_classes(CLASSES_PATH)
    parents_map = load_hierarchy(HIERARCHY_PATH)
    class_keywords_raw = load_keywords_raw(KEYWORDS_PATH)
    train_raw = load_corpus(TRAIN_CORPUS_PATH)
    test_raw  = load_corpus(TEST_CORPUS_PATH)
    num_classes = len(id2name)
    print(f"Classes: {num_classes}, Train: {len(train_raw):,}, Test: {len(test_raw):,}")

    print("\n[2/7] Building improved keyword index...")
    compiled_mw, compiled_cn, keyword_idf = build_keyword_index(class_keywords_raw, name2id)

    print("\n[3/7] Generating silver labels (multi-word keywords, IDF-weighted)...")
    labeled_train, unlabeled_train = generate_silver_labels(
        train_raw, compiled_mw, compiled_cn, keyword_idf, parents_map, num_classes
    )

    print("\n[4/7] Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = TaxoClassModel(CONFIG['model_name'], num_classes).to(CONFIG['device'])
    criterion = nn.BCEWithLogitsLoss()

    # ── Phase 1: Supervised training on silver labels ────────────────────────
    print(f"\n[5/7] Phase 1 — Supervised training ({CONFIG['epochs_init']} epochs)...")
    train_ds = TextDataset(labeled_train, tokenizer, CONFIG['max_len'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)

    total_steps = len(train_loader) * CONFIG['epochs_init']
    optimizer, scheduler = make_optimizer_scheduler(model, total_steps)

    for epoch in range(CONFIG['epochs_init']):
        loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            CONFIG['device'], f"  P1 Epoch {epoch+1}/{CONFIG['epochs_init']}"
        )
        print(f"  Epoch {epoch+1} avg loss: {loss:.4f}")

    # ── Phase 2: Self-Training ────────────────────────────────────────────────
    # Test data added as unlabeled candidates (transductive learning is allowed)
    candidate_data = unlabeled_train + [(t, np.zeros(num_classes)) for _, t in test_raw]

    for loop in range(CONFIG['st_iterations']):
        print(f"\n[6/7] Self-Training round {loop+1}/{CONFIG['st_iterations']}...")

        cand_ds = TextDataset(candidate_data, tokenizer, CONFIG['max_len'])
        cand_loader = DataLoader(cand_ds, batch_size=CONFIG['batch_size'] * 2, shuffle=False)

        model.eval()
        new_samples = []
        conf_thresh = CONFIG['confidence_threshold']

        with torch.no_grad():
            for i, batch in enumerate(tqdm(cand_loader, desc="  Pseudo-labeling")):
                input_ids = batch['input_ids'].to(CONFIG['device'])
                mask = batch['attention_mask'].to(CONFIG['device'])
                logits = model(input_ids, attention_mask=mask)
                probs = torch.sigmoid(logits).cpu().numpy()

                start = i * (CONFIG['batch_size'] * 2)
                for j, p_vec in enumerate(probs):
                    top_k = np.argsort(p_vec)[-3:]
                    if np.mean(p_vec[top_k]) >= conf_thresh:
                        lbl = np.zeros(num_classes, dtype=np.float32)
                        lbl[top_k] = 1.0
                        new_samples.append((candidate_data[start + j][0], lbl))

        print(f"  Pseudo-labels added: {len(new_samples):,}")
        if not new_samples:
            print("  No new pseudo-labels — stopping self-training early.")
            break

        current_train = labeled_train + new_samples
        st_loader = DataLoader(
            TextDataset(current_train, tokenizer, CONFIG['max_len']),
            batch_size=CONFIG['batch_size'], shuffle=True,
        )
        total_steps_st = len(st_loader) * CONFIG['epochs_st']
        optimizer, scheduler = make_optimizer_scheduler(model, total_steps_st)

        for epoch in range(CONFIG['epochs_st']):
            loss = train_one_epoch(
                model, st_loader, optimizer, scheduler, criterion,
                CONFIG['device'], f"  ST Epoch {epoch+1}/{CONFIG['epochs_st']}"
            )
            print(f"  ST Epoch {epoch+1} avg loss: {loss:.4f}")

    # ── Final Inference ───────────────────────────────────────────────────────
    print("\n[7/7] Final inference + hierarchy post-processing...")
    test_ids = [did for did, _ in test_raw]
    test_texts = [t for _, t in test_raw]
    test_data = [(t, np.zeros(num_classes)) for t in test_texts]

    test_loader = DataLoader(
        TextDataset(test_data, tokenizer, CONFIG['max_len']),
        batch_size=CONFIG['batch_size'] * 2, shuffle=False,
    )

    predicted_labels = predict_with_hierarchy(
        model, test_loader, CONFIG['device'], num_classes, parents_map
    )

    submission = pd.DataFrame({
        'id': test_ids,
        'label': [','.join(str(l) for l in labels) for labels in predicted_labels]
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved to: {SUBMISSION_PATH}")
    print(f"Total samples: {len(test_ids):,}")
    print("\nSample predictions:")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    run()
