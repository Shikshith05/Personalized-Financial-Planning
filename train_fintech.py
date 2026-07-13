"""
train_fintech.py — CNN + GRU (Spend Category Model)
=====================================================
GRU counterpart of the CNN+LSTM spend-category model.
Trains on spend_category_dataset.csv (10 classes: Food & Dining,
Shopping, Travel, Utilities, Investment, Loan & EMI, Healthcare,
Education, Entertainment, Personal Transfer).

This is the direct comparison point for the paper's headline result:
    CNN+LSTM (baseline) → 96% accuracy, 131,584 recurrent params
    CNN+GRU  (proposed) → ~comparable accuracy, ~99,072 recurrent params (~25% fewer)

Run AFTER train.py:
    python train_fintech.py

Saves:
    models/saved/fintech_model.keras
    models/saved/fintech_model_final.keras
    preprocessors/fintech_preprocessor.pkl
"""

import os

# MUST be set before tensorflow is imported — see train.py for the full
# explanation. Works around a oneDNN GRU gradient-flow bug on Windows CPU.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import set_random_seed, save_object, create_directories
from preprocessing.preprocess import SMSPreprocessor
from models.model import build_cnn_gru_model


class EpochMetricsCallback(tf.keras.callbacks.Callback):
    """Tracks per-epoch wall-clock time and peak memory (MB)."""

    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_memory = []
        self._epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        tracemalloc.start()
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.epoch_times.append(elapsed)
        self.epoch_memory.append(peak / 1024 / 1024)


def train_fintech(dataset_path=None, epochs=25, batch_size=32, lr=0.001):
    set_random_seed(42)
    # See train.py for the full explanation: TF_DETERMINISTIC_OPS=1 (set
    # inside set_random_seed) breaks GRU gradient flow on CPU. Turn it back
    # off; the numpy/random/tf seeds set above are unaffected.
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if dataset_path is None:
        dataset_path = os.path.join(
            script_dir, 'dataset', 'spend_category_dataset.csv')

    create_directories(script_dir)
    model_dir = os.path.join(script_dir, 'models', 'saved')
    prep_dir = os.path.join(script_dir, 'preprocessors')

    print("\n" + "=" * 60)
    print("  SPEND CATEGORY MODEL — CNN + GRU TRAINING")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(dataset_path).dropna(subset=['text', 'label'])
    before = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed:,} duplicate texts (train/test leakage prevention)")
    print(f"  Samples : {len(df):,}")
    print(f"  Classes : {sorted(df['label'].unique())}")
    print(df['label'].value_counts().to_string())

    dupes = df.duplicated(subset=['text']).sum()
    if dupes:
        print(f"  ⚠ {dupes} duplicate texts found (not removed — matches baseline run)")

    texts = df['text'].values
    labels = df['label'].values

    # ── Split ─────────────────────────────────────────────────────────────
    print("\n[2/5] Splitting 80/10/10...")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
    print(f"  Train {len(X_tr)} | Val {len(X_val)} | Test {len(X_te)}")

    # ── Preprocess ────────────────────────────────────────────────────────
    print("\n[3/5] Preprocessing...")
    prep = SMSPreprocessor(max_vocab_size=8000, max_seq_length=32)
    X_train, y_train = prep.fit_transform(X_tr, y_tr)
    X_val_p, y_val_p = prep.transform(X_val, y_val)
    X_test_p, y_test = prep.transform(X_te, y_te)

    cfg = prep.get_config()
    print(f"  Vocab: {cfg['vocab_size']}  Classes: {cfg['num_classes']}")
    save_object(prep, os.path.join(prep_dir, 'fintech_preprocessor.pkl'))

    # ── Build CNN + GRU ───────────────────────────────────────────────────
    print("\n[4/5] Building CNN+GRU...")
    model = build_cnn_gru_model(
        vocab_size=cfg['vocab_size'],
        max_seq_length=cfg['max_seq_length'],
        num_classes=cfg['num_classes'],
        learning_rate=lr,
        num_filters=128,
        gru_units=128,
        dense_units=64,
        dropout_rate=0.4,
    )
    model.summary()

    total_params = model.count_params()
    print(f"\n  Total parameters : {total_params:,}")
    print(f"  CNN+LSTM baseline: ~{total_params + 32896:,} params (estimated, 131,584 in recurrent layer)")
    print(f"  Parameter saving : ~32,896 params (~25% less in recurrent layer)")

    best_path = os.path.join(model_dir, 'fintech_model.keras')
    final_path = os.path.join(model_dir, 'fintech_model_final.keras')

    metrics_cb = EpochMetricsCallback()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path, monitor='val_accuracy', save_best_only=True),
        metrics_cb,
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training... (epochs={epochs}, batch={batch_size})")
    total_start = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val_p, y_val_p),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2,
    )
    total_time = time.time() - total_start
    model.save(final_path)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n── Test Evaluation ──────────────────────────────────────────")
    y_pred = np.argmax(model.predict(X_test_p, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=prep.classes_))

    test_loss, test_acc = model.evaluate(X_test_p, y_test, verbose=0)

    # ── Efficiency report ────────────────────────────────────────────────
    avg_time = np.mean(metrics_cb.epoch_times)
    avg_mem = np.mean(metrics_cb.epoch_memory)
    peak_mem = np.max(metrics_cb.epoch_memory)

    print(f"\n{'=' * 60}")
    print("  COMPUTATIONAL EFFICIENCY REPORT (Spend Category / Fintech)")
    print(f"{'=' * 60}")
    print(f"  Model            : CNN + GRU  (proposed)")
    print(f"  Baseline         : CNN + LSTM (96.00% test accuracy)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total parameters : {total_params:,}")
    print(f"  Total train time : {total_time:.1f}s")
    print(f"  Avg time/epoch   : {avg_time:.2f}s")
    print(f"  Avg memory/epoch : {avg_mem:.1f} MB")
    print(f"  Peak memory      : {peak_mem:.1f} MB")
    print(f"  Test Accuracy    : {test_acc * 100:.2f}%")
    print(f"{'=' * 60}")

    print(f"\n✓ Best model  → {best_path}")
    print(f"✓ Final model → {final_path}")
    print(f"✓ Preprocessor→ {os.path.join(prep_dir, 'fintech_preprocessor.pkl')}")
    print("=" * 60)

    return model, prep, {
        'X_test': X_test_p,
        'y_test': y_test,
        'test_acc': test_acc,
        'total_time': total_time,
        'avg_time_per_epoch': avg_time,
        'peak_memory_mb': peak_mem,
    }


if __name__ == '__main__':
    train_fintech()
