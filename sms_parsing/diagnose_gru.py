"""
diagnose_gru.py — Find out WHY the GRU model isn't learning.

Run this from inside your sms_classifier_cnn_gru folder:
    python diagnose_gru.py

It builds the exact same CNN+GRU model, runs it on a real batch from your
dataset, and prints:
  1. Whether the model's raw output already contains NaN/Inf (dead model).
  2. The output probability spread for a batch (are all predictions
     identical/collapsed before any training even happens?).
  3. The gradient norm for EVERY layer after one real training step —
     this is the key number. If the GRU layer's gradient is ~0.0 while
     Conv1D/Dense layers have normal-sized gradients, the GRU layer itself
     is dead (a real bug in that layer). If ALL gradients are ~0 or NaN,
     it's something upstream (data/loss/input). If gradients look
     reasonably sized (not 0, not NaN, not huge), the model CAN learn and
     the issue is more likely learning rate / epochs / class imbalance.
"""

import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ['TF_DETERMINISTIC_OPS'] = '0'

import sys
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.preprocess import SMSPreprocessor
from models.model import build_cnn_gru_model

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'dataset', 'indian_sms_dataset.csv')

print("=" * 60)
print("  GRU GRADIENT DIAGNOSTIC")
print("=" * 60)

df = pd.read_csv(dataset_path).dropna(subset=['text', 'label'])
print(f"Loaded {len(df)} rows, {df['label'].nunique()} classes")

texts = df['text'].values[:512]
labels = df['label'].values[:512]

prep = SMSPreprocessor(max_vocab_size=5000, max_seq_length=100)
X, y = prep.fit_transform(texts, labels)
cfg = prep.get_config()
print(f"Vocab: {cfg['vocab_size']}  Classes: {cfg['num_classes']}  Batch shape: {X.shape}")

model = build_cnn_gru_model(
    vocab_size=cfg['vocab_size'],
    max_seq_length=cfg['max_seq_length'],
    num_classes=cfg['num_classes'],
    learning_rate=0.001,
)

# ── Step 1: raw forward pass, before any training ──────────────────────
batch_x = X[:32]
batch_y = y[:32]

raw_out = model(batch_x, training=False).numpy()
print("\n[1] Raw output (before training):")
print(f"    contains NaN : {np.isnan(raw_out).any()}")
print(f"    contains Inf : {np.isinf(raw_out).any()}")
print(f"    min/max/mean : {raw_out.min():.6f} / {raw_out.max():.6f} / {raw_out.mean():.6f}")
print(f"    row 0 probs  : {np.round(raw_out[0], 4)}")
print(f"    row 1 probs  : {np.round(raw_out[1], 4)}")
row_std = raw_out.std(axis=0)
print(f"    per-class std across the 32 samples (near 0 = every sample "
      f"predicts identically before training): {np.round(row_std, 6)}")

# ── Step 2: one real training step, inspect gradients per layer ───────
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

with tf.GradientTape() as tape:
    preds = model(batch_x, training=True)
    loss_val = loss_fn(batch_y, preds)

grads = tape.gradient(loss_val, model.trainable_variables)

print(f"\n[2] Loss on this batch: {loss_val.numpy():.6f}")
print("\n[3] Gradient norm per trainable variable:")
print(f"    {'Layer/variable':45s} {'grad norm':>12s}  {'has NaN':>8s}")
print(f"    {'-'*45} {'-'*12}  {'-'*8}")
any_nan = False
for var, grad in zip(model.trainable_variables, grads):
    if grad is None:
        print(f"    {var.name:45s} {'None (no grad!)':>12s}")
        continue
    norm = tf.norm(grad).numpy()
    has_nan = bool(np.isnan(norm))
    any_nan = any_nan or has_nan
    print(f"    {var.name:45s} {norm:12.6f}  {str(has_nan):>8s}")

print("\n[4] Interpretation:")
if any_nan:
    print("    -> NaN gradients detected. The loss/model is numerically")
    print("       unstable. Likely cause: exploding values somewhere")
    print("       upstream (embedding scale, learning rate too high).")
else:
    gru_grads = [tf.norm(g).numpy() for v, g in zip(model.trainable_variables, grads)
                 if g is not None and 'gru' in v.name.lower()]
    other_grads = [tf.norm(g).numpy() for v, g in zip(model.trainable_variables, grads)
                   if g is not None and 'gru' not in v.name.lower()]
    if gru_grads and max(gru_grads) < 1e-6:
        print("    -> GRU layer gradients are ~0 while other layers are not.")
        print("       The GRU layer itself is dead / not receiving gradient.")
        print("       This points to a real bug in how GRU is wired or a")
        print("       TensorFlow kernel issue specific to this layer.")
    elif other_grads and max(other_grads) < 1e-6:
        print("    -> ALL gradients are ~0, not just GRU's. The problem is")
        print("       upstream of the recurrent layer (e.g. dead ReLUs in")
        print("       Conv1D, or the loss itself being flat).")
    else:
        print("    -> Gradients look non-zero and finite everywhere.")
        print("       The model CAN learn. If real training still shows")
        print("       zero improvement, suspect: learning rate too low/high")
        print("       for this batch, or verbose/callback issue hiding")
        print("       real progress, not a dead-layer bug.")

print("\n" + "=" * 60)
