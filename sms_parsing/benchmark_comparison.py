"""
benchmark_comparison.py
=======================
Runs CNN+LSTM vs CNN+GRU side by side on YOUR machine
and reports EXACT numbers for:
  - Parameters
  - Memory per epoch (MB)
  - Time per epoch (seconds)
  - Total training time
  - Final accuracy

Run this ONCE and you'll have real numbers for your paper.

Usage:
    python benchmark_comparison.py
"""

import os, sys, time, tracemalloc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.preprocess import SMSPreprocessor

# ── Suppress TF logs ──────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

EPOCHS     = 10   # same for both — fair comparison
BATCH_SIZE = 32
LR         = 0.001


# ── Build CNN + LSTM ──────────────────────────────────────────────────────
def build_lstm(vocab_size, seq_len, num_classes):
    inputs = tf.keras.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, 128)(inputs)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.LSTM(128)(x)              # ← LSTM
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model


# ── Build CNN + GRU ───────────────────────────────────────────────────────
def build_gru(vocab_size, seq_len, num_classes):
    inputs = tf.keras.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, 128)(inputs)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GRU(128)(x)               # ← GRU (only change)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="CNN_GRU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model


# ── Per-epoch timing + memory callback ───────────────────────────────────
class BenchmarkCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times  = []
        self.epoch_memory = []
        self._start       = None

    def on_epoch_begin(self, epoch, logs=None):
        tracemalloc.start()
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._start
        _, peak  = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.epoch_times.append(elapsed)
        self.epoch_memory.append(peak / 1024 / 1024)
        acc     = logs.get('accuracy', 0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        print(f"  Epoch {epoch+1:02d} | {elapsed:.2f}s | "
              f"{peak/1024/1024:.1f}MB | "
              f"acc={acc:.1f}% val={val_acc:.1f}%")


# ── Run one model ─────────────────────────────────────────────────────────
def run_model(model, X_tr, y_tr, X_val, y_val, X_te, y_te, label):
    print(f"\n  {'─'*55}")
    print(f"  Running: {label}")
    print(f"  {'─'*55}")

    cb = BenchmarkCallback()
    total_start = time.perf_counter()
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb],
        verbose=0,
    )
    total_time = time.perf_counter() - total_start

    _, test_acc = model.evaluate(X_te, y_te, verbose=0)

    return {
        'label':       label,
        'params':      model.count_params(),
        'total_time':  total_time,
        'avg_time':    np.mean(cb.epoch_times),
        'min_time':    np.min(cb.epoch_times),
        'max_time':    np.max(cb.epoch_times),
        'avg_memory':  np.mean(cb.epoch_memory),
        'peak_memory': np.max(cb.epoch_memory),
        'min_memory':  np.min(cb.epoch_memory),
        'test_acc':    test_acc * 100,
        'final_val_acc': hist.history['val_accuracy'][-1] * 100,
        'epoch_times':  cb.epoch_times,
        'epoch_memory': cb.epoch_memory,
        'history':      hist.history,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  CNN+LSTM vs CNN+GRU — EXACT BENCHMARK")
    print(f"  Device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print(f"  TensorFlow: {tf.__version__}")
    print("="*60)

    # Load data
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset', 'indian_sms_dataset.csv')
    df = pd.read_csv(dataset_path).dropna(subset=['text','label'])
    print(f"\n  Dataset: {len(df):,} samples")

    texts  = df['text'].values
    labels = df['label'].values

    # Split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # Preprocess
    prep = SMSPreprocessor(max_vocab_size=5000, max_seq_length=100)
    X_train, y_train = prep.fit_transform(X_tr, y_tr)
    X_val_p, y_val_p = prep.transform(X_val, y_val)
    X_test,  y_test  = prep.transform(X_te,  y_te)

    cfg = prep.get_config()
    V, S, C = cfg['vocab_size'], cfg['max_seq_length'], cfg['num_classes']
    print(f"  Vocab: {V}  SeqLen: {S}  Classes: {C}")

    # ── Run CNN + LSTM ────────────────────────────────────────────────────
    print("\n[1/2] Benchmarking CNN + LSTM (baseline)...")
    lstm_model  = build_lstm(V, S, C)
    lstm_result = run_model(lstm_model, X_train, y_train,
                            X_val_p, y_val_p, X_test, y_test,
                            "CNN + LSTM")

    # ── Run CNN + GRU ─────────────────────────────────────────────────────
    print("\n[2/2] Benchmarking CNN + GRU (proposed)...")
    gru_model  = build_gru(V, S, C)
    gru_result = run_model(gru_model, X_train, y_train,
                           X_val_p, y_val_p, X_test, y_test,
                           "CNN + GRU")

    # ── Exact comparison ──────────────────────────────────────────────────
    L = lstm_result
    G = gru_result

    param_saved    = L['params']      - G['params']
    param_pct      = param_saved      / L['params']      * 100
    time_saved     = L['avg_time']    - G['avg_time']
    time_pct       = time_saved       / L['avg_time']    * 100
    mem_saved      = L['peak_memory'] - G['peak_memory']
    mem_pct        = mem_saved        / L['peak_memory'] * 100
    total_saved    = L['total_time']  - G['total_time']
    acc_diff       = G['test_acc']    - L['test_acc']

    print("\n" + "="*60)
    print("  EXACT BENCHMARK RESULTS")
    print("="*60)
    print(f"\n  {'Metric':<35} {'CNN+LSTM':>12} {'CNN+GRU':>12} {'Saving':>10}")
    print(f"  {'-'*72}")
    print(f"  {'Total Parameters':<35} {L['params']:>12,} {G['params']:>12,} {param_saved:>+10,}")
    print(f"  {'Parameter Reduction':<35} {'':>12} {'':>12} {param_pct:>+9.1f}%")
    print(f"  {'-'*72}")
    print(f"  {'Avg Time per Epoch (s)':<35} {L['avg_time']:>12.3f} {G['avg_time']:>12.3f} {time_saved:>+10.3f}s")
    print(f"  {'Min Time per Epoch (s)':<35} {L['min_time']:>12.3f} {G['min_time']:>12.3f}")
    print(f"  {'Max Time per Epoch (s)':<35} {L['max_time']:>12.3f} {G['max_time']:>12.3f}")
    print(f"  {'Time Saving per Epoch':<35} {'':>12} {'':>12} {time_pct:>+9.1f}%")
    print(f"  {'Total Training Time (s)':<35} {L['total_time']:>12.1f} {G['total_time']:>12.1f} {total_saved:>+10.1f}s")
    print(f"  {'-'*72}")
    print(f"  {'Avg Memory per Epoch (MB)':<35} {L['avg_memory']:>12.1f} {G['avg_memory']:>12.1f} {L['avg_memory']-G['avg_memory']:>+10.1f}MB")
    print(f"  {'Peak Memory (MB)':<35} {L['peak_memory']:>12.1f} {G['peak_memory']:>12.1f} {mem_saved:>+10.1f}MB")
    print(f"  {'Min Memory per Epoch (MB)':<35} {L['min_memory']:>12.1f} {G['min_memory']:>12.1f}")
    print(f"  {'Memory Saving (peak)':<35} {'':>12} {'':>12} {mem_pct:>+9.1f}%")
    print(f"  {'-'*72}")
    print(f"  {'Test Accuracy (%)':<35} {L['test_acc']:>11.2f}% {G['test_acc']:>11.2f}% {acc_diff:>+9.2f}%")
    print(f"  {'Final Val Accuracy (%)':<35} {L['final_val_acc']:>11.2f}% {G['final_val_acc']:>11.2f}%")
    print("="*60)

    # ── Per-epoch table ───────────────────────────────────────────────────
    print(f"\n  PER-EPOCH BREAKDOWN:")
    print(f"  {'Epoch':<7} {'LSTM Time':>10} {'GRU Time':>10} {'Faster by':>11} "
          f"{'LSTM Mem':>10} {'GRU Mem':>10} {'Less by':>9}")
    print(f"  {'-'*73}")
    for i in range(EPOCHS):
        lt = L['epoch_times'][i];  gt = G['epoch_times'][i]
        lm = L['epoch_memory'][i]; gm = G['epoch_memory'][i]
        print(f"  {i+1:<7} {lt:>9.2f}s {gt:>9.2f}s {lt-gt:>+10.2f}s "
              f"{lm:>9.1f}MB {gm:>9.1f}MB {lm-gm:>+8.1f}MB")

    # ── Save results ──────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir  = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # CSV
    rows = []
    for i in range(EPOCHS):
        rows.append({
            'epoch':          i+1,
            'lstm_time_s':    round(L['epoch_times'][i],  3),
            'gru_time_s':     round(G['epoch_times'][i],  3),
            'time_saved_s':   round(L['epoch_times'][i] - G['epoch_times'][i], 3),
            'lstm_memory_mb': round(L['epoch_memory'][i], 1),
            'gru_memory_mb':  round(G['epoch_memory'][i], 1),
            'mem_saved_mb':   round(L['epoch_memory'][i] - G['epoch_memory'][i], 1),
            'lstm_val_acc':   round(L['history']['val_accuracy'][i]*100, 2),
            'gru_val_acc':    round(G['history']['val_accuracy'][i]*100, 2),
        })
    csv_path = os.path.join(plots_dir, 'benchmark_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  ✓ Full results saved → {csv_path}")

    # Summary CSV
    summary = {
        'metric':          ['Parameters','Avg Time/Epoch (s)','Total Time (s)',
                            'Peak Memory (MB)','Test Accuracy (%)'],
        'cnn_lstm':        [L['params'], round(L['avg_time'],3),
                            round(L['total_time'],1), round(L['peak_memory'],1),
                            round(L['test_acc'],2)],
        'cnn_gru':         [G['params'], round(G['avg_time'],3),
                            round(G['total_time'],1), round(G['peak_memory'],1),
                            round(G['test_acc'],2)],
        'saving':          [param_saved, round(time_saved,3),
                            round(total_saved,1), round(mem_saved,1),
                            round(acc_diff,2)],
        'saving_percent':  [round(param_pct,1), round(time_pct,1),
                            round(total_saved/L['total_time']*100,1),
                            round(mem_pct,1), '—'],
    }
    summary_path = os.path.join(plots_dir, 'benchmark_summary.csv')
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"  ✓ Summary saved      → {summary_path}")

    print(f"\n  KEY TAKEAWAY FOR PAPER:")
    print(f"  CNN+GRU saves {param_pct:.1f}% parameters, "
          f"{time_pct:.1f}% time/epoch, "
          f"{mem_pct:.1f}% peak memory")
    print(f"  with only {abs(acc_diff):.2f}% accuracy difference\n")


if __name__ == '__main__':
    main()
EOF