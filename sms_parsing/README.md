# SMS Parsing & Classification using Hybrid CNN-GRU

On-device SMS understanding for the **Personalized Financial Planning** project. Messy bank/UPI SMS go in; categorised, structured transactions come out. The core model is a hybrid **CNN + GRU** network, adapted from the CNN-LSTM design in the paper **"On-Device Information Extraction from SMS using Hybrid Hierarchical Classification"** (Vatsal et al., arXiv:2002.02755), with the LSTM replaced by a GRU for fewer parameters and faster training.

## What this module does

Two CNN-GRU models work together in a hierarchical pipeline:

| Model | Task | Classes | Trained by |
|---|---|---|---|
| **Main SMS classifier** | Categorise any SMS | 8: Transaction, OTP, Promotion, Bills, Shopping, Food, Travel, Personal | `train.py` |
| **Spend-category classifier** | Subcategorise Transaction SMS | 10: Food & Dining, Shopping, Travel, Utilities, Investment, Loan & EMI, Healthcare, Education, Entertainment, Personal Transfer | `train_fintech.py` |

## Pipeline

```
SMS text
  → regex preprocessing (preprocessing/preprocess.py)
  → Binary decision: Transaction / Non-Transaction
    (derived from the main CNN+GRU model, models/saved/best_model.keras)
  → if Non-Transaction: return immediately
  → if Transaction:
      → regex entity extraction (transaction_extractor.py)
         amount, date, account, bank, beneficiary, ...
      → CNN+GRU spend-category classification
         (models/saved/fintech_model.keras, with rule + ML hybrid in spend_classifier.py)
```

## Project Structure

```
sms_parsing/
│
├── dataset/
│   ├── indian_sms_dataset.csv          # Main 8-class dataset
│   ├── hard_realistic_augment.csv      # Harder, more realistic samples (merged in train.py)
│   ├── spend_category_dataset.csv      # 10-class spend-category dataset
│   ├── fintech_sms_dataset3.csv        # Additional fintech SMS data
│   └── real_sms.csv                    # Real-world SMS for testing
│
├── models/
│   ├── model.py                        # CNN-GRU architecture definition
│   └── saved/                          # Trained models
│       ├── best_model.keras            # Main classifier (best val accuracy)
│       ├── final_model.keras           # Main classifier (final epoch)
│       ├── fintech_model.keras         # Spend-category model (best)
│       ├── fintech_model_final.keras   # Spend-category model (final)
│       └── history.pkl                 # Training history
│
├── preprocessing/
│   └── preprocess.py                   # Text cleaning, tokenisation, padding
│
├── preprocessors/
│   ├── preprocessor.pkl                # Tokenizer + label encoder (main model)
│   └── fintech_preprocessor.pkl        # Tokenizer + label encoder (spend model)
│
├── plots/                              # Evaluation plots and metrics
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── training_metrics.png
│   └── epoch_metrics.csv
│
├── train.py                            # Train the main 8-class classifier
├── train_fintech.py                    # Train the 10-class spend-category model
├── evaluate.py                         # Evaluation and metrics
├── predict.py                          # Interactive / batch prediction + full pipeline
├── transaction_extractor.py            # Regex-based entity extraction
├── spend_classifier.py                 # Rule + ML hybrid spend classifier
├── benchmark_comparison.py             # CNN+LSTM vs CNN+GRU benchmark
├── diagnose_gru.py                     # Gradient diagnostics for the GRU model
├── utils.py                            # Helper functions
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.11 recommended (the Dockerfile uses `python:3.11-slim`)
- A virtual environment is recommended

### Setup

```bash
cd sms_parsing
pip install -r requirements.txt
```

Pinned versions: `tensorflow==2.17.0`, `keras==3.5.0`, `numpy==2.0.2`, `pandas==2.2.3`, `scikit-learn==1.5.2`, `matplotlib==3.9.2`, `seaborn==0.13.2`.

### Docker (optional)

```bash
docker build -t sms-parsing .
docker run -it sms-parsing        # starts the interactive predictor
```

## Model Architecture

Defined in `models/model.py` as `build_cnn_gru_model()`:

```
Input (token IDs, length 32)
        ↓
Embedding (128 dimensions)
        ↓
Conv1D (128 filters, kernel size 5, ReLU, padding="same")
        ↓
MaxPooling1D (pool size 2)
        ↓
LayerNormalization
        ↓
GRU (128 units, return_sequences=True)
        ↓
GlobalMaxPooling1D
        ↓
Dropout
        ↓
Dense (64 units) + ReLU
        ↓
Dropout (30%)
        ↓
Dense (num_classes) + Softmax
```

**Why each layer?**
- **Embedding**: converts word indices to dense vectors.
- **Conv1D + MaxPool**: detects local keyword patterns such as "debited for", "trf to" and "OTP is", and halves the sequence length.
- **LayerNormalization**: keeps the unbounded ReLU output in a safe range so the GRU's sigmoid/tanh gates do not saturate. A saturated GRU has no separate cell state to fall back on, which causes the model to collapse to predicting the majority class.
- **GRU**: captures sequential context with two gates (update and reset) and no separate cell state.
- **GlobalMaxPooling1D**: pools over all timesteps instead of using only the last one. SMS are short and heavily padded, so the final timestep is mostly padding.
- **Dropout + Dense**: regularisation and a non-linear classification head.
- **Softmax**: class probabilities.

**Configuration used**

| Setting | Main classifier (`train.py`) | Spend classifier (`train_fintech.py`) |
|---|---|---|
| Max vocabulary | 5,000 | 8,000 |
| Sequence length | 32 | 32 |
| Dropout (after GRU) | 0.5 (default) | 0.4 |
| Output classes | 8 | 10 |
| Max epochs | 20 | 25 |
| Batch size | 32 | 32 |

### Why GRU instead of LSTM?

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (input, forget, output) | 2 (update, reset) |
| Recurrent params (128 units, 128-dim input) | 131,584 | 98,688 |
| Cell state | Separate | None (single hidden state) |

At these sizes the GRU saves about 32,896 parameters (~25%) in the recurrent layer, and trains faster with a lower memory footprint. `benchmark_comparison.py` measures parameters, time per epoch, memory and accuracy for both on your own machine.

GRU equations:

```
z  = sigmoid(Wz · [h_prev, x])        # update gate
r  = sigmoid(Wr · [h_prev, x])        # reset gate
h~ = tanh(W · [r * h_prev, x])        # candidate hidden state
h  = (1 - z) * h_prev + z * h~        # new hidden state
```

## Preprocessing

1. **Text cleaning**
   - Lowercase; remove URLs and email addresses; collapse extra whitespace
   - Preserve important symbols: ₹, Rs., numbers, letters, X (account masks)
   - Remove most punctuation except periods and hyphens
2. **Tokenisation**: words are mapped to indices with an `<OOV>` token for unknown words (vocabulary size 5,000 for the main model, 8,000 for the spend model)
3. **Padding / truncation** to 32 tokens
4. **Label encoding** with a saved `LabelEncoder` for inference

The fitted tokenizer and label encoder are pickled to `preprocessors/` so training and inference use identical preprocessing.

## Usage

### 1. Train the main classifier

```bash
python train.py
```

- Loads `indian_sms_dataset.csv` and `hard_realistic_augment.csv`
- Removes exact-duplicate messages
- Splits **80/10/10 grouped by message template** (see below)
- Trains with batch size 32 for up to 20 epochs
- Reports per-epoch time and memory, and compares against the CNN-LSTM baseline
- Saves the model, preprocessor, history and plots

### 2. Train the spend-category model

Run **after** `train.py`:

```bash
python train_fintech.py
```

- Trains on `spend_category_dataset.csv` (10 classes), removes duplicate texts, stratified 80/10/10 split
- Up to 25 epochs, batch size 32
- Saves `fintech_model.keras`, `fintech_model_final.keras` and `fintech_preprocessor.pkl`

### 3. Evaluate

```bash
python evaluate.py
```

Reports accuracy, precision, recall and F1, a per-class classification report, a confusion matrix and training curves.

### 4. Predict

```bash
# Single 8-class classifier, interactive
python predict.py

# Full pipeline (binary → entity extraction → spend category), interactive
python predict.py --pipeline

# Full pipeline on a CSV with a 'text' column
python predict.py --pipeline --batch dataset/real_sms.csv

# Custom model or preprocessor paths
python predict.py --model path/to/model.keras --preprocessor path/to/preprocessor.pkl
```

Example:

| SMS Text | Predicted Class |
|---|---|
| Rs.500 debited from SBI account | Transaction |
| OTP: 123456 Valid for 10 mins | OTP |
| 50% off on all items this weekend | Promotion |
| Bill amount Rs.5000 due on 31st | Bills |
| Flight booking confirmed. PNR: ABC123 | Travel |
| Hey, how are you? | Personal |

If `models/saved/fintech_model.keras` is missing, run `python train_fintech.py` before using `--pipeline`.

### 5. Benchmark and diagnostics

```bash
python benchmark_comparison.py    # CNN+LSTM vs CNN+GRU on your machine
python diagnose_gru.py            # per-layer gradient norms for the GRU model
```

## Training Configuration

- **Optimizer**: Adam (learning rate 0.001, `clipnorm=1.0` as a safeguard against exploding gradients)
- **Loss**: SparseCategoricalCrossentropy
- **Metric**: Accuracy
- **Random seed**: 42

**Callbacks**
1. **EarlyStopping**: monitors validation loss, patience 5, restores best weights
2. **ReduceLROnPlateau**: monitors validation loss, factor 0.5, patience 3, min LR 1e-7
3. **ModelCheckpoint**: monitors validation accuracy, saves only the best model

## Avoiding Data Leakage

Many SMS differ only in the amount, OTP or reference number, so a random split lets near-identical messages appear in both train and test and inflates accuracy. `train.py` therefore:

1. Drops exact-duplicate texts
2. Converts each message to a structural **template** by masking numbers, OTPs and reference IDs
3. Uses `StratifiedGroupKFold` to keep all messages with the same template in a single split while preserving class balance
4. Asserts that **no template is shared** across train, validation and test

Accuracy on this template-grouped test set (**91.67%** on the 8-class task) is lower than a naive random split would give, but a more honest estimate of real-world performance.

## Known Issues & Workarounds

- **oneDNN GRU bug (Windows CPU):** TensorFlow's oneDNN-accelerated GRU kernel can silently break gradient flow, which freezes loss and accuracy at the majority-class baseline. The training scripts set `TF_ENABLE_ONEDNN_OPTS=0` before importing TensorFlow.
- **Deterministic ops:** `TF_DETERMINISTIC_OPS=1` also breaks GRU gradient flow on CPU, so the scripts turn it back off after seeding. NumPy, Python and TensorFlow seeds are unaffected.
- `reset_after=False` is used in the GRU layer (classic GRU formulation, compatible with the parameter count above).

## Limitations

- Datasets are largely synthetic or template-based; real SMS contain typos, abbreviations, Hinglish, regional languages and emojis
- Out-of-vocabulary words are common in real messages
- Class distributions in the real world are imbalanced, unlike the training sets
- Some messages need conversation context to classify correctly
- Regional-language SMS are not yet supported

**Suggested improvements:** collect more real SMS, augment data, balance classes, pretrain on a larger SMS corpus, retrain periodically, flag low-confidence predictions for human review, and quantise to TFLite for on-device deployment.

## Troubleshooting

| Issue | Solution |
|---|---|
| `FileNotFoundError: Dataset not found` | Check the CSV files are in `dataset/` with `text` and `label` columns |
| `Model not found` | Run `python train.py` (and `python train_fintech.py` for `--pipeline`) |
| `ModuleNotFoundError: tensorflow` | `pip install -r requirements.txt` |
| Accuracy frozen at a constant value | Make sure `TF_ENABLE_ONEDNN_OPTS=0`, then run `python diagnose_gru.py` |
| Low accuracy on custom SMS | Unseen vocabulary or an unfamiliar format; retrain with similar examples |
| Out of memory | Reduce the batch size in the training function |

## References

- S. Vatsal et al., "On-Device Information Extraction from SMS using Hybrid Hierarchical Classification," arXiv:2002.02755, 2020.
- K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation," EMNLP, 2014. (GRU)
- TensorFlow / Keras: https://www.tensorflow.org/

## Part of

[Personalized Financial Planning](https://github.com/Shikshith05/Personalized-Financial-Planning), together with `expense_forecasting` and `explainable_rl_planner`.

## License

Created for educational and research purposes.
