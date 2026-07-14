# SMS Classification using Hybrid CNN-LSTM Architecture

This project implements an on-device SMS classification system using a hybrid CNN-LSTM neural network architecture, inspired by the research paper **"On-Device Information Extraction from SMS using Hybrid Hierarchical Classification"**.

The system classifies Indian SMS messages into 8 balanced categories:
- **Transaction** - Banking transactions, transfers
- **OTP** - One-Time Passwords for authentication
- **Promotion** - Marketing and promotional messages
- **Bills** - Bill payments and invoices
- **Shopping** - E-commerce and shopping notifications
- **Food** - Food delivery and restaurant messages
- **Travel** - Travel bookings and confirmations
- **Personal** - Personal and miscellaneous messages

## Project Structure

```
sms_classifier/
│
├── dataset/
│   └── indian_sms_dataset.csv          # Input dataset with SMS texts and labels
│
├── models/
│   ├── model.py                         # CNN-LSTM architecture definition
│   └── saved/                           # Directory for trained models
│       ├── best_model.keras             # Best model (saved during training)
│       ├── final_model.keras            # Final trained model
│       └── history.pkl                  # Training history
│
├── preprocessing/
│   └── preprocess.py                    # Text preprocessing and tokenization
│
├── plots/                               # Directory for evaluation plots
│   ├── confusion_matrix.png             # Confusion matrix heatmap
│   └── training_history.png             # Training/validation curves
│
├── preprocessors/
│   └── preprocessor.pkl                 # Saved tokenizer and label encoder
│
├── train.py                             # Training script
├── evaluate.py                          # Evaluation and metrics
├── predict.py                           # Interactive prediction script
├── utils.py                             # Utility functions
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (already exists as `torch_gpu`)

### Setup

1. **Install dependencies** in the existing `torch_gpu` environment:

```bash
# Navigate to project directory
cd sms_classifier

# Install required packages (if not already installed)
pip install tensorflow numpy pandas scikit-learn matplotlib keras
```

2. **Place the dataset**:

Ensure `indian_sms_dataset.csv` is in the `dataset/` folder with columns:
- `text` - SMS message content
- `label` - SMS category (one of the 8 classes)

## Usage

### 1. Training

Train the model on the SMS dataset:

```bash
python train.py
```

**What happens:**
- Loads the dataset from `dataset/indian_sms_dataset.csv`
- Splits into 80% training, 10% validation, 10% testing
- Preprocesses text (cleaning, tokenization, padding)
- Builds the CNN-LSTM model
- Trains for 20 epochs with batch size 32
- Uses callbacks:
  - **EarlyStopping**: Stops if validation loss doesn't improve for 5 epochs
  - **ReduceLROnPlateau**: Reduces learning rate if validation loss plateaus
  - **ModelCheckpoint**: Saves the best model based on validation accuracy
- Saves model, preprocessor, and training history

**Expected output:**
```
============================================================
SMS CLASSIFICATION - TRAINING
============================================================

[1/5] Loading dataset...
✓ Loaded 1000 samples from dataset/indian_sms_dataset.csv
✓ Classes: ['Bills' 'Food' 'OTP' 'Personal' 'Promotion' 'Shopping' 'Transaction' 'Travel']
✓ Class distribution:
Bills           125
Food            125
...

[2/5] Splitting dataset...
✓ Dataset split:
  - Training: 800 samples (80.0%)
  - Validation: 100 samples (10.0%)
  - Testing: 100 samples (10.0%)

[3/5] Preprocessing data...
✓ Vocabulary size: 2845
✓ Number of classes: 8
✓ Max sequence length: 100

[4/5] Building model...
Model: "sequential"
...

[5/5] Training model...
Epoch 1/20
25/25 [==============================] - 2s 65ms/step - loss: 2.0789 - accuracy: 0.1375 - val_loss: 1.9354 - val_accuracy: 0.25
...
```

### 2. Evaluation

Evaluate the trained model and generate metrics:

```bash
python evaluate.py
```

**What happens:**
- Loads the best trained model
- Loads the preprocessor
- Evaluates on test set
- Computes and displays:
  - **Accuracy**: Overall correctness
  - **Precision**: True positives / (True positives + False positives)
  - **Recall**: True positives / (True positives + False negatives)
  - **F1 Score**: Harmonic mean of precision and recall
  - **Classification Report**: Per-class metrics
- Generates plots:
  - **Confusion Matrix**: Shows classification performance per class
  - **Training History**: Shows accuracy and loss over epochs

**Expected output:**
```
============================================================
SMS CLASSIFICATION - EVALUATION
============================================================

[1/5] Loading model...
✓ Model loaded from models/saved/best_model.h5

[2/5] Loading preprocessor...
✓ Preprocessor loaded

[3/5] Preparing test data...
✓ Test set prepared: 100 samples

[4/5] Evaluating model...

Metric               Score     
------------------------------
Accuracy             0.9800
Precision            0.9812
Recall               0.9800
F1 Score             0.9805

CLASSIFICATION REPORT:
              precision    recall  f1-score   support

        Bills       1.00      1.00      1.00        12
         Food       1.00      0.92      0.96        12
          OTP       1.00      1.00      1.00        13
      Personal       0.92      1.00      0.96        12
    Promotion       1.00      1.00      1.00        13
     Shopping       0.93      0.93      0.93        15
   Transaction       1.00      1.00      1.00        13
       Travel       1.00      1.00      1.00        17

    accuracy                           0.98       107
   macro avg       0.98      0.98      0.98       107
weighted avg       0.98      0.98      0.98       107

[5/5] Generating plots...
✓ Confusion matrix saved to plots/confusion_matrix.png
✓ Training history plot saved to plots/training_history.png
```

### 3. Interactive Prediction

Classify new SMS messages in real-time:

```bash
python predict.py
```

**Usage:**
```
============================================================
SMS CLASSIFICATION - INTERACTIVE PREDICTION
============================================================

Type an SMS to classify. Type 'quit' or 'exit' to exit.

Enter SMS: Rs.500 debited from SBI account

Predicted Class: Transaction
Confidence: 98.73%

Enter SMS: OTP: 123456 - Valid for 10 minutes

Predicted Class: OTP
Confidence: 99.45%

Enter SMS: quit

Thank you for using SMS Classifier!
```

**Example SMS inputs:**

| SMS Text | Expected Class |
|----------|-----------------|
| Rs.500 debited from SBI account | Transaction |
| OTP: 123456 Valid for 10 mins | OTP |
| 50% off on all items this weekend | Promotion |
| Bill amount Rs.5000 due on 31st | Bills |
| Order delivered! Track here | Shopping |
| Pizza at 50% off. Order now | Food |
| Flight booking confirmed. PNR: ABC123 | Travel |
| Hey, how are you? | Personal |

### 4. Full Finance Tracker Pipeline (integrated from sms_classifier_lstm_cnn)

Beyond the single 8-class classifier above, this project now includes the
end-to-end pipeline ported from the `sms_classifier_lstm_cnn` project:

```
SMS text
  → regex preprocessing (preprocessing/preprocess.py — SMSPreprocessor.clean_text)
  → Binary Transaction / Non-Transaction decision
    (derived from this project's CNN+GRU model, models/saved/best_model.keras)
  → if Non-Transaction: return immediately (same response shape as the
    original LSTM+CNN project)
  → if Transaction:
      → regex entity extraction (transaction_extractor.py)
      → CNN+GRU Transaction Subcategory classification
        (models/saved/fintech_model.keras, trained by train_fintech.py)
```

Run it with:

```bash
# Interactive
python predict.py --pipeline

# Batch (reads a 'text' column from a CSV)
python predict.py --pipeline --batch dataset/real_sms.csv
```

The subcategory model is trained separately — if `models/saved/fintech_model.keras`
doesn't exist yet, run `python train_fintech.py` first. The original single-model
`python predict.py` (no `--pipeline` flag) still works exactly as before.

New files added for this integration:
- `transaction_extractor.py` — regex-based entity extraction (amount, date,
  account, bank, beneficiary, etc.), reused unchanged from the LSTM+CNN project.
- `spend_classifier.py` — rule + ML hybrid Transaction Subcategory classifier,
  reused from the LSTM+CNN project but now backed by this project's own
  CNN+GRU `fintech_model.keras` instead of the CNN+LSTM one.
- `FinanceTrackerPipeline` class (in `predict.py`) — orchestrates the full
  binary → subcategory workflow described above.

## Model Architecture

The hybrid CNN-LSTM architecture combines convolutional and recurrent layers:

```
Input (Sequence of integers)
        ↓
Embedding Layer (128 dimensions)
        ↓
Conv1D (128 filters, kernel size 5) + ReLU
        ↓
MaxPooling1D (pool size 2)
        ↓
LSTM (128 units, stateless)
        ↓
Dropout (50%)
        ↓
Dense (64 units) + ReLU
        ↓
Dropout (30%)
        ↓
Dense (8 units) + Softmax
        ↓
Output (Class probabilities)
```

**Why this architecture?**
- **Embedding**: Converts word indices to dense vectors for semantic meaning
- **Conv1D**: Captures local patterns and short dependencies in SMS text
- **LSTM**: Captures long-term dependencies and sequential information
- **Dropout**: Prevents overfitting by randomly dropping neurons
- **Dense layers**: Learns complex non-linear relationships
- **Softmax**: Outputs probability distribution over 8 classes

## Preprocessing Details

The preprocessing pipeline:

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs and email addresses
   - Remove extra whitespace
   - Preserve important symbols: ₹, Rs., numbers, letters, X (account masks)
   - Remove most punctuation except periods and hyphens

2. **Tokenization**:
   - Convert text to sequences of word indices
   - Vocabulary size: 5,000 most common words
   - Unknown words mapped to `<UNK>` token

3. **Padding**:
   - Pad sequences to fixed length: 100 tokens
   - Short sequences padded with zeros
   - Long sequences truncated

4. **Label Encoding**:
   - Convert class labels to integer indices (0-7)
   - Preserved in LabelEncoder for inference

## Training Configuration

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss**: SparseCategoricalCrossentropy
- **Metric**: Accuracy
- **Epochs**: 20 (with early stopping)
- **Batch Size**: 32
- **Train/Val/Test Split**: 80/10/10
- **Random Seed**: 42 (for reproducibility)

## Callbacks

1. **EarlyStopping**:
   - Monitor: validation loss
   - Patience: 5 epochs
   - Restores best weights

2. **ReduceLROnPlateau**:
   - Monitor: validation loss
   - Factor: 0.5 (multiply learning rate by 0.5)
   - Patience: 3 epochs
   - Min learning rate: 1e-7

3. **ModelCheckpoint**:
   - Monitor: validation accuracy
   - Saves only the best model

## Expected Performance

On the synthetic Indian SMS dataset:

- **Test Accuracy**: 97-99%
- **Precision**: 97-99% (weighted average)
- **Recall**: 97-99% (weighted average)
- **F1 Score**: 97-99%

The high performance is expected because:
- The dataset is well-balanced (equal samples per class)
- Classes are well-separated semantically
- The synthetic data doesn't contain the noise of real-world SMS
- The model is specifically designed for SMS classification

## Limitations of Synthetic Data

While the model performs excellently on this synthetic dataset, real-world deployment faces challenges:

### Dataset Limitations:

1. **Perfect formatting**: Real SMS contain typos, abbreviations, mixed languages
2. **Balanced distribution**: Real-world SMS have unbalanced class distributions
3. **Structured messages**: Synthetic SMS follow patterns; real SMS are chaotic
4. **No slang/colloquialisms**: Missing regional language and informal language
5. **No context**: Real SMS may be ambiguous without conversation history
6. **No special characters**: Missing emojis, special symbols common in real SMS
7. **Language simplicity**: Real SMS contain Hinglish, code-switching, abbreviations

### Model Limitations for Real-World Use:

1. **Domain shift**: Performance drops when tested on truly unseen SMS patterns
2. **Class imbalance**: Real data has skewed class distributions
3. **Temporal changes**: New SMS patterns emerge over time
4. **Language evolution**: Slang and abbreviations change constantly
5. **Out-of-vocabulary words**: Many real-world terms won't be in training vocabulary
6. **Multi-language**: Indian SMS often mix English and Indian languages
7. **Context dependency**: Some SMS require conversation context for classification

### Recommended Improvements for Real-World Deployment:

1. **Data collection**: Gather real SMS from diverse sources
2. **Augmentation**: Use data augmentation (back-translation, synonym replacement)
3. **Class balancing**: Use techniques like oversampling, undersampling, SMOTE
4. **Transfer learning**: Pre-train on large SMS corpus
5. **Ensemble methods**: Combine multiple models
6. **Active learning**: Continuously improve with hard examples
7. **Regular retraining**: Update model periodically with new data
8. **Multi-language support**: Include models for different languages
9. **Human-in-the-loop**: Review and correct misclassifications
10. **Uncertainty sampling**: Flag low-confidence predictions

## File Descriptions

### `train.py`
Main training script. Orchestrates:
- Dataset loading and splitting
- Text preprocessing
- Model building
- Training with callbacks
- Model and preprocessor saving

### `evaluate.py`
Evaluation script that:
- Loads trained model and preprocessor
- Computes accuracy, precision, recall, F1 score
- Generates classification report
- Plots confusion matrix
- Plots training history curves

### `predict.py`
Interactive prediction script with:
- `SMSClassifier` class for inference
- `interactive_predict()` function for user input
- Batch prediction capability
- Confidence score display

### `preprocessing/preprocess.py`
Text preprocessing module with:
- `SMSPreprocessor` class
- Text cleaning
- Tokenization and padding
- Label encoding
- Serialization/deserialization

### `models/model.py`
Model architecture definition with:
- `build_cnn_lstm_model()` function
- Hybrid CNN-LSTM architecture
- Model compilation with Adam optimizer
- `get_model_summary()` helper function

### `utils.py`
Utility functions:
- `set_random_seed()` - Set TensorFlow and NumPy seeds
- `save_object()` - Pickle saving
- `load_object()` - Pickle loading
- `create_directories()` - Directory creation
- `ensure_directory()` - Ensure parent directories exist

## Reproducibility

The project ensures reproducibility by:

1. **Fixed random seed**: 42 set for all random operations
2. **Deterministic operations**: `TF_DETERMINISTIC_OPS=1`
3. **Stratified splits**: Maintain class distribution in train/val/test splits
4. **Saved preprocessor**: Ensures same text processing for inference
5. **Model checkpointing**: Saves best model for consistent results

## Performance Optimization

For production deployment:

1. **Model quantization**: Convert to TFLite for mobile devices
2. **Model pruning**: Remove unnecessary weights
3. **Model distillation**: Train smaller model from larger one
4. **Caching**: Cache tokenizer vocabulary and embedding matrix
5. **Batch inference**: Process multiple SMS efficiently

## Troubleshooting

### Issue: "FileNotFoundError: Dataset not found"
**Solution**: Ensure `indian_sms_dataset.csv` is in `dataset/` folder

### Issue: "Model not found" during evaluation or prediction
**Solution**: Run `python train.py` first to train the model

### Issue: "ModuleNotFoundError: tensorflow"
**Solution**: Install TensorFlow: `pip install tensorflow`

### Issue: Low accuracy on custom SMS
**Possible causes**:
- SMS contains unseen words (out-of-vocabulary)
- SMS format differs from training data
- Class boundary is ambiguous
- Model needs retraining on new data

### Issue: Out of memory during training
**Solution**:
- Reduce batch size: `python train.py --batch-size 16`
- Use fewer epochs: `python train.py --epochs 10`
- Reduce max vocabulary size in `preprocess.py`

## References

- **Paper**: "On-Device Information Extraction from SMS using Hybrid Hierarchical Classification"
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **Scikit-learn**: https://scikit-learn.org/
- **SMS Dataset**: Balanced Indian SMS corpus with 8 categories

## License

This project is created for educational and research purposes.

## Author

SMS Classification Project - 2024

---

**Questions or Issues?** Check the troubleshooting section or review the code comments.
