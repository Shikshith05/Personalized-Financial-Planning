"""
Evaluation script for SMS classification model using TensorFlow/Keras.
Computes metrics, generates plots, and confusion matrix.
"""

import os
import sys
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_object
from preprocessing.preprocess import SMSPreprocessor


def evaluate(
    model_path: str = None,
    preprocessor_path: str = None,
    history_path: str = None,
    dataset_path: str = None,
    output_dir: str = None
) -> dict:
    """
    Evaluate trained model on test set and generate plots.
    
    Args:
        model_path: Path to trained model (uses default if None)
        preprocessor_path: Path to preprocessor (uses default if None)
        history_path: Path to training history (uses default if None)
        dataset_path: Path to dataset CSV (uses default if None)
        output_dir: Directory to save plots (uses default if None)
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_path is None:
        model_path = os.path.join(script_dir, 'models', 'saved', 'best_model.keras')
    if preprocessor_path is None:
        preprocessor_path = os.path.join(script_dir, 'preprocessors', 'preprocessor.pkl')
    if history_path is None:
        history_path = os.path.join(script_dir, 'models', 'saved', 'history.pkl')
    if dataset_path is None:
        dataset_path = os.path.join(script_dir, 'dataset', 'indian_sms_dataset.csv')
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SMS CLASSIFICATION - EVALUATION (TensorFlow/Keras)")
    print("="*60)

    # Load model
    print("\n[1/5] Loading model...")
    if os.path.exists(model_path):
        pass
    else:
        alt_path = os.path.join(script_dir, 'models', 'saved', 'best_model.h5')
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            alt_path = os.path.join(script_dir, 'models', 'saved', 'best_model.keras')
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load preprocessor first to get config
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    preprocessor: SMSPreprocessor = load_object(preprocessor_path)
    config = preprocessor.get_config()

    # Load model
    model = tf.keras.models.load_model(model_path)

    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Preprocessor loaded")
    
    # Load and prepare test data
    print("\n[2/5] Preparing test data...")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['text', 'label'])
    
    # Use the same split as training (last 10%)
    from sklearn.model_selection import train_test_split
    _, test_texts = train_test_split(
        df['text'].values,
        test_size=0.1,
        random_state=42,
        stratify=df['label'].values
    )
    _, test_labels = train_test_split(
        df['label'].values,
        test_size=0.1,
        random_state=42,
        stratify=df['label'].values
    )
    
    X_test, y_test = preprocessor.transform(test_texts, test_labels)
    print(f"✓ Test set prepared: {len(X_test)} samples")
    
    # Evaluate model
    print("\n[3/5] Evaluating model...")
    y_pred_probs = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'Metric':<20} {'Score':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {accuracy:<10.4f}")
    print(f"{'Precision':<20} {precision:<10.4f}")
    print(f"{'Recall':<20} {recall:<10.4f}")
    print(f"{'F1 Score':<20} {f1:<10.4f}")
    
    # Classification report
    print("\nCLASSIFICATION REPORT:")
    print("-" * 80)
    print(classification_report(y_test, y_pred, target_names=preprocessor.classes_, zero_division=0))
    
    # Plot confusion matrix
    print("\n[4/5] Generating plots...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=preprocessor.classes_,
        yticklabels=preprocessor.classes_,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {cm_path}")
    plt.close()
    
    # Plot training history if available
    print("\n[5/5] Saving evaluation results...")
    if os.path.exists(history_path):
        history_dict = load_object(history_path)
        
        # Accuracy plot
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['loss'], label='Training Loss', linewidth=2)
        plt.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {history_plot_path}")
        plt.close()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }


if __name__ == "__main__":
    metrics = evaluate()
