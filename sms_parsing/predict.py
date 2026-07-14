"""
Prediction script for SMS classification using TensorFlow/Keras.
Allows users to input SMS and get predictions.

This file now also hosts the full Finance Tracker pipeline
(FinanceTrackerPipeline, below), integrated from the sms_classifier_lstm_cnn
project:

  Step 1 — Regex preprocessing (SMSPreprocessor.clean_text, shared/identical
            across both projects — see preprocessing/preprocess.py)
  Step 2 — Binary Transaction vs Non-Transaction decision, derived from this
            project's own CNN+GRU multi-class model (models/saved/best_model.keras,
            trained by train.py on indian_sms_dataset.csv). This mirrors the
            exact binary-decision workflow used in the LSTM+CNN project's
            predict.py: the underlying model classifies among 8 SMS
            categories, and "is_transaction" is derived as
            (sms_category == "Transaction").
  Step 3 — If Transaction: regex entity extraction (transaction_extractor.py)
            + CNN+GRU Transaction Subcategory classification (spend_classifier.py,
            backed by models/saved/fintech_model.keras — trained by
            train_fintech.py using build_cnn_gru_model()).
  Step 4 — If Non-Transaction: short-circuit with the same response shape
            used in the original LSTM+CNN project (no entity extraction,
            no subcategory classification).

The original single-model SMSClassifier class below is preserved unchanged
so any existing code that imports it keeps working exactly as before.
"""

import os
import sys
import json
import argparse
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_object
from preprocessing.preprocess import SMSPreprocessor
class SMSClassifier:
    """
    Wrapper class for SMS classification inference using TensorFlow/Keras.
    """
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize classifier with model and preprocessor.
        
        Args:
            model_path: Path to trained model (uses default if None)
            preprocessor_path: Path to preprocessor (uses default if None)
        
        Raises:
            FileNotFoundError: If model or preprocessor files don't exist
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            model_path = os.path.join(script_dir, 'models', 'saved', 'best_model.keras')
        if preprocessor_path is None:
            preprocessor_path = os.path.join(script_dir, 'preprocessors', 'preprocessor.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        # Load preprocessor
        self.preprocessor: SMSPreprocessor = load_object(preprocessor_path)
        self.model = tf.keras.models.load_model(model_path)
        
        print("✓ Model and preprocessor loaded successfully")
    
    def predict(self, sms_text: str) -> Tuple[str, float]:
        """
        Predict class for input SMS.
        
        Args:
            sms_text: Input SMS text
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        try:
            # Preprocess
            X, _ = self.preprocessor.transform([sms_text])
            probs = self.model.predict(X, verbose=0)[0]

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx]) * 100
            
            predicted_class = self.preprocessor.classes_[pred_idx]
            
            return predicted_class, confidence
        
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            raise
    
    def predict_batch(self, sms_texts: list) -> list:
        """
        Predict classes for multiple SMS.
        
        Args:
            sms_texts: List of SMS texts
        
        Returns:
            List of tuples (predicted_class, confidence)
        """
        results = []
        for sms in sms_texts:
            pred_class, confidence = self.predict(sms)
            results.append((pred_class, confidence))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# FinanceTrackerPipeline — integrated Binary + Subcategory workflow
# (ported from sms_classifier_lstm_cnn/predict.py)
# ─────────────────────────────────────────────────────────────────────────────
from transaction_extractor import TransactionExtractor
from spend_classifier import SpendClassifier


class FinanceTrackerPipeline:
    """
    Full pipeline, CNN+GRU project version:

        SMS text
          → regex preprocessing (SMSPreprocessor.clean_text)
          → Binary Transaction / Non-Transaction decision
            (derived from the CNN+GRU multi-class model — best_model.keras)
          → entity extraction (if Transaction)          [regex, transaction_extractor.py]
          → CNN+GRU subcategory classification (if Transaction)
            [fintech_model.keras, via spend_classifier.py]

    This is a drop-in equivalent of the LSTM+CNN project's
    FinanceTrackerPipeline — same steps, same output shape — with the
    CNN+GRU architecture used for both the main SMS classifier and the
    subcategory classifier instead of CNN+LSTM.
    """

    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if model_path is None:
            model_path = os.path.join(
                script_dir, 'models', 'saved', 'best_model.keras')
        if preprocessor_path is None:
            preprocessor_path = os.path.join(
                script_dir, 'preprocessors', 'preprocessor.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Main model not found: {model_path}\n"
                "Run: python train.py")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found: {preprocessor_path}\n"
                "Run: python train.py")

        # Main SMS classifier — the CNN+GRU multi-class model. It is
        # trained on 8 categories (Transaction/OTP/Promotion/Bills/
        # Shopping/Food/Travel/Personal); the binary Transaction vs
        # Non-Transaction decision is derived from its output, exactly
        # as in the LSTM+CNN project.
        self.preprocessor: SMSPreprocessor = load_object(preprocessor_path)
        self.model = tf.keras.models.load_model(model_path)
        print("✓ Main CNN+GRU SMS classifier loaded")

        # Entity extractor (no model needed — pure regex)
        self.extractor = TransactionExtractor()

        # Subcategory classifier (rules + CNN+GRU fintech model)
        self.spend_clf = SpendClassifier(use_ml=True)

    # ── Core predict ──────────────────────────────────────────────────────
    def predict(self, sms: str) -> dict:
        """
        Full pipeline prediction for one SMS.

        Returns a dict:
        {
            "sms_category":   "Transaction",
            "sms_confidence": 99.91,
            "is_transaction": True,
            "entities": {
                "transaction_type": "Debit",
                "amount":           12760.0,
                "currency":         "INR",
                "date":             "17-06-26",
                "account":          "XX2428",
                "beneficiary":      "VIK N",
                "upi_ref":          "99884791743",
                "ref_number":       None,
                "bank":             "KarnatakaBank",
                "balance":          13592.47
            },
            "spend_category":   "Personal Transfer",
            "spend_confidence": 0.60,
            "spend_method":     "heuristic"
        }
        """
        # ── Step 1 & 2: Regex preprocessing + Binary decision ──────────────
        # SMSPreprocessor.transform() applies clean_text() (the shared
        # regex preprocessing) before tokenizing/padding for the model.
        X, _     = self.preprocessor.transform([sms])
        probs    = self.model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        sms_cat  = self.preprocessor.classes_[pred_idx]
        sms_conf = float(probs[pred_idx]) * 100

        result = {
            "sms_category":     sms_cat,
            "sms_confidence":   round(sms_conf, 2),
            "is_transaction":   sms_cat == "Transaction",
            "entities":         None,
            "spend_category":   None,
            "spend_confidence": None,
            "spend_method":     None,
        }

        # ── Step 3: Only for Transaction SMS ──────────────────────────────
        if sms_cat == "Transaction":
            entities = self.extractor.extract(sms)
            result["entities"] = entities

            spend_cat, spend_conf, method = self.spend_clf.classify(
                sms, entities)
            result["spend_category"]   = spend_cat
            result["spend_confidence"] = round(spend_conf * 100, 1)
            result["spend_method"]     = method

        # Step 4 (Non-Transaction) is implicit: entities/spend_* stay None,
        # matching the original LSTM+CNN project's response shape exactly.
        return result

    def predict_batch(self, sms_list: list) -> list:
        return [self.predict(sms) for sms in sms_list]


def _print_pipeline_result(result: dict, idx: int = None):
    """Pretty print a single FinanceTrackerPipeline result."""
    prefix = f"[{idx}] " if idx is not None else ""
    print(f"\n{prefix}{'─'*60}")
    print(f"  SMS Category   : {result['sms_category']} "
          f"({result['sms_confidence']:.1f}%)")

    if result['is_transaction']:
        e = result['entities']
        print(f"  ── Entities ────────────────────────────────────")
        print(f"  Type           : {e.get('transaction_type', '—')}")
        print(f"  Amount         : ₹{e.get('amount'):,.2f}" if e.get('amount') else "  Amount         : —")
        print(f"  Date           : {e.get('date') or '—'}")
        print(f"  Account        : {e.get('account') or '—'}")
        print(f"  Beneficiary    : {e.get('beneficiary') or '—'}")
        print(f"  Bank           : {e.get('bank') or '—'}")
        print(f"  UPI Ref        : {e.get('upi_ref') or '—'}")
        print(f"  Balance After  : ₹{e.get('balance'):,.2f}" if e.get('balance') else "  Balance After  : —")
        print(f"  ── Subcategory (CNN+GRU) ───────────────────────")
        print(f"  Category       : {result['spend_category']} "
              f"({result['spend_confidence']}%) "
              f"[via {result['spend_method']}]")
    else:
        print(f"  Not a transaction — no entity extraction needed.")
    print(f"  {'─'*58}")


def pipeline_interactive_mode(pipeline: FinanceTrackerPipeline):
    print("\n" + "="*60)
    print("  CNN+GRU FINANCE TRACKER — INTERACTIVE MODE")
    print("  Type an SMS and press Enter. Type 'quit' to exit.")
    print("="*60)

    while True:
        try:
            sms = input("\nEnter SMS: ").strip()
            if not sms:
                continue
            if sms.lower() in ('quit', 'exit', 'q'):
                print("Bye!")
                break
            result = pipeline.predict(sms)
            _print_pipeline_result(result)
        except KeyboardInterrupt:
            print("\nBye!")
            break


def pipeline_batch_mode(pipeline: FinanceTrackerPipeline, csv_path: str):
    """
    Read a CSV with a 'text' column, predict all rows, save results.
    """
    print(f"\n[Batch] Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    if 'text' not in df.columns:
        raise ValueError("CSV must have a 'text' column")

    results = pipeline.predict_batch(df['text'].tolist())

    rows = []
    for sms, r in zip(df['text'], results):
        e = r.get('entities') or {}
        rows.append({
            "text":             sms,
            "sms_category":     r['sms_category'],
            "sms_confidence":   r['sms_confidence'],
            "is_transaction":   r['is_transaction'],
            "spend_category":   r.get('spend_category', ''),
            "spend_confidence": r.get('spend_confidence', ''),
            "spend_method":     r.get('spend_method', ''),
            "txn_type":         e.get('transaction_type', ''),
            "amount":           e.get('amount', ''),
            "date":             e.get('date', ''),
            "account":          e.get('account', ''),
            "beneficiary":      e.get('beneficiary', ''),
            "bank":             e.get('bank', ''),
            "upi_ref":          e.get('upi_ref', ''),
            "balance":          e.get('balance', ''),
        })

    out_df   = pd.DataFrame(rows)
    out_path = csv_path.replace('.csv', '_predictions.csv')
    out_df.to_csv(out_path, index=False)

    print(f"\n✓ Predictions saved to: {out_path}")
    print(f"  Total SMS processed : {len(out_df)}")
    print(f"  Transactions found  : {out_df['is_transaction'].sum()}")

    txn_rows = out_df[out_df['is_transaction']]
    if not txn_rows.empty:
        print(f"\n  Subcategory Breakdown:")
        print(txn_rows['spend_category'].value_counts().to_string())

    return out_df


def interactive_predict(model_path: str = None, preprocessor_path: str = None) -> None:
    """
    Interactive prediction loop.
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
    """
    try:
        classifier = SMSClassifier(model_path, preprocessor_path)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("\nPlease ensure the model is trained first by running: python train.py")
        return
    
    print("\n" + "="*60)
    print("SMS CLASSIFICATION - INTERACTIVE PREDICTION (TensorFlow/Keras)")
    print("="*60)
    print("\nType an SMS to classify. Type 'quit' or 'exit' to exit.\n")
    
    while True:
        try:
            sms = input("Enter SMS: ").strip()
            
            if not sms:
                print("✗ Please enter a valid SMS\n")
                continue
            
            if sms.lower() in ['quit', 'exit']:
                print("\nThank you for using SMS Classifier!")
                break
            
            predicted_class, confidence = classifier.predict(sms)
            
            print(f"\nPredicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%\n")
        
        except KeyboardInterrupt:
            print("\n\nThank you for using SMS Classifier!")
            break
        except Exception as e:
            print(f"✗ Error: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CNN+GRU SMS Classifier / Finance Tracker Pipeline')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run the full Finance Tracker pipeline '
                             '(binary Transaction/Non-Transaction + '
                             'CNN+GRU subcategory classification) instead '
                             'of the single 8-class SMS classifier.')
    parser.add_argument('--batch', type=str, default=None,
                        help='Path to CSV file for batch prediction '
                             '(reads a "text" column).')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to main model file')
    parser.add_argument('--preprocessor', type=str, default=None,
                        help='Path to preprocessor file')
    args = parser.parse_args()

    if args.pipeline:
        # Full workflow: regex preprocessing -> binary decision ->
        # (if Transaction) entity extraction + CNN+GRU subcategory model.
        try:
            pipeline = FinanceTrackerPipeline(args.model, args.preprocessor)
        except FileNotFoundError as e:
            print(f"✗ {e}")
            sys.exit(1)

        if args.batch:
            pipeline_batch_mode(pipeline, args.batch)
        else:
            pipeline_interactive_mode(pipeline)
    else:
        # Original behaviour: single CNN+GRU 8-class SMS classifier,
        # unchanged from before this integration.
        interactive_predict(args.model, args.preprocessor)
