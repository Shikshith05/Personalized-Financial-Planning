"""
spend_classifier.py  (CNN+GRU integration)
=====================================
Classifies what a Transaction SMS was spent on (the "subcategory" stage
of the pipeline).

Ported from the sms_classifier_lstm_cnn project's spend_classifier.py.
The rule-based layers (transaction-type shortcuts, UPI handle lookup,
merchant keyword matching, personal-transfer heuristic) are reused
unchanged. The ONLY thing swapped out is the ML model that backs
Rule 3 — it now loads this project's own CNN+GRU subcategory model
(models/saved/fintech_model.keras, trained by train_fintech.py using
build_cnn_gru_model from models/model.py) instead of the CNN+LSTM
model from the original project. Nothing about the CNN+GRU
architecture itself is touched here; this file only consumes the
already-trained artifact.

Strategy (rule-first, ML fills the gap for unseen merchants):
  1. Transaction type shortcuts (salary, EMI, ATM, refund, interest) ← FAST RULES
  2. UPI handle / merchant keyword lookup on beneficiary field        ← HIGH-CONFIDENCE LOOKUP
  3. CNN+GRU subcategory model trained on spend_category_dataset.csv  ← ML MODEL
     Works on full SMS text so even unknown merchants get classified
  4. UPI handle / merchant keyword lookup on full SMS text            ← FALLBACK
  5. Merchant name word matching                                     ← FALLBACK
  6. Personal transfer heuristic                                     ← LAST RESORT

Place at: sms_classifier_cnn_gru_v9_final/spend_classifier.py
"""

import os, re, sys, pickle
from typing import Tuple, Optional
import numpy as np

# ── Known UPI handle → category (fallback only) ───────────────────────────────
# Unchanged from the LSTM+CNN project — these categories match the CNN+GRU
# project's own spend_category_dataset.csv label set exactly (Food & Dining,
# Shopping, Travel, Healthcare, Education, Utilities, Investment, ...).
UPI_HANDLE_MAP = {
    "swiggy": "Food & Dining",    "zomato": "Food & Dining",
    "dominos": "Food & Dining",   "kfc": "Food & Dining",
    "mcdonalds": "Food & Dining", "pizzahut": "Food & Dining",
    "faasos": "Food & Dining",    "box8": "Food & Dining",
    "flipkart": "Shopping",       "amazon": "Shopping",
    "myntra": "Shopping",         "meesho": "Shopping",
    "ajio": "Shopping",           "nykaa": "Shopping",
    "zudio": "Shopping",          "decathlon": "Shopping",
    "irctc": "Travel",            "ola": "Travel",
    "uber": "Travel",             "rapido": "Travel",
    "redbus": "Travel",           "makemytrip": "Travel",
    "cleartrip": "Travel",        "blusmart": "Travel",
    "pvr": "Entertainment",       "inox": "Entertainment",
    "bookmyshow": "Entertainment","netflix": "Entertainment",
    "spotify": "Entertainment",   "hotstar": "Entertainment",
    "apollo": "Healthcare",       "practo": "Healthcare",
    "pharmeasy": "Healthcare",    "netmeds": "Healthcare",
    "1mg": "Healthcare",          "thyrocare": "Healthcare",
    "medplus": "Healthcare",      "fortis": "Healthcare",
    "manipal": "Healthcare",      "wellness": "Healthcare",
    "pharmacy": "Healthcare",     "lal path": "Healthcare",
    "byjus": "Education",         "unacademy": "Education",
    "vedantu": "Education",       "udemy": "Education",
    "bescom": "Utilities",        "tneb": "Utilities",
    "tatapower": "Utilities",     "airtel": "Utilities",
    "jio": "Utilities",           "actfibernet": "Utilities",
    "zerodha": "Investment",      "groww": "Investment",
    "upstox": "Investment",       "kuvera": "Investment",
}

MERCHANT_WORD_MAP = {
    "Shopping":      ["technology", "tech", "electronics", "digital", "mobile",
                      "fashion", "clothing", "textiles", "garments", "retail",
                      "store", "shop", "mart", "centre", "center", "traders",
                      "zudio", "westside", "pantaloons", "lifestyle"],
    "Food & Dining": ["restaurant", "hotel", "cafe", "kitchen", "foods",
                      "catering", "bakery", "sweets", "biryani", "dhaba",
                      "mess", "canteen", "eatery", "tiffin", "darshini"],
    "Healthcare":    ["hospital", "clinic", "medical", "pharmacy", "health",
                      "diagnostic", "lab", "dental", "nursing", "care"],
    "Travel":        ["travels", "travel", "tours", "cab", "taxi",
                      "transport", "petrol", "fuel", "fastag", "ksrtc"],
    "Education":     ["school", "college", "academy", "institute", "coaching",
                      "classes", "tutorials", "university", "fees"],
    "Utilities":     ["electricity", "power", "energy", "telecom",
                      "broadband", "networks", "communications"],
}


class SpendClassifier:
    """
    Transaction Subcategory classifier.

    This is the same rule + ML hybrid design as the LSTM+CNN project's
    SpendClassifier, but Rule 3 now calls this project's CNN+GRU model
    (models/saved/fintech_model.keras) instead of the CNN+LSTM one.
    """

    def __init__(self, use_ml: bool = True):
        self.use_ml    = use_ml
        self._model    = None
        self._prep     = None
        self._classes  = None
        if use_ml:
            self._load_model()

    def _load_model(self):
        try:
            import tensorflow as tf
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # These artifacts are produced by THIS project's train_fintech.py,
            # which calls build_cnn_gru_model() from models/model.py —
            # i.e. the CNN+GRU architecture, unchanged.
            model_path = os.path.join(script_dir, 'models', 'saved', 'fintech_model.keras')
            prep_path  = os.path.join(script_dir, 'preprocessors', 'fintech_preprocessor.pkl')

            if os.path.exists(model_path) and os.path.exists(prep_path):
                self._model   = tf.keras.models.load_model(model_path)
                with open(prep_path, 'rb') as f:
                    self._prep = pickle.load(f)
                self._classes = self._prep.classes_
                print(f"✓ CNN+GRU subcategory model loaded | Classes: {list(self._classes)}")
            else:
                print("⚠ CNN+GRU subcategory model not found. Run: python train_fintech.py")
        except Exception as e:
            print(f"⚠ Could not load CNN+GRU subcategory model: {e}")

    def classify(self, sms: str, extracted: dict) -> Tuple[str, float, str]:
        """
        Returns (category, confidence 0-1, method_used)
        """
        text_lower = sms.lower()
        txn_type   = extracted.get('transaction_type', '')
        beneficiary= (extracted.get('beneficiary') or '').lower()

        # ── Rule 1: Hard transaction type shortcuts ────────────────────────
        if txn_type == 'Salary Credit':
            return 'Income / Salary', 1.0, 'txn_type'
        if txn_type == 'ATM Withdrawal':
            return 'Cash Withdrawal', 1.0, 'txn_type'
        if txn_type == 'Interest Credit':
            return 'Investment', 0.95, 'txn_type'
        if txn_type == 'Refund':
            return 'Refund', 1.0, 'txn_type'
        if txn_type == 'EMI Debit':
            return 'Loan & EMI', 0.95, 'txn_type'

        # ── Rule 2: High-confidence keyword match (runs BEFORE ML) ─────────
        # If the beneficiary clearly contains a known brand name, trust
        # this over the ML model — it's a more reliable signal.
        for keyword, category in UPI_HANDLE_MAP.items():
            if keyword in beneficiary:
                return category, 0.95, 'upi_lookup'

        # ── Rule 3: CNN+GRU ML model (for unrecognized merchants) ───────────
        if self.use_ml and self._model is not None:
            try:
                X, _ = self._prep.transform([sms])
                probs = self._model.predict(X, verbose=0)[0]
                idx   = int(np.argmax(probs))
                conf  = float(probs[idx])
                cat   = self._classes[idx]
                # Only trust ML if confidence > 55%
                if conf > 0.55:
                    return cat, conf, 'ml_model'
            except Exception as e:
                print(f"⚠ ML predict error: {e}")

        # ── Rule 4: UPI handle / merchant keyword in full SMS text ─────────
        for keyword, category in UPI_HANDLE_MAP.items():
            if keyword in text_lower:
                return category, 0.85, 'upi_lookup'

        # ── Rule 5: Merchant name word matching ────────────────────────────
        for category, keywords in MERCHANT_WORD_MAP.items():
            for kw in keywords:
                if kw in beneficiary:
                    return category, 0.80, 'merchant_name'

        # ── Rule 6: Personal transfer heuristic ───────────────────────────
        if beneficiary and re.match(r'^[a-z\s]{3,25}$', beneficiary):
            return 'Personal Transfer', 0.65, 'heuristic'

        return 'Unknown', 0.0, 'unknown'
