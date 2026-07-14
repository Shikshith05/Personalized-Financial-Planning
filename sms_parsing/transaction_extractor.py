"""
transaction_extractor.py
========================
Extracts structured fields from a Transaction SMS using regex rules.

Extracted fields:
    - transaction_type : Debit / Credit / UPI / IMPS / NEFT / ATM / Salary etc.
    - amount           : Numeric value (float)
    - currency         : INR (always for Indian SMS)
    - date             : Extracted date string
    - account          : Masked account number
    - beneficiary      : Who received / sent the money
    - upi_ref          : UPI reference number
    - ref_number       : General reference / txn ID
    - bank             : Bank name detected
    - balance          : Available balance after transaction

Reused as-is from the sms_classifier_lstm_cnn project (pure regex,
no model dependency — works identically regardless of which neural
architecture is used downstream for classification).

Place this file at:
    sms_classifier_cnn_gru_v9_final/transaction_extractor.py
"""

import re
from typing import Optional


# ── Known Indian bank names ────────────────────────────────────────────────
KNOWN_BANKS = [
    "SBI", "HDFC Bank", "HDFC", "ICICI Bank", "ICICI",
    "Axis Bank", "Axis", "Kotak Mahindra Bank", "Kotak",
    "Canara Bank", "Union Bank", "Indian Bank",
    "Punjab National Bank", "PNB", "Bank of Baroda", "BOB",
    "IDFC FIRST Bank", "IDFC", "Federal Bank",
    "Yes Bank", "DBS Bank", "RBL Bank", "IndusInd Bank",
    "Bandhan Bank", "South Indian Bank", "Karnataka Bank",
    "UCO Bank", "Central Bank", "City Union Bank",
]

# ── Transaction type keyword mapping ──────────────────────────────────────
TXN_TYPE_PATTERNS = [
    (r'\b(salary|sal credited|payroll|stipend)\b',        'Salary Credit'),
    (r'\b(refund|reversal|cashback credited)\b',           'Refund'),
    (r'\b(emi|equated monthly|installment|instalment)\b',  'EMI Debit'),
    (r'\b(atm\s*w[di]thdrawal|atm\s*wdl|cash\s*w[di]thdrawal|cash\s*wdl|cash\s*withdrawn)\b', 'ATM Withdrawal'),
    (r'\bfailed\b|\bdeclined\b|\bunsuccessful\b',          'Failed'),
    (r'\b(interest\s*credit|int\s*credit|interest\s*of)\b','Interest Credit'),
    # UPI — check before generic debit/credit
    (r'\bupi\b',                                           'UPI Transfer'),
    (r'\bimps\b',                                          'IMPS Transfer'),
    (r'\bneft\b',                                          'NEFT Transfer'),
    (r'\brtgs\b',                                          'RTGS Transfer'),
    # "Sent Rs.X From BANK To NAME" → UPI
    (r'^\s*sent\s+(?:rs|inr|₹)',                          'UPI Transfer'),
    (r'\b(debited|debit|deducted|withdrawn|withdrawal|wdl|purchase|spent|paid)\b', 'Debit'),
    (r'\b(credited|credit|received|deposited|added)\b',    'Credit'),
]


class TransactionExtractor:
    """
    Regex-based entity extractor for Indian bank transaction SMS.
    Works entirely offline — no ML needed for this step.
    """

    # ── Amount patterns ────────────────────────────────────────────────────
    AMOUNT_PATTERNS = [
        r'(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)',   # Rs.1,234.56 / INR 1234 / ₹500
        r'([\d,]+(?:\.\d{1,2})?)\s*(?:rs\.?|inr|rupees)', # 1234 Rs / 500 INR
        r'(?:rs|inr|₹)\s*:?\s*([\d,]+(?:\.\d{1,2})?)',  # Rs:1234
        # "debited by 120.00" / "credited by 500" — no currency symbol
        r'(?:debited|credited|debit|credit)\s+(?:by|for|of|with)?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    ]

    # ── Date patterns ──────────────────────────────────────────────────────
    DATE_PATTERNS = [
        r'\b(\d{2}[-/\.]\d{2}[-/\.]\d{2,4})\b',            # 17-06-26 / 17/06/2026
        r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})\b',
        # "20Jun26" / "22Dec25" — no space between day, month, year
        r'\b(\d{1,2}(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{2,4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b',                         # 2026-06-17 ISO
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        # "22-Jun-26" with dashes around month name
        r'\b(\d{1,2}-(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*-\d{2,4})\b',
    ]

    # ── Account number patterns ────────────────────────────────────────────
    ACCOUNT_PATTERNS = [
        r'a[/\\]?c\s*(?:no\.?)?\s*([xX*\d]{4,12})',         # A/c XX2428 / a/c 1332
        r'acct?\s*(?:no\.?)?\s*([xX*\d]{4,12})',
        r'account\s*(?:no\.?)?\s*([xX*\d]{4,12})',
        r'(?:card|a/c)\s+(?:ending|no\.?)\s*(\d{4})',        # card ending 1234
        r'\*{2,}(\d{4})',                                     # ****2428
    ]

    # ── Reference number patterns ──────────────────────────────────────────
    REF_PATTERNS = [
        r'(?:ref(?:erence)?(?:\s*no\.?)?|txn\s*id|tran(?:saction)?\s*(?:id|ref)|receipt)\s*[:\-#]?\s*([A-Z0-9]{6,16})',
        r'upi\s*(?:ref|id|txn)\s*[:\-#]?\s*([A-Z0-9]{6,20})',
    ]

    # ── UPI reference number (purely numeric, long) ───────────────────────
    UPI_REF_PATTERNS = [
        r'upi\s*[:\-]?\s*(\d{9,15})',
        r'upi\s*ref\s*[:\-]?\s*(\d{9,15})',
        r'(?<!\d)(\d{10,13})(?!\d)',                          # standalone 10-13 digit number
    ]

    # ── Balance patterns ───────────────────────────────────────────────────
    BALANCE_PATTERNS = [
        r'(?:avl?\.?\s*bal(?:ance)?|balance\s*(?:is|:)|bal\s*(?:after|:)?)\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
        r'(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:avl|available|balance)',
    ]

    # ── Beneficiary / merchant patterns ───────────────────────────────────
    # IMPORTANT: must stop before "Refno", "Call", "for dispute", "SMS BLOCK",
    # and must NEVER match a pure phone number (those appear at SMS end as
    # "SMS BLOCK 123 to 9876543210" — easy to mis-capture as beneficiary).
    _STOP = r'(?:\s+on\b|\s+ref\b|\s*refno\b|\s+via\b|\s+at\b|\s+if\s+not|\s+call|\s+for\s+dispute|\s+sms\s+block|[,.;]|$)'

    BENEFICIARY_PATTERNS = [
        # Handle missing space: "*7404To NAME" — insert lookahead boundary
        r'(?:paid to|sent to|trf to|transfer(?:red)? to)\s+([A-Za-z][A-Za-z0-9\s&\.]{2,35}?)' + _STOP,
        # "To NAME On 25/05/26" (Sent Rs.X From BANK To NAME On DATE)
        r'\bto\s+([A-Z][A-Za-z0-9\s&\.]{2,35}?)' + _STOP,
        # "BANK Acct XXNNN debited for Rs X on DATE; NAME credited"
        r';\s*([A-Za-z][A-Za-z0-9\s&\.]{2,35}?)\s+credited',
        r'(?:to|beneficiary)[:\s]+([A-Za-z][A-Za-z\s&\.]{2,25})(?:\s*[.\-,]|$)',
        r'at\s+([A-Z][A-Za-z\s&]{2,25})(?:\s*[.\-,]|$)',
    ]

    def _normalize(self, sms: str) -> str:
        """
        Fix common real-world SMS formatting issues where words run
        together with no space (banks often truncate/concatenate text).
        """
        s = sms
        # Insert space before "To"/"On"/"Ref" when glued to digits or letters
        # e.g. "*7404To MEDPLUS" -> "*7404 To MEDPLUS"
        s = re.sub(r'(?<=[0-9\*])(To|On|Ref)(?=\s?[A-Z])', r' \1 ', s)
        s = re.sub(r'(?<=[a-zA-Z])(Ref)(?=\d)', r' \1', s)
        # "13/06/26Ref" -> "13/06/26 Ref"
        s = re.sub(r'(\d{2,4})(Ref)', r'\1 \2', s)
        return s

    def extract(self, sms: str) -> dict:
        """
        Extract all transaction entities from an SMS string.

        Args:
            sms: Raw SMS text

        Returns:
            dict with keys: transaction_type, amount, currency, date,
                            account, beneficiary, upi_ref, ref_number,
                            bank, balance
        """
        sms  = self._normalize(sms)
        text = sms.lower()

        return {
            "transaction_type": self._get_txn_type(text),
            "amount":           self._get_amount(text),
            "currency":         "INR",
            "date":             self._get_date(text),
            "account":          self._get_account(text),
            "beneficiary":      self._get_beneficiary(sms),   # keep original case
            "upi_ref":          self._get_upi_ref(text),
            "ref_number":       self._get_ref(text),
            "bank":             self._get_bank(sms),           # keep original case
            "balance":          self._get_balance(text),
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_txn_type(self, text: str) -> str:
        for pattern, label in TXN_TYPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return label
        return "Unknown"

    def _get_amount(self, text: str) -> Optional[float]:
        for pat in self.AMOUNT_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(',', ''))
                except ValueError:
                    continue
        return None

    def _get_date(self, text: str) -> Optional[str]:
        for pat in self.DATE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _get_account(self, text: str) -> Optional[str]:
        for pat in self.ACCOUNT_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().upper()
        return None

    def _get_ref(self, text: str) -> Optional[str]:
        for pat in self.REF_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().upper()
        return None

    def _get_upi_ref(self, text: str) -> Optional[str]:
        for pat in self.UPI_REF_PATTERNS:
            m = re.search(pat, text)
            if m:
                val = m.group(1)
                if len(val) >= 9:
                    return val
        return None

    def _get_balance(self, text: str) -> Optional[float]:
        for pat in self.BALANCE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(',', ''))
                except ValueError:
                    continue
        return None

    def _get_bank(self, text: str) -> Optional[str]:
        tl = text.lower()
        # Handle both "Karnataka Bank" and "KarnatakaBank" (no space)
        BANK_VARIANTS = {
            "karnatakabank":   "Karnataka Bank",
            "hdfcbank":        "HDFC Bank",
            "icicibank":       "ICICI Bank",
            "axisbank":        "Axis Bank",
            "kotakbank":       "Kotak Mahindra Bank",
            "sbibank":         "SBI",
            "pnbbank":         "Punjab National Bank",
            "unionbank":       "Union Bank",
            "canarabank":      "Canara Bank",
            "bobbank":         "Bank of Baroda",
            "idfcbank":        "IDFC FIRST Bank",
            "induslndbank":    "IndusInd Bank",
            "yesbank":         "Yes Bank",
            "rblbank":         "RBL Bank",
            "federalbank":     "Federal Bank",
        }
        for variant, name in BANK_VARIANTS.items():
            if variant in tl.replace(' ', ''):
                return name
        for bank in KNOWN_BANKS:
            if bank.lower() in tl:
                return bank
        m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Bank)', text)
        if m:
            return m.group(1).strip()
        return None

    # Words that indicate we accidentally matched dispute/helpline text,
    # not an actual beneficiary name
    _BENEFICIARY_BLACKLIST = {
        'call', 'sms', 'block', 'dispute', 'services', 'helpline',
        'customer', 'care', 'support', 'toll', 'free', 'no', 'number',
    }

    def _get_beneficiary(self, text: str) -> Optional[str]:
        for pat in self.BENEFICIARY_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().rstrip('.,- ')

                # Reject pure digits (phone numbers, ref numbers)
                if re.fullmatch(r'[\d\s]+', val):
                    continue

                # Reject if it's mostly/only a phone-number-like token
                if re.search(r'^\d{8,}$', val.replace(' ', '')):
                    continue

                # Reject blacklisted helpline/dispute phrases
                if val.lower().strip() in self._BENEFICIARY_BLACKLIST:
                    continue
                first_word = val.lower().split()[0] if val.split() else ''
                if first_word in self._BENEFICIARY_BLACKLIST:
                    continue

                if len(val) >= 2:
                    return val
        return None
