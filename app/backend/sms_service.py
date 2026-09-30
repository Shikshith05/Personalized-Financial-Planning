"""SMS understanding, backed by sms_parsing.FinanceTrackerPipeline."""

import contextlib
import io
import re
import sys
import threading
from typing import Any, Dict, List, Optional

from .config import INFLOW_TRANSACTION_TYPES, SMS_PARSER_DIR, SPEND_TO_EXPENSE_CATEGORY

sys.dont_write_bytecode = True

MAX_MESSAGES = 25
MAX_MESSAGE_LENGTH = 600

_DEBIT_WORDS = re.compile(r'\b(debited|debit|deducted|withdrawn|spent|paid|sent|purchase)\b', re.IGNORECASE)
_CREDIT_WORDS = re.compile(r'\b(credited|received|deposited)\b', re.IGNORECASE)


class SMSError(ValueError):
    """Invalid SMS input; the message is safe to show to the user."""


def split_messages(text: str) -> List[str]:
    """One SMS per non-empty line."""
    return [line.strip() for line in (text or '').splitlines() if line.strip()]


def _direction(text: str, entities: dict, spend_category: Optional[str]) -> str:
    """'out' for spending, 'in' for money received, 'none' if nothing moved."""
    txn_type = entities.get('transaction_type') or ''
    if txn_type == 'Failed':
        return 'none'
    if txn_type in INFLOW_TRANSACTION_TYPES or spend_category in ('Income / Salary', 'Refund'):
        return 'in'
    if _CREDIT_WORDS.search(text) and not _DEBIT_WORDS.search(text):
        return 'in'
    return 'out'


class SMSService:
    """Loads the CNN+GRU models once and serialises access to them."""

    def __init__(self):
        self.pipeline = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

        try:
            if str(SMS_PARSER_DIR) not in sys.path:
                sys.path.insert(0, str(SMS_PARSER_DIR))
            # The module prints load progress with characters a Windows
            # console may not be able to encode.
            with contextlib.redirect_stdout(io.StringIO()):
                from predict import FinanceTrackerPipeline
                self.pipeline = FinanceTrackerPipeline()
            self.sms_classes = [str(label) for label in self.pipeline.preprocessor.classes_]
        except Exception as exc:
            self.error = f'SMS models could not be loaded: {type(exc).__name__}: {exc}'

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def status(self) -> Dict[str, Any]:
        status = {
            'ready': self.ready,
            'message': self.error or 'CNN+GRU SMS classifier and spend-category model loaded',
            'model': 'CNN+GRU SMS classifier, regex entity extraction, CNN+GRU spend categoriser',
        }
        if self.ready:
            status['classes'] = self.sms_classes
        return status

    def analyze(self, messages: List[str]) -> List[Dict[str, Any]]:
        if not self.ready:
            raise SMSError(self.error or 'SMS models are not loaded.')
        messages = [str(message).strip() for message in messages if str(message).strip()]
        if not messages:
            raise SMSError('Paste at least one SMS to analyse.')
        if len(messages) > MAX_MESSAGES:
            raise SMSError(f'Analyse at most {MAX_MESSAGES} messages at a time.')

        results = []
        with self._lock:
            for text in messages:
                text = text[:MAX_MESSAGE_LENGTH]
                with contextlib.redirect_stdout(io.StringIO()):
                    raw = self.pipeline.predict(text)
                results.append(self._shape(text, raw))
        return results

    @staticmethod
    def _shape(text: str, raw: dict) -> Dict[str, Any]:
        entities = raw.get('entities') or {}
        spend_category = raw.get('spend_category')
        is_transaction = bool(raw.get('is_transaction'))
        amount = entities.get('amount')

        direction = _direction(text, entities, spend_category) if is_transaction else 'none'
        expense_category = None
        if direction == 'out':
            expense_category = SPEND_TO_EXPENSE_CATEGORY.get(spend_category, 'Miscellaneous')

        return {
            'text': text,
            'sms_category': raw.get('sms_category'),
            'sms_confidence': raw.get('sms_confidence'),
            'is_transaction': is_transaction,
            'direction': direction,
            'amount': float(amount) if amount is not None else None,
            'transaction_type': entities.get('transaction_type'),
            'merchant': entities.get('beneficiary'),
            'bank': entities.get('bank'),
            'account': entities.get('account'),
            'date': entities.get('date'),
            'reference': entities.get('upi_ref') or entities.get('ref_number'),
            'balance': entities.get('balance'),
            'spend_category': spend_category,
            'spend_confidence': raw.get('spend_confidence'),
            'spend_method': raw.get('spend_method'),
            'expense_category': expense_category,
            'counts_as_expense': bool(direction == 'out' and amount and expense_category),
        }
