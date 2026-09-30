"""Paths and static configuration shared by the backend services.

The three research modules (sms_parsing, expense_forecasting,
explainable_rl_planner) are consumed read-only. Everything the app generates
(trained artifacts, caches) is written under app/artifacts/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / 'app'
FRONTEND_DIR = APP_DIR / 'frontend'
ARTIFACT_DIR = APP_DIR / 'artifacts'

SMS_PARSER_DIR = ROOT / 'sms_parsing'
RL_PLANNER_DIR = ROOT / 'explainable_rl_planner'
FORECAST_DIR = ROOT / 'expense_forecasting'

USERS_FILE = FORECAST_DIR / 'users.xlsx'
EXPENSES_FILE = FORECAST_DIR / 'monthly_expenses.xlsx'
CALENDAR_FILE = FORECAST_DIR / 'calendar_events.xlsx'

# An artifact exported by 01_Hybrid_Training.ipynb is used when present;
# otherwise the one built by `python -m app.backend.build_artifacts`.
FORECAST_ARTIFACT_CANDIDATES = [
    FORECAST_DIR / 'hybrid_artifacts.pkl',
    ARTIFACT_DIR / 'hybrid_artifacts.pkl',
]
FORECAST_DATA_CACHE = ARTIFACT_DIR / 'forecast_data.pkl'

POLICY_MODEL_CANDIDATES = [
    ARTIFACT_DIR / 'savings_policy_model.pt',
    RL_PLANNER_DIR / 'savings_policy_model.pt',
]

# SMS spend categories (sms_parsing/spend_classifier.py) -> the expense
# taxonomy the forecasting model was trained on. None = not an expense.
SPEND_TO_EXPENSE_CATEGORY = {
    'Food & Dining': 'Food',
    'Shopping': 'Shopping',
    'Travel': 'Travel',
    'Entertainment': 'Entertainment',
    'Healthcare': 'Health',
    'Education': 'Education',
    'Utilities': 'Utilities',
    'Investment': 'Investments',
    'Loan & EMI': 'Miscellaneous',
    'Personal Transfer': 'Miscellaneous',
    'Cash Withdrawal': 'Miscellaneous',
    'Unknown': 'Miscellaneous',
    'Income / Salary': None,
    'Refund': None,
}

# Transaction types that put money into the account rather than spend it.
INFLOW_TRANSACTION_TYPES = {'Credit', 'Salary Credit', 'Interest Credit', 'Refund'}

SAMPLE_SMS = [
    {
        'label': 'Food delivery (UPI)',
        'text': 'Rs.1,287.00 debited from A/c XX2428 on 17-09-26 to ZOMATO via UPI Ref 426098114523. Avl Bal Rs.23,145.67 -HDFC Bank',
    },
    {
        'label': 'Cab ride',
        'text': 'Paid Rs.320 to Uber India via UPI. Ref 9988776655. Bal Rs.5,210.00 -Kotak',
    },
    {
        'label': 'Pharmacy (card)',
        'text': 'INR 3,450.00 spent on HDFC Bank Card xx8812 at APOLLO PHARMACY on 2026-09-10. Avl Lmt: INR 1,12,000',
    },
    {
        'label': 'College fee',
        'text': 'ICICI Bank Acct XX992 debited for Rs 12000.00 on 03-Sep-26; PES UNIVERSITY credited. UPI:425599887766.',
    },
    {
        'label': 'Shopping',
        'text': 'Rs.2500.00 debited from A/c XX1234 on 12-09-26 to ABC STORES via UPI Ref 412345678901. Avl Bal Rs.18,420.10 -SBI',
    },
    {
        'label': 'Loan EMI',
        'text': 'Rs.15000 debited from A/c XX1234 towards EMI for Loan A/c 99881. -SBI',
    },
    {
        'label': 'Salary credit',
        'text': 'Dear Customer, INR 45,000.00 credited to your A/c XX9921 on 01-09-26 towards SALARY. Avl Bal INR 61,210.55 - ICICI Bank',
    },
    {
        'label': 'OTP (not a transaction)',
        'text': 'Your OTP for login is 482913. Do not share it with anyone.',
    },
    {
        'label': 'Promotion (not a transaction)',
        'text': 'Flat 50% OFF on all shoes this weekend only! Shop now at Myntra. T&C apply',
    },
]

PRIORITIES = ['low', 'medium', 'high', 'critical']
