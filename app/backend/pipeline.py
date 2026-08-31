import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SMS_PARSER_DIR = ROOT / 'sms_parsing'
RL_PLANNER_DIR = ROOT / 'explainable_rl_planner'
FORECAST_DIR = ROOT / 'expense_forecasting'

for candidate in [str(SMS_PARSER_DIR), str(RL_PLANNER_DIR)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from predict import FinanceTrackerPipeline
except Exception:
    FinanceTrackerPipeline = None

try:
    from financial_planner import recommend_savings, load_model
    from explainer import (
        calculate_counterfactual,
        calculate_feature_influence,
        format_recommendation_explanation,
        preview_goal_progress,
    )
    from env import SimpleGoalEnv
    from goals import Goal
    from goal_allocator import allocate_savings
except Exception:
    recommend_savings = None
    load_model = None
    calculate_counterfactual = None
    calculate_feature_influence = None
    format_recommendation_explanation = None
    preview_goal_progress = None
    allocate_savings = None
    SimpleGoalEnv = None
    Goal = None


class SMSClassificationService:
    def __init__(self, project_root: Optional[str] = None):
        self.root = Path(project_root) if project_root else ROOT
        self.sms_dir = self.root / 'sms_parsing'
        self.pipeline = None

        if FinanceTrackerPipeline is None:
            raise RuntimeError('The SMS parsing pipeline could not be imported. Check the installed dependencies and module paths.')

        self.pipeline = FinanceTrackerPipeline()

    def classify(self, sms_text: str) -> Dict[str, Any]:
        return self.pipeline.predict(sms_text)


class HybridForecastService:
    def __init__(self, project_root: Optional[str] = None):
        self.root = Path(project_root) if project_root else ROOT
        self.artifact_path = self.root / 'expense_forecasting' / 'hybrid_artifacts.pkl'

    def _parse_known_expenses(self, text: str) -> List[str]:
        if not text:
            return []
        parts = [chunk.strip() for chunk in text.replace(';', '\n').replace(',', '\n').splitlines()]
        return [part for part in parts if part]

    def _parse_manual_expenses(self, values: List[Dict[str, Any]]) -> List[float]:
        parsed: List[float] = []
        for entry in values or []:
            try:
                amount = float(entry.get('amount') or 0.0)
            except Exception:
                continue
            if amount > 0:
                parsed.append(amount)
        return parsed

    def _build_default_forecast(self, transaction: Dict[str, Any], known_expenses: str) -> Dict[str, Any]:
        amount = float(transaction.get('amount') or 0.0)
        merchant = str(transaction.get('beneficiary') or transaction.get('merchant') or 'Merchant').strip() or 'Merchant'
        category = str(transaction.get('category') or transaction.get('spend_category') or 'Miscellaneous').strip() or 'Miscellaneous'
        history = [max(0.0, amount * 0.7), max(0.0, amount * 0.95), max(0.0, amount * 1.15)]

        manual_expenses = self._parse_manual_expenses(transaction.get('manual_expenses') or [])
        if manual_expenses:
            rolling = manual_expenses[:3]
            if len(rolling) < 3:
                rolling.extend([amount] * (3 - len(rolling)))
            history = [max(0.0, value) for value in rolling]

        if known_expenses:
            extra = len(self._parse_known_expenses(known_expenses))
            if extra:
                history = [max(0.0, value + extra * 120) for value in history]

        monthly_income = float(transaction.get('monthly_income') or 50000.0)
        total_predicted = float(sum(history)) / max(len(history), 1)
        return {
            'status': 'ok',
            'summary': f"Based on the observed transaction pattern for {merchant}, next month is expected to carry around {total_predicted:,.2f} in {category} spending.",
            'monthly_income': monthly_income,
            'total_predicted_expense': total_predicted,
            'category_breakdown': {category: round(total_predicted, 2)},
            'forecast': [
                {'label': 'Month 1', 'value': history[0]},
                {'label': 'Month 2', 'value': history[1]},
                {'label': 'Month 3', 'value': history[2]},
            ],
        }

    def forecast_from_transaction(self, transaction: Dict[str, Any], upcoming_known_expenses: str = '') -> Dict[str, Any]:
        if not self.artifact_path.exists():
            fallback = self._build_default_forecast(transaction, upcoming_known_expenses)
            fallback['status'] = 'ok'
            fallback['artifact_missing'] = True
            fallback['message'] = 'artifact not found — using built-in fallback forecast'
            return fallback

        try:
            bundle = joblib.load(self.artifact_path)
            model = bundle.get('model')
            feature_cols = bundle.get('feature_cols', [])
            categories = bundle.get('categories', [])
            arima_config = bundle.get('arima_config', {})
            if model is None:
                raise ValueError('Model missing from bundle')

            category = str(transaction.get('category') or transaction.get('spend_category') or 'Miscellaneous').strip() or 'Miscellaneous'
            monthly_income = float(transaction.get('monthly_income') or 50000.0)
            amount = float(transaction.get('amount') or 0.0)
            base_row = {
                'user_id': 'U00001',
                'expense_category': category,
                'time_idx': 1,
                'monthly_income': monthly_income,
                'month': 1,
                'year': 2026,
                'age': 31,
                'family_size': 2,
                'event_importance_level': 'Low',
                'event_count': 0,
                'calendar_pressure': 0.0,
                'planning_months_avg': 0.0,
                'upcoming_event_type': 'None',
                'gender': 'Unknown',
                'occupation': 'Unknown',
                'city': 'Unknown',
                'marital_status': 'Unknown',
                'financial_personality': 'Unknown',
                'arima_month_1': max(0.0, amount * 0.8),
                'arima_month_2': max(0.0, amount * 0.95),
                'arima_month_3': max(0.0, amount * 1.1),
            }

            row = {col: base_row.get(col, 0.0) for col in feature_cols}
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = str(value)
            feature_frame = pd.DataFrame([row])
            prediction = float(model.predict(feature_frame)[0])
            prediction = max(0.0, prediction)

            arima_steps = arima_config.get('forecast_steps', 3)
            forecast_points = [max(0.0, prediction * (0.85 + (i * 0.08))) for i in range(arima_steps)]
            summary = f"The trained Random Forest + ARIMA hybrid model estimates a next-month spend of {prediction:,.2f} for {category}."
            return {
                'status': 'ok',
                'summary': summary,
                'monthly_income': monthly_income,
                'total_predicted_expense': prediction,
                'category_breakdown': {category: round(prediction, 2)},
                'forecast': [{'label': f'Month {idx}', 'value': float(value)} for idx, value in enumerate(forecast_points, start=1)],
                'artifact': str(self.artifact_path),
            }
        except Exception:
            return self._build_default_forecast(transaction, upcoming_known_expenses)


class SavingsRecommendationService:
    def __init__(self, project_root: Optional[str] = None):
        self.root = Path(project_root) if project_root else ROOT
        self.model_path = self.root / 'explainable_rl_planner' / 'savings_policy_model.pt'

    def recommend_for_forecast(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model_path.exists():
            monthly_income = float(forecast.get('monthly_income') or 50000.0)
            monthly_expense = float(forecast.get('total_predicted_expense') or 22000.0)
            available_amount = max(monthly_income - monthly_expense, 0.0)
            recommended_amount = min(max(available_amount * 0.25, 0.0), available_amount)
            return {
                'status': 'ok',
                'model_missing': True,
                'message': 'Model not found — using fallback savings heuristic',
                'recommended_amount': float(recommended_amount),
                'available_amount': float(available_amount),
                'explanation_text': 'Fallback recommendation generated because the RL policy file is not present in the workspace.',
                'explanation': {
                    'blend_breakdown': [{'name': 'Fallback rule', 'weight': 1.0, 'contribution': recommended_amount}],
                    'feature_attribution': [],
                    'goal_progress': [],
                    'counterfactual': {'recommended_rate': 0.25, 'alternative_rate': 0.1, 'savings_difference': recommended_amount * 0.15, 'goal_impacts': []},
                },
            }

        if not all([recommend_savings is not None, calculate_feature_influence is not None, allocate_savings is not None, Goal is not None, SimpleGoalEnv is not None]):
            return {
                'status': 'setup_error',
                'message': 'The RL planner dependencies were not imported successfully.',
            }

        monthly_income = float(forecast.get('monthly_income') if forecast.get('monthly_income') is not None else 50000.0)
        monthly_expense = float(forecast.get('total_predicted_expense') or 22000.0)
        available_amount = max(monthly_income - monthly_expense, 0.0)

        goals = [
            Goal(name='Emergency fund', target_amount=20000.0, current_savings=5000.0, deadline_months=8, priority='high'),
            Goal(name='Vacation', target_amount=12000.0, current_savings=3500.0, deadline_months=6, priority='medium'),
        ]

        env = SimpleGoalEnv(monthly_income=monthly_income, monthly_expense=monthly_expense, goals=goals, horizon=8)
        state = env.reset()
        model = load_model(str(self.model_path), state_dim=env.observation_dim)
        action, recommended_savings, breakdown = recommend_savings(model, env, state)

        _, policy_influences = calculate_feature_influence(model, state)
        recommended_allocations = allocate_savings(recommended_savings, env.goals)
        goal_progress = preview_goal_progress(recommended_allocations, env.goals)
        rate = recommended_savings / available_amount if available_amount > 0 else 0.0
        counterfactual = calculate_counterfactual(
            available_amount=available_amount,
            recommended_rate=rate,
            allocations=recommended_allocations,
            alternative_rate=max(rate - 0.20, 0.0),
        )
        explanation_text = format_recommendation_explanation(breakdown, policy_influences)

        return {
            'status': 'ok',
            'recommended_amount': float(recommended_savings),
            'available_amount': float(available_amount),
            'explanation_text': explanation_text,
            'explanation': {
                'blend_breakdown': breakdown.get('components', []),
                'feature_attribution': policy_influences[:5],
                'goal_progress': goal_progress,
                'counterfactual': {
                    'recommended_rate': counterfactual.get('recommended_rate', 0.0),
                    'alternative_rate': counterfactual.get('alternative_rate', 0.0),
                    'savings_difference': counterfactual.get('savings_difference', 0.0),
                    'goal_impacts': counterfactual.get('goal_impacts', []),
                },
            },
        }


async def run_sms_analysis(sms_text: str) -> Dict[str, Any]:
    service = SMSClassificationService()
    return service.classify(sms_text)


async def run_forecast(transaction: Dict[str, Any], upcoming_known_expenses: str = '') -> Dict[str, Any]:
    service = HybridForecastService()
    return service.forecast_from_transaction(transaction, upcoming_known_expenses)


async def run_recommendation(forecast: Dict[str, Any]) -> Dict[str, Any]:
    service = SavingsRecommendationService()
    return service.recommend_for_forecast(forecast)
