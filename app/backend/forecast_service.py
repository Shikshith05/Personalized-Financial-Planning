"""Next-month expense forecasting for the demo profiles."""

import calendar as month_calendar
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import forecast_core as core
from .config import (
    CALENDAR_FILE, EXPENSES_FILE, FORECAST_ARTIFACT_CANDIDATES, FORECAST_DATA_CACHE, USERS_FILE,
)

HISTORY_MONTHS = 24
MAX_EVENTS = 12


class ForecastError(ValueError):
    """Invalid forecast input; the message is safe to show to the user."""


def load_raw_data():
    return pd.read_excel(USERS_FILE), pd.read_excel(EXPENSES_FILE), pd.read_excel(CALENDAR_FILE)


def select_demo_users(users_df, expenses_df, calendar_df, count: int = 8) -> List[str]:
    """Pick demo profiles that leave room to save, one per financial personality.

    Many users in the synthetic dataset spend more than they earn, which
    leaves the savings planner nothing to work with. Profiles with something
    on the calendar for the forecast month are preferred.
    """
    ordered = expenses_df.sort_values(['user_id', 'year', 'month'])
    recent = ordered.groupby('user_id').tail(6)
    recent = recent.assign(total=recent[core.EXPENSE_CATEGORIES].sum(axis=1))
    stats = recent.groupby('user_id').agg(total=('total', 'mean'), income=('income', 'mean'),
                                          peak=('total', 'max'))
    stats['ratio'] = stats['total'] / stats['income']
    stats = stats.join(users_df.set_index('user_id')[['financial_personality', 'occupation']])

    last = ordered.iloc[-1]
    year, month = (int(last['year']) + 1, 1) if int(last['month']) == 12 else (int(last['year']), int(last['month']) + 1)
    with_events = set()
    for frame in core.project_recurring_events(calendar_df):
        dates = frame['event_date']
        with_events |= set(frame.loc[(dates.dt.year == year) & (dates.dt.month == month), 'user_id'])
    stats['has_event'] = stats.index.isin(with_events)

    eligible = stats[stats['ratio'].between(0.50, 0.88) & (stats['peak'] < stats['income'] * 1.1)
                     & stats['income'].between(25000, 200000)]
    eligible = eligible.sort_values(['has_event', 'ratio'], ascending=[False, True])

    chosen, used_occupations = [], set()
    for _, group in eligible.groupby('financial_personality', sort=True):
        fresh = group[~group['occupation'].isin(used_occupations)]
        pick = (fresh if len(fresh) else group).iloc[0]
        chosen.append(pick.name)
        used_occupations.add(pick['occupation'])
    return sorted(chosen)[:count]


def build_data_cache(users_df, expenses_df, calendar_df) -> Dict[str, Any]:
    return {
        'users': users_df,
        'expenses': expenses_df.sort_values(['user_id', 'year', 'month']).reset_index(drop=True),
        'calendar': calendar_df,
        'demo_user_ids': select_demo_users(users_df, expenses_df, calendar_df),
    }


class ForecastService:
    def __init__(self, artifact_path: Optional[Path] = None, data_cache_path: Optional[Path] = None):
        import joblib

        self.artifact_path = Path(artifact_path) if artifact_path else next(
            (path for path in FORECAST_ARTIFACT_CANDIDATES if path.exists()), None)
        self.bundle: Optional[dict] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

        cache_path = Path(data_cache_path) if data_cache_path else FORECAST_DATA_CACHE
        if cache_path.exists():
            data = pd.read_pickle(cache_path)
        else:
            data = build_data_cache(*load_raw_data())
        self.users = data['users'].set_index('user_id', drop=False)
        self.expenses = data['expenses']
        self.demo_user_ids = data['demo_user_ids']
        self.calendar, self.recurring = core.project_recurring_events(data['calendar'])

        if self.artifact_path is None:
            self.error = 'Hybrid model not built yet. Run: python -m app.backend.build_artifacts'
        else:
            try:
                self.bundle = joblib.load(self.artifact_path)
                self.bundle['model'].n_jobs = 1
            except Exception as exc:
                self.error = f'Could not load the hybrid forecast model: {exc}'

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    @property
    def categories(self) -> List[str]:
        return self.bundle['categories'] if self.bundle else core.EXPENSE_CATEGORIES

    def status(self) -> Dict[str, Any]:
        status = {
            'ready': self.ready,
            'message': self.error or 'Hybrid ARIMA + Random Forest model loaded',
            'model': 'Per-category ARIMA(1,1,1) feeding a Random Forest regressor',
        }
        if self.bundle:
            status['metrics'] = self.bundle.get('test_metrics')
            status['arima_only_metrics'] = self.bundle.get('arima_only_test_metrics')
            status['trees'] = self.bundle.get('best_rf_params', {}).get('n_estimators')
        return status

    # -- profiles ----------------------------------------------------------

    def _history(self, user_id: str) -> pd.DataFrame:
        rows = self.expenses[self.expenses['user_id'] == user_id]
        if rows.empty:
            raise ForecastError(f'No expense history for profile {user_id}.')
        return rows.tail(HISTORY_MONTHS).reset_index(drop=True)

    @staticmethod
    def _next_month(history: pd.DataFrame):
        last = history.iloc[-1]
        year, month = int(last['year']), int(last['month'])
        return (year + 1, 1) if month == 12 else (year, month + 1)

    def _calendar_events(self, user_id: str, year: int, month: int) -> List[dict]:
        """Events on the profile's calendar for the forecast month."""
        events = []
        for frame in (self.calendar, self.recurring):
            rows = frame[(frame['user_id'] == user_id)
                         & (frame['event_date'].dt.year == year)
                         & (frame['event_date'].dt.month == month)]
            for row in rows.itertuples(index=False):
                events.append({
                    'title': str(row.event_title),
                    'event_type': str(row.event_type),
                    'estimated_cost': round(float(row.estimated_total_cost), 2),
                    'importance': str(row.importance),
                    'planning_months': float(row.planning_months),
                })
        return events

    def profile(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.users.index:
            raise ForecastError(f'Unknown profile {user_id}.')
        user = self.users.loc[user_id]
        history = self._history(user_id)
        year, month = self._next_month(history)
        totals = history[core.EXPENSE_CATEGORIES].sum(axis=1)
        recent = history.tail(6)
        return {
            'user_id': user_id,
            'age': int(user['age']),
            'gender': str(user['gender']),
            'occupation': str(user['occupation']),
            'city': str(user['city']),
            'marital_status': str(user['marital_status']),
            'family_size': int(user['family_size']),
            'financial_personality': str(user['financial_personality']),
            'monthly_income': round(float(history.iloc[-1]['income']), 2),
            'avg_monthly_spend': round(float(totals.tail(6).mean()), 2),
            'top_categories': [
                {'category': category, 'amount': round(float(amount), 2)}
                for category, amount in recent[core.EXPENSE_CATEGORIES].mean().nlargest(3).items()
            ],
            'history_months': int(len(history)),
            'forecast_month': {'year': year, 'month': month,
                               'label': f'{month_calendar.month_abbr[month]} {year}'},
            'calendar_events': self._calendar_events(user_id, year, month),
        }

    def profiles(self) -> List[Dict[str, Any]]:
        return [self.profile(user_id) for user_id in self.demo_user_ids]

    # -- forecasting -------------------------------------------------------

    def _parse_events(self, events: Optional[List[dict]]) -> List[dict]:
        known_types = set(core.EVENT_CATEGORY_MAP)
        parsed = []
        for raw in (events or [])[:MAX_EVENTS]:
            event_type = str(raw.get('event_type') or '').strip()
            if event_type not in known_types:
                raise ForecastError(f'Unknown event type "{event_type}".')
            try:
                cost = float(raw.get('estimated_cost'))
                planning = float(raw.get('planning_months') if raw.get('planning_months') is not None else 1.0)
            except (TypeError, ValueError):
                raise ForecastError(f'"{event_type}": estimated cost must be a number.')
            if not np.isfinite(cost) or cost <= 0:
                raise ForecastError(f'"{event_type}": estimated cost must be greater than zero.')
            importance = str(raw.get('importance') or core.infer_event_importance(event_type)).title()
            if importance not in core.IMPORTANCE_WEIGHT:
                importance = core.infer_event_importance(event_type)
            parsed.append({
                'event_type': event_type,
                'estimated_cost': cost,
                'importance': importance,
                'planning_months': float(np.clip(planning, 0.0, 12.0)),
            })
        return parsed

    def _parse_ledger(self, ledger: Optional[List[dict]]) -> Dict[str, float]:
        """Sum SMS-derived spending per forecast category."""
        totals: Dict[str, float] = {}
        for entry in ledger or []:
            category = str(entry.get('category') or '')
            if category not in self.categories:
                raise ForecastError(f'Unknown expense category "{category}".')
            try:
                amount = float(entry.get('amount'))
            except (TypeError, ValueError):
                raise ForecastError('Ledger amounts must be numbers.')
            if not np.isfinite(amount) or amount < 0:
                raise ForecastError('Ledger amounts cannot be negative.')
            totals[category] = totals.get(category, 0.0) + amount
        return totals

    def forecast(self, user_id: str, monthly_income: Optional[float] = None,
                 ledger: Optional[List[dict]] = None, events: Optional[List[dict]] = None) -> Dict[str, Any]:
        """Forecast next month's spend per category.

        `ledger` entries (spending parsed from SMS) are added to the latest
        month of the profile's history before the models run.
        """
        if user_id not in self.users.index:
            raise ForecastError(f'Unknown profile {user_id}.')
        categories = self.categories
        history = self._history(user_id).copy()
        ledger_totals = self._parse_ledger(ledger)
        parsed_events = self._parse_events(events)

        last_index = history.index[-1]
        for category, amount in ledger_totals.items():
            history.loc[last_index, category] += amount

        if monthly_income is None:
            monthly_income = float(history.iloc[-1]['income'])
        else:
            try:
                monthly_income = float(monthly_income)
            except (TypeError, ValueError):
                raise ForecastError('Monthly income must be a number.')
            if not np.isfinite(monthly_income) or monthly_income <= 0:
                raise ForecastError('Monthly income must be greater than zero.')

        totals = history[categories].sum(axis=1)
        recent_avg = history.tail(6)[categories].mean()
        last_month = history.iloc[-1]

        if self.ready:
            with self._lock:
                result = core.predict_next_month(
                    self.bundle, self.users.loc[user_id].to_dict(), history, monthly_income, parsed_events)
            predictions, arima = result['predictions'], result['arima']
            year, month = result['year'], result['month']
            event_features = result['event_features']
            model = {
                'name': 'Hybrid ARIMA + Random Forest',
                'fallback': False,
                'metrics': self.bundle.get('test_metrics'),
            }
        else:
            # Without the trained model, fall back to each category's
            # trailing three-month average and say so.
            year, month = self._next_month(history)
            trailing = history.tail(3)[categories].mean()
            predictions = {category: float(trailing[category]) for category in categories}
            arima = {category: [predictions[category]] * 3 for category in categories}
            event_features = core.build_event_features([], categories, 1.0)
            model = {'name': '3-month average (hybrid model not built)', 'fallback': True,
                     'metrics': None, 'message': self.error}

        rows = []
        for category in categories:
            features = event_features[category]
            rows.append({
                'category': category,
                'predicted': round(predictions[category], 2),
                'arima': round(arima[category][0], 2),
                'last_month': round(float(last_month[category]), 2),
                'avg_6m': round(float(recent_avg[category]), 2),
                'sms_added': round(ledger_totals.get(category, 0.0), 2),
                'event_count': int(features['event_count']),
                'event_type': features['upcoming_event_type'] if features['event_count'] else None,
                'calendar_pressure': round(float(features['calendar_pressure']), 4),
            })
        rows.sort(key=lambda row: row['predicted'], reverse=True)

        total_predicted = float(sum(predictions.values()))
        return {
            'status': 'ok',
            'model': model,
            'user_id': user_id,
            'target': {'year': year, 'month': month, 'label': f'{month_calendar.month_abbr[month]} {year}'},
            'monthly_income': round(monthly_income, 2),
            'total_predicted_expense': round(total_predicted, 2),
            'total_last_month': round(float(totals.iloc[-1]), 2),
            'total_avg_6m': round(float(totals.tail(6).mean()), 2),
            'projected_surplus': round(monthly_income - total_predicted, 2),
            'categories': rows,
            'history': [
                {
                    'label': f"{month_calendar.month_abbr[int(row['month'])]} {str(int(row['year']))[2:]}",
                    'year': int(row['year']),
                    'month': int(row['month']),
                    'total': round(float(total), 2),
                    'income': round(float(row['income']), 2),
                }
                for (_, row), total in zip(history.iterrows(), totals)
            ],
            'arima_trend': [
                round(float(sum(arima[category][step] for category in categories)), 2) for step in range(3)
            ],
            'sms_added_total': round(float(sum(ledger_totals.values())), 2),
            'events': parsed_events,
        }
