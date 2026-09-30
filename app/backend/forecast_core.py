"""Hybrid ARIMA + Random Forest expense forecasting.

A script port of expense_forecasting/01_Hybrid_Training.ipynb (training) and
02_Demo.ipynb (inference) so the app can build and use the same artifact
without a notebook. Configuration values, feature columns and the tuned
Random Forest parameters are the ones recorded in the training notebook.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EXPENSE_CATEGORIES = [
    'Food', 'Cafe', 'Taxi', 'Public_Transport', 'Shopping', 'Entertainment',
    'Health', 'Education', 'Rent', 'Utilities', 'Travel', 'Gifts',
    'Insurance', 'Investments', 'Miscellaneous',
]

ARIMA_ORDER = (1, 1, 1)
ARIMA_MIN_HISTORY = 12
ARIMA_FORECAST_STEPS = 3
ARIMA_LOWER_BAND = 0.30
ARIMA_UPPER_BAND = 2.50

TEST_MONTHS = 3

EVENT_CATEGORY_MAP = {
    'Birthday': ['Food', 'Entertainment', 'Shopping'],
    'Wedding': ['Food', 'Shopping', 'Travel', 'Gifts'],
    'Anniversary': ['Food', 'Gifts'],
    'Festival': ['Shopping', 'Food', 'Gifts'],
    'Vacation': ['Travel', 'Food', 'Entertainment'],
    'Conference': ['Travel', 'Food'],
    'Business Trip': ['Travel', 'Food'],
    'Insurance Renewal': ['Insurance'],
    'Medical Appointment': ['Health'],
    'College Fee': ['Education'],
    'School Fee': ['Education'],
    'EMI Due': ['Miscellaneous'],
    'Rent Due': ['Rent'],
    'Car Service': ['Miscellaneous'],
    'Laptop Purchase': ['Miscellaneous'],
    'Mobile Purchase': ['Miscellaneous'],
    'Baby Shower': ['Gifts', 'Shopping', 'Food'],
    'Housewarming': ['Shopping', 'Food', 'Gifts'],
    'Exam': ['Education'],
    'Graduation': ['Gifts', 'Food', 'Shopping'],
    'Parents Visit': ['Food'],
    'Family Function': ['Food', 'Gifts', 'Shopping'],
    'Vehicle Purchase': ['Miscellaneous'],
    'Subscription Renewal': ['Entertainment', 'Miscellaneous'],
}
DEFAULT_EVENT_CATEGORIES = ['Miscellaneous']
RECURRING_EVENT_TYPES = {
    'Birthday', 'Anniversary', 'Insurance Renewal', 'Festival',
    'Rent Due', 'Subscription Renewal', 'EMI Due',
}
IMPORTANCE_RANK = {'Low': 1, 'Medium': 2, 'High': 3}
IMPORTANCE_WEIGHT = {'Low': 0.5, 'Medium': 1.0, 'High': 1.5}
CALENDAR_TRAILING_WINDOW = 6

# Inference-time importance defaults per event type (02_Demo.ipynb).
HIGH_IMPORTANCE_EVENT_TYPES = {
    'Wedding', 'Insurance Renewal', 'Rent Due', 'EMI Due',
    'College Fee', 'School Fee', 'Vehicle Purchase',
}
MEDIUM_IMPORTANCE_EVENT_TYPES = {
    'Birthday', 'Anniversary', 'Festival', 'Vacation', 'Conference',
    'Business Trip', 'Medical Appointment', 'Baby Shower', 'Housewarming',
    'Exam', 'Graduation', 'Family Function', 'Subscription Renewal', 'Parents Visit',
}

CATEGORICAL_FEATURE_COLS = [
    'user_id', 'expense_category', 'gender', 'occupation', 'city',
    'marital_status', 'financial_personality',
    'upcoming_event_type', 'event_importance_level',
]

FEATURE_COLS = [
    'user_id', 'expense_category', 'time_idx', 'monthly_income', 'year', 'month',
    'age', 'gender', 'occupation', 'city', 'marital_status', 'family_size',
    'financial_personality', 'event_count', 'calendar_pressure', 'planning_months_avg',
    'upcoming_event_type', 'event_importance_level',
    'arima_month_1', 'arima_month_2', 'arima_month_3',
]
ARIMA_COLS = ['arima_month_1', 'arima_month_2', 'arima_month_3']

TARGET_COL = 'monthly_expense'

# Best cross-validated parameters found by the notebook's TimeSeriesSplit
# grid search; reused here so the build does not repeat the search.
BEST_RF_PARAMS = {'max_depth': 20, 'min_samples_leaf': 2, 'n_estimators': 300}

RANDOM_SEED = 42

DEMO_COLS = [
    'user_id', 'age', 'gender', 'occupation', 'city',
    'marital_status', 'family_size', 'financial_personality',
]


# ---------------------------------------------------------------------------
# Helpers the notebooks import from hybrid_utils
# ---------------------------------------------------------------------------

def clean_arima_forecast(forecast, last_value, lower_band, upper_band):
    """Return a finite, non-negative, stable forecast.

    Any step that is non-finite, negative, or outside
    [lower_band, upper_band] x the last observed value falls back to the
    last observed value.
    """
    last_value = max(float(last_value), 0.0) if np.isfinite(last_value) else 0.0
    cleaned = np.asarray(forecast, dtype=float).copy()
    for i, value in enumerate(cleaned):
        unstable = not np.isfinite(value) or value < 0
        if not unstable and last_value > 0:
            unstable = value < lower_band * last_value or value > upper_band * last_value
        if unstable:
            cleaned[i] = last_value
    return cleaned


def safe_encode_column(series, encoder):
    """Encode with a fitted LabelEncoder, mapping unseen labels to class 0."""
    lookup = {label: idx for idx, label in enumerate(encoder.classes_)}
    return series.astype(str).map(lambda value: lookup.get(value, 0)).astype(int)


def infer_event_importance(event_type: str) -> str:
    if event_type in HIGH_IMPORTANCE_EVENT_TYPES:
        return 'High'
    if event_type in MEDIUM_IMPORTANCE_EVENT_TYPES:
        return 'Medium'
    return 'Low'


def fit_arima_forecast(series, order=ARIMA_ORDER, min_history=ARIMA_MIN_HISTORY,
                       steps=ARIMA_FORECAST_STEPS, lower_band=ARIMA_LOWER_BAND,
                       upper_band=ARIMA_UPPER_BAND):
    """Forecast `steps` months for one series, falling back to the last value."""
    from statsmodels.tsa.arima.model import ARIMA

    series = np.asarray(series, dtype=float)
    last_value = float(series[-1]) if len(series) else 0.0

    if len(series) < min_history:
        forecast = np.full(steps, last_value)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fitted = ARIMA(series, order=tuple(order)).fit()
                forecast = np.asarray(fitted.forecast(steps=steps), dtype=float)
        except Exception:
            forecast = np.full(steps, last_value)

    return clean_arima_forecast(forecast, last_value, lower_band, upper_band)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def build_expense_panel(expenses_df, users_df, categories=EXPENSE_CATEGORIES):
    """Melt wide monthly expenses into one row per (user, category, month)."""
    users = users_df[DEMO_COLS].copy()

    long_df = expenses_df.melt(
        id_vars=['user_id', 'year', 'month', 'income'],
        value_vars=categories,
        var_name='expense_category',
        value_name='monthly_expense',
    ).rename(columns={'income': 'monthly_income'})

    base_year = int(long_df['year'].min())
    long_df['time_idx'] = (long_df['year'] - base_year) * 12 + (long_df['month'] - 1)

    full_index = pd.MultiIndex.from_product(
        [long_df['user_id'].unique(), categories,
         range(long_df['time_idx'].min(), long_df['time_idx'].max() + 1)],
        names=['user_id', 'expense_category', 'time_idx'],
    )
    panel = pd.DataFrame(index=full_index).reset_index()
    panel = panel.merge(
        long_df[['user_id', 'expense_category', 'time_idx', 'monthly_expense', 'monthly_income']],
        on=['user_id', 'expense_category', 'time_idx'],
        how='left',
    )
    panel['monthly_expense'] = panel['monthly_expense'].fillna(0.0).clip(lower=0.0)
    panel['year'] = base_year + panel['time_idx'] // 12
    panel['month'] = panel['time_idx'] % 12 + 1
    panel['monthly_income'] = (
        panel.groupby('user_id')['monthly_income'].transform(lambda s: s.ffill().bfill())
    )
    panel = panel.merge(users, on='user_id', how='left')
    panel = panel.sort_values(['user_id', 'expense_category', 'time_idx']).reset_index(drop=True)

    if panel.isna().sum().sum() != 0:
        raise ValueError('Unexpected missing values remain after panel construction')
    return panel, base_year


def _explode_events(calendar_df, id_suffix=''):
    rows = []
    for row in calendar_df.itertuples(index=False):
        for category in EVENT_CATEGORY_MAP.get(row.event_type, DEFAULT_EVENT_CATEGORIES):
            rows.append((
                f'{row.event_id}{id_suffix}', row.user_id, row.event_date, row.event_title,
                row.event_type, row.importance, row.estimated_total_cost,
                row.planning_months, category,
            ))
    return pd.DataFrame(rows, columns=[
        'event_id', 'user_id', 'event_date', 'event_title', 'event_type',
        'importance', 'estimated_total_cost', 'planning_months', 'expense_category',
    ])


def project_recurring_events(calendar_df):
    """Historical events plus recurring event types projected a year ahead."""
    calendar_df = calendar_df.copy()
    calendar_df['event_date'] = pd.to_datetime(calendar_df['event_date'])
    recurring = calendar_df[calendar_df['event_type'].isin(RECURRING_EVENT_TYPES)].copy()
    recurring['event_date'] = recurring['event_date'] + pd.DateOffset(years=1)
    return calendar_df, recurring


def build_calendar_long(calendar_df):
    historical, recurring = project_recurring_events(calendar_df)
    calendar_long = pd.concat(
        [_explode_events(historical), _explode_events(recurring, id_suffix='_proj')],
        ignore_index=True,
    )
    calendar_long['year'] = calendar_long['event_date'].dt.year
    calendar_long['month'] = calendar_long['event_date'].dt.month
    return calendar_long


def aggregate_calendar_features(calendar_long):
    """One row of calendar features per (user, category, year, month)."""
    keys = ['user_id', 'expense_category', 'year', 'month']
    agg = calendar_long.groupby(keys).agg(
        event_count=('event_id', 'count'),
        total_estimated_cost=('estimated_total_cost', 'sum'),
        planning_months_avg=('planning_months', 'mean'),
    ).reset_index()

    dominant_idx = calendar_long.groupby(keys)['estimated_total_cost'].idxmax()
    dominant = calendar_long.loc[dominant_idx, keys + ['event_type', 'importance']].rename(
        columns={'event_type': 'upcoming_event_type', 'importance': 'event_importance_level'})

    agg = agg.merge(dominant, on=keys, how='left')
    agg['importance_weight'] = agg['event_importance_level'].map(IMPORTANCE_WEIGHT).fillna(0.5)
    return agg


def attach_calendar_features(panel, calendar_agg, trailing_window=CALENDAR_TRAILING_WINDOW):
    """Merge calendar features; pressure is normalised by a causal baseline."""
    panel = panel.sort_values(['user_id', 'expense_category', 'time_idx']).copy()
    panel['trailing_avg_expense'] = (
        panel.groupby(['user_id', 'expense_category'])['monthly_expense']
        .transform(lambda s: s.rolling(window=trailing_window, min_periods=1).mean().shift(1))
        .fillna(0.0)
    )
    panel['hist_total'] = (
        panel.groupby(['user_id', 'time_idx'])['trailing_avg_expense'].transform('sum').clip(lower=1.0)
    )

    merged = panel.merge(calendar_agg, on=['user_id', 'expense_category', 'year', 'month'], how='left')
    for col in ['event_count', 'total_estimated_cost', 'planning_months_avg', 'importance_weight']:
        merged[col] = merged[col].fillna(0.0)
    merged['upcoming_event_type'] = merged['upcoming_event_type'].fillna('None')
    merged['event_importance_level'] = merged['event_importance_level'].fillna('Low')
    merged['calendar_pressure'] = (
        merged['total_estimated_cost'] / merged['hist_total']
    ) * merged['importance_weight']

    merged = merged.drop(
        columns=['trailing_avg_expense', 'hist_total', 'total_estimated_cost', 'importance_weight'])

    for col in CATEGORICAL_FEATURE_COLS:
        merged[col] = merged[col].astype(str)
    for col in ['age', 'family_size', 'month', 'year', 'monthly_expense', 'monthly_income',
                'event_count', 'calendar_pressure', 'planning_months_avg']:
        merged[col] = merged[col].astype(float)
    merged['time_idx'] = merged['time_idx'].astype(int)
    return merged


def _arima_chunk(chunk):
    return [(user_id, category, fit_arima_forecast(series)) for user_id, category, series in chunk]


def train_arima_forecasts(train_panel, n_jobs=-1, log=print):
    """One ARIMA forecast per (user, category), fit on training months only."""
    from joblib import Parallel, delayed

    ordered = train_panel.sort_values('time_idx')
    series_list = [
        (user_id, category, group[TARGET_COL].to_numpy(dtype=float))
        for (user_id, category), group in ordered.groupby(['user_id', 'expense_category'])
    ]
    chunk_size = 250
    chunks = [series_list[i:i + chunk_size] for i in range(0, len(series_list), chunk_size)]
    log(f'  fitting {len(series_list):,} ARIMA{ARIMA_ORDER} models in {len(chunks)} batches ...')
    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_arima_chunk)(chunk) for chunk in chunks)

    records = [
        {'user_id': user_id, 'expense_category': category,
         'arima_month_1': forecast[0], 'arima_month_2': forecast[1], 'arima_month_3': forecast[2]}
        for batch in results for user_id, category, forecast in batch
    ]
    arima_df = pd.DataFrame.from_records(records)
    values = arima_df[ARIMA_COLS].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all(), 'Invalid ARIMA output'
    return arima_df


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mape': float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100),
        'r2': float(r2_score(y_true, y_pred)),
    }


def train_hybrid_artifact(users_raw, expenses_raw, calendar_raw, rf_params=None,
                          n_jobs=-1, log=print):
    """Run the full training notebook pipeline and return the artifact bundle."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    rf_params = dict(rf_params or BEST_RF_PARAMS)

    log('  building expense panel and calendar features ...')
    panel, base_year = build_expense_panel(expenses_raw, users_raw)
    panel = attach_calendar_features(panel, aggregate_calendar_features(build_calendar_long(calendar_raw)))

    cutoff = int(panel['time_idx'].max()) - TEST_MONTHS + 1
    train_df = panel[panel['time_idx'] < cutoff].copy()
    test_df = panel[panel['time_idx'] >= cutoff].copy()
    log(f'  chronological split: {len(train_df):,} train rows / {len(test_df):,} test rows')

    arima_df = train_arima_forecasts(train_df, n_jobs=n_jobs, log=log)
    keys = ['user_id', 'expense_category']
    train_df = train_df.merge(arima_df, on=keys, how='left', validate='many_to_one')
    test_df = test_df.merge(arima_df, on=keys, how='left', validate='many_to_one')

    combined = pd.concat([train_df, test_df], axis=0, keys=['train', 'test'])
    encoders = {}
    for col in CATEGORICAL_FEATURE_COLS:
        encoder = LabelEncoder()
        combined[col] = encoder.fit_transform(combined[col].astype(str))
        encoders[col] = encoder
    train_encoded = combined.xs('train').reset_index(drop=True)
    test_encoded = combined.xs('test').reset_index(drop=True)

    log(f'  training RandomForestRegressor {rf_params} ...')
    model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=n_jobs, **rf_params)
    model.fit(train_encoded[FEATURE_COLS], train_encoded[TARGET_COL].astype(float))

    y_test = test_encoded[TARGET_COL].astype(float)
    test_metrics = evaluate_regression(y_test, model.predict(test_encoded[FEATURE_COLS]))
    arima_only_metrics = evaluate_regression(y_test, test_encoded['arima_month_1'])
    log(f"  test R2 {test_metrics['r2']:.4f} | MAE {test_metrics['mae']:.1f} | "
        f"RMSE {test_metrics['rmse']:.1f} | MAPE {test_metrics['mape']:.2f}%")

    importance = sorted(
        ({'feature': f, 'importance': float(v)} for f, v in zip(FEATURE_COLS, model.feature_importances_)),
        key=lambda item: item['importance'], reverse=True,
    )
    model.n_jobs = 1  # single-row inference is faster without worker dispatch

    return {
        'model': model,
        'encoders': encoders,
        'feature_cols': FEATURE_COLS,
        'target_col': TARGET_COL,
        'categorical_feature_cols': CATEGORICAL_FEATURE_COLS,
        'categories': EXPENSE_CATEGORIES,
        'base_year': base_year,
        'arima_config': {
            'order': list(ARIMA_ORDER),
            'min_history': ARIMA_MIN_HISTORY,
            'forecast_steps': ARIMA_FORECAST_STEPS,
            'lower_band': ARIMA_LOWER_BAND,
            'upper_band': ARIMA_UPPER_BAND,
        },
        'calendar_config': {
            'event_category_map': EVENT_CATEGORY_MAP,
            'default_event_categories': DEFAULT_EVENT_CATEGORIES,
            'recurring_event_types': sorted(RECURRING_EVENT_TYPES),
            'importance_rank': IMPORTANCE_RANK,
            'importance_weight': IMPORTANCE_WEIGHT,
            'trailing_window': CALENDAR_TRAILING_WINDOW,
        },
        'split_config': {'test_months': TEST_MONTHS, 'split_cutoff_time_idx': cutoff},
        'best_rf_params': rf_params,
        'test_metrics': test_metrics,
        'arima_only_test_metrics': arima_only_metrics,
        'feature_importance': importance,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_event_features(events: List[dict], categories: List[str], baseline: float,
                         calendar_config: Optional[dict] = None) -> Dict[str, dict]:
    """Per-category calendar features for the forecast month.

    Mirrors training: each event is routed to the categories its type
    touches, and calendar_pressure = total estimated cost / the user's
    trailing average monthly spend x the dominant event's importance weight.
    """
    calendar_config = calendar_config or {}
    category_map = calendar_config.get('event_category_map', EVENT_CATEGORY_MAP)
    default_categories = calendar_config.get('default_event_categories', DEFAULT_EVENT_CATEGORIES)
    weights = calendar_config.get('importance_weight', IMPORTANCE_WEIGHT)

    features = {
        category: {
            'event_count': 0, 'calendar_pressure': 0.0, 'planning_months_avg': 0.0,
            'upcoming_event_type': 'None', 'event_importance_level': 'Low',
        }
        for category in categories
    }
    baseline = max(float(baseline), 1.0)

    routed: Dict[str, list] = {}
    for event in events or []:
        for category in category_map.get(event['event_type'], default_categories):
            routed.setdefault(category, []).append(event)

    for category, items in routed.items():
        if category not in features:
            continue
        dominant = max(items, key=lambda item: item['estimated_cost'])
        total_cost = sum(item['estimated_cost'] for item in items)
        features[category] = {
            'event_count': len(items),
            'calendar_pressure': float(total_cost / baseline * weights.get(dominant['importance'], 0.5)),
            'planning_months_avg': float(np.mean([item['planning_months'] for item in items])),
            'upcoming_event_type': dominant['event_type'],
            'event_importance_level': dominant['importance'],
        }
    return features


def predict_next_month(bundle: dict, profile: dict, history: pd.DataFrame, monthly_income: float,
                       events: List[dict]) -> dict:
    """Predict next month's spend per category for one user.

    `history` holds one row per month (year, month, income + one column per
    category), oldest first.
    """
    categories = bundle['categories']
    arima_config = bundle['arima_config']
    calendar_config = bundle['calendar_config']

    arima = {
        category: fit_arima_forecast(
            history[category].to_numpy(dtype=float),
            order=arima_config['order'], min_history=arima_config['min_history'],
            steps=arima_config['forecast_steps'], lower_band=arima_config['lower_band'],
            upper_band=arima_config['upper_band'],
        )
        for category in categories
    }

    last = history.iloc[-1]
    next_year, next_month = (int(last['year']) + 1, 1) if int(last['month']) == 12 \
        else (int(last['year']), int(last['month']) + 1)
    time_idx = (next_year - bundle['base_year']) * 12 + (next_month - 1)

    recent = history.tail(calendar_config['trailing_window'])
    baseline = float(max(recent[categories].sum(axis=1).mean(), 1.0))
    event_features = build_event_features(events, categories, baseline, calendar_config)

    rows = []
    for category in categories:
        ef = event_features[category]
        rows.append({
            'user_id': profile['user_id'],
            'expense_category': category,
            'time_idx': time_idx,
            'monthly_income': monthly_income,
            'year': next_year,
            'month': next_month,
            'age': profile['age'],
            'gender': profile['gender'],
            'occupation': profile['occupation'],
            'city': profile['city'],
            'marital_status': profile['marital_status'],
            'family_size': profile['family_size'],
            'financial_personality': profile['financial_personality'],
            **ef,
            'arima_month_1': arima[category][0],
            'arima_month_2': arima[category][1],
            'arima_month_3': arima[category][2],
        })
    frame = pd.DataFrame(rows)
    for col, encoder in bundle['encoders'].items():
        if col in frame.columns:
            frame[col] = safe_encode_column(frame[col], encoder)

    predictions = np.clip(bundle['model'].predict(frame[bundle['feature_cols']]), 0.0, None)

    return {
        'year': next_year,
        'month': next_month,
        'predictions': {category: float(value) for category, value in zip(categories, predictions)},
        'arima': {category: [float(v) for v in arima[category]] for category in categories},
        'event_features': event_features,
        'baseline_monthly_spend': baseline,
    }
