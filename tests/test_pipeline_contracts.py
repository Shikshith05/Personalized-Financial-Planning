"""Contract tests for the demo app's backend.

Tests that need a trained artifact are skipped until
`python -m app.backend.build_artifacts` has been run. The SMS models are
not loaded here (TensorFlow start-up is slow); SMS shaping is tested on a
recorded pipeline result instead.
"""

import numpy as np
import pytest

from app.backend import forecast_core as core
from app.backend.config import FORECAST_ARTIFACT_CANDIDATES, FORECAST_DATA_CACHE, POLICY_MODEL_CANDIDATES
from app.backend.forecast_service import ForecastError, ForecastService
from app.backend.planner_service import PlannerError, SavingsPlannerService
from app.backend.sms_service import SMSService, split_messages

needs_forecast = pytest.mark.skipif(
    not (FORECAST_DATA_CACHE.exists() and any(path.exists() for path in FORECAST_ARTIFACT_CANDIDATES)),
    reason='forecast artifact not built',
)
needs_policy = pytest.mark.skipif(
    not any(path.exists() for path in POLICY_MODEL_CANDIDATES), reason='savings policy not built')

GOALS = [
    {'name': 'Emergency fund', 'target_amount': 60000, 'current_savings': 15000, 'deadline_months': 12, 'priority': 'high'},
    {'name': 'Laptop', 'target_amount': 30000, 'current_savings': 3000, 'deadline_months': 8, 'priority': 'medium'},
]


# -- forecast core ------------------------------------------------------------

def test_clean_arima_forecast_replaces_unstable_steps():
    cleaned = core.clean_arima_forecast([np.nan, -5.0, 10.0, 1000.0, 120.0], last_value=100.0,
                                        lower_band=0.3, upper_band=2.5)
    assert cleaned.tolist() == [100.0, 100.0, 100.0, 100.0, 120.0]


def test_event_features_follow_training_definition():
    features = core.build_event_features(
        [{'event_type': 'Birthday', 'estimated_cost': 5000.0, 'importance': 'Medium', 'planning_months': 1.0}],
        core.EXPENSE_CATEGORIES, baseline=50000.0)
    assert features['Food']['event_count'] == 1
    assert features['Food']['calendar_pressure'] == pytest.approx(0.1)  # 5000 / 50000 x weight 1.0
    assert features['Rent']['event_count'] == 0
    assert features['Rent']['upcoming_event_type'] == 'None'


# -- SMS ----------------------------------------------------------------------

def test_split_messages_one_per_line():
    assert split_messages('first\n\n  second  \n') == ['first', 'second']


def test_sms_result_is_mapped_to_a_forecast_category():
    raw = {
        'sms_category': 'Transaction', 'sms_confidence': 99.9, 'is_transaction': True,
        'entities': {'transaction_type': 'UPI Transfer', 'amount': 320.0, 'beneficiary': 'Uber India',
                     'bank': 'Kotak', 'date': None, 'account': None, 'upi_ref': '99', 'ref_number': None,
                     'balance': 5210.0},
        'spend_category': 'Travel', 'spend_confidence': 95.0, 'spend_method': 'upi_lookup',
    }
    shaped = SMSService._shape('Paid Rs.320 to Uber India via UPI.', raw)
    assert shaped['direction'] == 'out'
    assert shaped['expense_category'] == 'Travel'
    assert shaped['counts_as_expense'] is True


def test_salary_credit_is_not_an_expense():
    raw = {
        'sms_category': 'Transaction', 'sms_confidence': 100.0, 'is_transaction': True,
        'entities': {'transaction_type': 'Salary Credit', 'amount': 45000.0},
        'spend_category': 'Income / Salary', 'spend_confidence': 100.0, 'spend_method': 'txn_type',
    }
    shaped = SMSService._shape('INR 45,000.00 credited to your A/c towards SALARY', raw)
    assert shaped['direction'] == 'in'
    assert shaped['counts_as_expense'] is False


# -- forecast service ---------------------------------------------------------

@pytest.fixture(scope='module')
def forecast_service():
    return ForecastService()


@needs_forecast
def test_forecast_contract(forecast_service):
    user_id = forecast_service.demo_user_ids[0]
    result = forecast_service.forecast(user_id)
    assert result['status'] == 'ok'
    assert result['model']['fallback'] is False
    assert len(result['categories']) == len(core.EXPENSE_CATEGORIES)
    assert len(result['history']) == 24
    assert all(row['predicted'] >= 0 for row in result['categories'])
    assert result['total_predicted_expense'] == pytest.approx(
        sum(row['predicted'] for row in result['categories']), abs=0.5)


@needs_forecast
def test_large_planned_purchase_raises_the_forecast(forecast_service):
    user_id = forecast_service.demo_user_ids[0]
    base = forecast_service.forecast(user_id)
    with_event = forecast_service.forecast(
        user_id, events=[{'event_type': 'Laptop Purchase', 'estimated_cost': 60000}])
    assert with_event['total_predicted_expense'] > base['total_predicted_expense']


@needs_forecast
def test_forecast_rejects_bad_input(forecast_service):
    user_id = forecast_service.demo_user_ids[0]
    with pytest.raises(ForecastError):
        forecast_service.forecast('nobody')
    with pytest.raises(ForecastError):
        forecast_service.forecast(user_id, ledger=[{'category': 'Yachts', 'amount': 10}])
    with pytest.raises(ForecastError):
        forecast_service.forecast(user_id, events=[{'event_type': 'Birthday', 'estimated_cost': -1}])


# -- planner ------------------------------------------------------------------

@pytest.fixture(scope='module')
def planner():
    return SavingsPlannerService()


@needs_policy
def test_plan_contract(planner):
    plan = planner.plan(60000, 45000, GOALS)
    assert plan['status'] == 'ok'
    assert plan['month'] == 1
    assert 0 <= plan['recommended_savings'] <= plan['available_amount'] == 15000
    assert [item['name'] for item in plan['blend']] == ['RL policy suggestion', 'Amount needed for deadlines']
    assert sum(item['weight'] for item in plan['blend']) == pytest.approx(1.0)
    assert sum(item['amount'] for item in plan['allocations']) == pytest.approx(plan['recommended_savings'], rel=1e-6)
    assert len(plan['drivers']) == 9
    assert {outcome['goal'] for outcome in plan['projection']['outcomes']} == {'Emergency fund', 'Laptop'}


@needs_policy
def test_plan_adapts_to_what_was_actually_saved(planner):
    first = planner.plan(60000, 45000, GOALS)
    low = planner.plan(60000, 45000, GOALS, actual_savings=[first['recommended_savings'] * 0.2])
    high = planner.plan(60000, 45000, GOALS, actual_savings=[first['recommended_savings']])
    assert low['month'] == high['month'] == 2
    assert 'Your recent saving behaviour' in [item['name'] for item in low['blend']]
    assert low['recommended_savings'] < high['recommended_savings']
    assert low['history'][0]['gap'] > 0


@needs_policy
def test_plan_without_surplus_explains_itself(planner):
    plan = planner.plan(40000, 52000, GOALS)
    assert plan['status'] == 'no_surplus'
    assert 'nothing left' in plan['message']


@needs_policy
def test_plan_rejects_bad_goals(planner):
    with pytest.raises(PlannerError):
        planner.plan(60000, 45000, [])
    with pytest.raises(PlannerError):
        planner.plan(60000, 45000, [{'name': 'x', 'target_amount': 0, 'deadline_months': 3}])
    with pytest.raises(PlannerError):
        planner.plan(60000, 45000, [{'name': 'x', 'target_amount': 100, 'deadline_months': 0}])


@needs_policy
def test_duplicate_goal_names_are_kept_apart(planner):
    goals = [dict(GOALS[0]), dict(GOALS[0])]
    plan = planner.plan(60000, 45000, goals)
    assert len({goal['name'] for goal in plan['goals']}) == 2


@needs_policy
def test_what_if_scales_each_goal_by_its_share(planner):
    plan = planner.plan(60000, 45000, GOALS)
    result = planner.what_if(plan['available_amount'], plan['recommended_savings'], plan['allocations'],
                             plan['recommended_savings'] / 2)
    assert result['savings_difference'] == pytest.approx(plan['recommended_savings'] / 2)
    assert sum(item['contribution_loss'] for item in result['goal_impacts']) == pytest.approx(
        result['savings_difference'])


# -- HTTP ---------------------------------------------------------------------

def test_status_and_index_are_served_without_loading_models():
    from fastapi.testclient import TestClient

    from app.backend.main import app

    client = TestClient(app)  # no lifespan: models are not loaded
    assert client.get('/').status_code == 200
    status = client.get('/api/status').json()
    assert set(status['modules']) == {'sms', 'forecast', 'planner'}
    assert client.post('/api/plan', json={'monthly_income': 1}).status_code == 400
    assert client.post('/api/forecast', json={'user_id': 'U00001'}).status_code == 503
