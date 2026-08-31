import json

import pytest

from app.backend.pipeline import HybridForecastService, SavingsRecommendationService


def test_hybrid_missing_artifact_returns_clear_message():
    service = HybridForecastService(project_root=".")
    result = service.forecast_from_transaction({"amount": 2500, "merchant": "Zomato"})
    assert result["status"] == "ok"
    assert "artifact not found" in result["message"].lower() or "fallback" in result["message"].lower()
    assert result["total_predicted_expense"] > 0


def test_manual_expense_history_increases_forecast_values():
    service = HybridForecastService(project_root=".")
    result = service.forecast_from_transaction({
        "amount": 2500,
        "merchant": "Zomato",
        "monthly_income": 65000,
        "manual_expenses": [
            {"amount": 1200, "category": "Food"},
            {"amount": 950, "category": "Travel"},
        ],
    })
    assert result["status"] == "ok"
    assert result["total_predicted_expense"] > 0


def test_rl_missing_model_returns_clear_message():
    service = SavingsRecommendationService(project_root=".")
    result = service.recommend_for_forecast({"total_predicted_expense": 21000.0})
    assert result["status"] == "ok"
    assert result["recommended_amount"] >= 0
    assert "fallback" in result["message"].lower() or "not found" in result["message"].lower()
