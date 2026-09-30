# Personalized Financial Planning

From messy bank SMS to an explained, goal-aware savings recommendation.

Budgeting apps show what was spent. This project reads the transaction SMS,
forecasts next month's spending by category, recommends how much to save
toward the user's goals, and explains the recommendation.

| Stage | Folder | What it does |
|---|---|---|
| 1. Understand | [`sms_parsing/`](sms_parsing) | CNN+GRU classifier separates transaction SMS from OTPs, promotions and the rest; a regex extractor pulls out amount, merchant, bank and date; a second CNN+GRU model tags the spend category. |
| 2. Forecast | [`expense_forecasting/`](expense_forecasting) | Per-user, per-category ARIMA forecasts feed a tuned Random Forest together with income, demographics and calendar-event features. Chronological train/test split; R² 0.944, MAE ₹1,163 on the held-out last three months. |
| 3. Plan & explain | [`explainable_rl_planner/`](explainable_rl_planner) | A REINFORCE policy suggests a savings rate; it is blended with the user's real saving behaviour and goal deadlines. Every recommendation comes with a blend breakdown, feature attribution, goal-progress preview and a what-if. |
| Demo | [`app/`](app) | Web app that runs the three stages end to end. |

## Run the demo

```bash
pip install -r requirements.txt
python -m app.backend.build_artifacts   # one time, about 10 minutes
python -m app.backend.main              # http://127.0.0.1:8000
```

Details, API and layout: [`app/README.md`](app/README.md).

## Team

Shikshith V · Poojitha L. B. Raghavendra · Mythri S — CSE (AI & ML), PES University
