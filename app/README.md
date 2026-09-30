# Demo app

A single-page web app that walks through the whole pipeline on one screen:

1. **Profile** – pick a user from the forecasting dataset (24 months of history).
2. **Understand** – paste bank SMS; `sms_parsing` classifies them, extracts the
   amount and merchant, and tags the spend category. Spending goes into a ledger.
3. **Forecast** – `expense_forecasting`'s hybrid ARIMA + Random Forest predicts
   next month's spend per category, using the history, the SMS ledger and any
   planned calendar events.
4. **Plan & explain** – `explainable_rl_planner` recommends a monthly saving for
   your goals and explains it: blend breakdown, feature attribution, goal
   progress and a what-if. Log what you actually saved and the next month adapts.

The app only *reads* the three module folders. Everything it generates goes to
`app/artifacts/` (git-ignored).

## Run it

Python 3.11 is recommended (TensorFlow does not support the newest Python yet).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m app.backend.build_artifacts   # one time, about 10 minutes
python -m app.backend.main              # then open http://127.0.0.1:8000
```

The header shows three status pills. The SMS models take about 30 seconds to
load after start-up; the page is usable while they warm up.

### What `build_artifacts` does

Trained model files are not in git (`*.pkl`, `*.pt` are ignored), so they are
built locally:

| File | What | How |
|---|---|---|
| `forecast_data.pkl` | cached copy of the three spreadsheets | read once from `expense_forecasting/*.xlsx` |
| `hybrid_artifacts.pkl` | ARIMA + Random Forest bundle | script port of `01_Hybrid_Training.ipynb`, reusing the notebook's tuned Random Forest parameters instead of repeating the grid search |
| `savings_policy_model.pt` | savings policy network | `explainable_rl_planner/train.py`'s `train_model()` run over randomly drawn incomes and goal sets, so one policy serves any user |

If you already have `expense_forecasting/hybrid_artifacts.pkl` from running the
training notebook, the app uses that one and skips the forecast training.

Use `--force` to rebuild, `--trees 100` for a smaller/faster forest.

## Layout

```
app/
  backend/
    main.py              FastAPI routes, background model loading
    sms_service.py       wraps sms_parsing.FinanceTrackerPipeline
    forecast_core.py     training + inference logic ported from the notebooks
    forecast_service.py  profiles, ledger and calendar events -> forecast
    planner_service.py   wraps explainable_rl_planner (recommend, explain, what-if)
    build_artifacts.py   one-time setup
    config.py            paths, category mapping, sample SMS
  frontend/
    index.html  styles.css  app.js  charts.js     (no build step, no CDN)
tests/
  test_pipeline_contracts.py
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | which models are loaded |
| GET | `/api/meta` | demo profiles, categories, event types, sample SMS |
| POST | `/api/sms/analyze` | `{text}` (one SMS per line) → classification + entities |
| POST | `/api/forecast` | `{user_id, monthly_income, ledger, events}` → per-category forecast |
| POST | `/api/plan` | `{monthly_income, monthly_expense, goals, actual_savings}` → recommendation + explanation |
| POST | `/api/plan/what-if` | what a smaller saving would cost each goal |

`/api/plan` is stateless: `actual_savings` lists what was really saved in each
earlier month, and the server replays those months through the environment.

## Tests

```bash
python -m pytest tests -q
```

## Things worth knowing

- SMS spending is added to the **latest month** of the chosen profile's history
  before forecasting; a profile is needed because the forecaster is trained on
  per-user history.
- SMS spend categories are mapped onto the forecaster's 15 categories in
  `config.py` (`Loan & EMI`, `Personal Transfer` and unknowns go to
  `Miscellaneous`). The category can be corrected per message in the UI.
- Calendar pressure at inference uses the event's estimated cost divided by the
  user's trailing six-month spend, the same definition as in training.
- Demo profiles are chosen automatically from users whose spending leaves room
  to save; many users in the synthetic dataset spend more than they earn, which
  gives the planner nothing to allocate.
