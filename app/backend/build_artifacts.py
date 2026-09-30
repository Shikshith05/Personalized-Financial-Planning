"""One-time setup: build the artifacts the app needs that are not in git.

    python -m app.backend.build_artifacts            # build whatever is missing
    python -m app.backend.build_artifacts --force    # rebuild everything

Outputs go to app/artifacts/ (git-ignored):
  forecast_data.pkl        cached copy of the expense_forecasting spreadsheets
  hybrid_artifacts.pkl     ARIMA + Random Forest bundle (same layout as the
                           one 01_Hybrid_Training.ipynb saves)
  savings_policy_model.pt  policy network for the explainable RL planner
"""

import argparse
import sys
import time

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='rebuild artifacts that already exist')
    parser.add_argument('--trees', type=int, default=None,
                        help='override the Random Forest size (notebook value: 300)')
    parser.add_argument('--jobs', type=int, default=-1, help='parallel workers (default: all cores)')
    args = parser.parse_args()

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import joblib
    import pandas as pd

    from . import forecast_core as core
    from .config import ARTIFACT_DIR, FORECAST_ARTIFACT_CANDIDATES, FORECAST_DATA_CACHE, FORECAST_DIR
    from .forecast_service import build_data_cache, load_raw_data

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    raw = None

    # 1. Spreadsheet cache -------------------------------------------------
    if args.force or not FORECAST_DATA_CACHE.exists():
        print('[1/3] Caching expense_forecasting spreadsheets ...')
        started = time.time()
        raw = load_raw_data()
        cache = build_data_cache(*raw)
        pd.to_pickle(cache, FORECAST_DATA_CACHE)
        print(f"  {len(cache['users']):,} users, {len(cache['expenses']):,} monthly rows, "
              f"demo profiles: {', '.join(cache['demo_user_ids'])}  ({time.time() - started:.0f}s)")
    else:
        print('[1/3] Spreadsheet cache already present.')

    # 2. Forecast model ----------------------------------------------------
    notebook_artifact = FORECAST_DIR / 'hybrid_artifacts.pkl'
    app_artifact = ARTIFACT_DIR / 'hybrid_artifacts.pkl'
    if notebook_artifact.exists() and not args.force:
        print(f'[2/3] Using the notebook-trained forecast model at {notebook_artifact}.')
    elif app_artifact.exists() and not args.force:
        print('[2/3] Forecast model already present.')
    else:
        print('[2/3] Training the hybrid ARIMA + Random Forest forecast model (a few minutes) ...')
        started = time.time()
        raw = raw or load_raw_data()
        rf_params = dict(core.BEST_RF_PARAMS)
        if args.trees:
            rf_params['n_estimators'] = args.trees
        bundle = core.train_hybrid_artifact(*raw, rf_params=rf_params, n_jobs=args.jobs)
        joblib.dump(bundle, app_artifact, compress=3)
        print(f'  saved {app_artifact.name} ({app_artifact.stat().st_size / 1e6:.0f} MB, '
              f'{time.time() - started:.0f}s)')

    # 3. Savings policy ----------------------------------------------------
    policy_path = ARTIFACT_DIR / 'savings_policy_model.pt'
    if policy_path.exists() and not args.force:
        print('[3/3] Savings policy already present.')
    else:
        print('[3/3] Training the savings policy (REINFORCE, about a minute) ...')
        started = time.time()
        import torch

        from .planner_service import train_policy

        model = train_policy()
        torch.save(model.state_dict(), policy_path)
        print(f'  saved {policy_path.name} ({time.time() - started:.0f}s)')

    assert any(path.exists() for path in FORECAST_ARTIFACT_CANDIDATES)
    print('\nDone. Start the app with:  python -m app.backend.main')


if __name__ == '__main__':
    main()
