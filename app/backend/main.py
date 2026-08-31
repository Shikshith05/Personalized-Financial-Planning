from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import run_forecast, run_recommendation, run_sms_analysis

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / 'frontend'

app = FastAPI(title='Financial Planning Demo')

app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={'error': f'{type(exc).__name__}: {exc}'},
    )


@app.get('/')
async def index():
    try:
        return FileResponse(FRONTEND_DIR / 'index.html')
    except Exception as exc:
        return JSONResponse(status_code=500, content={'error': str(exc)})


@app.post('/sms/analyze')
async def sms_analyze(payload: dict):
    try:
        body = payload or {}
        return await run_sms_analysis(body.get('sms_text', ''))
    except Exception as exc:
        return JSONResponse(status_code=400, content={'error': f'SMS analysis failed: {exc}'})


@app.post('/forecast/analyze')
async def forecast_analyze(payload: dict):
    try:
        body = payload or {}
        transaction = body.get('transaction') or body
        upcoming_known_expenses = body.get('upcoming_known_expenses', '')
        return await run_forecast(transaction, upcoming_known_expenses)
    except Exception as exc:
        return JSONResponse(status_code=400, content={'error': f'Forecast failed: {exc}'})


@app.post('/rl/recommend')
async def rl_recommend(payload: dict):
    try:
        body = payload or {}
        forecast = body.get('forecast') or body
        return await run_recommendation(forecast)
    except Exception as exc:
        return JSONResponse(status_code=400, content={'error': f'Recommendation failed: {exc}'})


@app.post('/pipeline')
async def pipeline(payload: dict):
    try:
        body = payload or {}
        sms_text = body.get('sms_text', '')
        upcoming_known_expenses = body.get('upcoming_known_expenses', '')
        manual_expenses = body.get('manual_expenses', [])

        sms_result = await run_sms_analysis(sms_text)
        forecast_result = {'status': 'skipped', 'message': 'No transaction found. Forecast skipped.'}
        recommendation = {'status': 'skipped', 'message': 'No transaction found. Recommendation skipped.'}

        if sms_result.get('is_transaction'):
            transaction_payload = {
                'amount': (sms_result.get('entities') or {}).get('amount'),
                'beneficiary': (sms_result.get('entities') or {}).get('beneficiary'),
                'bank': (sms_result.get('entities') or {}).get('bank'),
                'category': sms_result.get('spend_category'),
                'monthly_income': body.get('monthly_income', 50000),
                'merchant': (sms_result.get('entities') or {}).get('beneficiary') or sms_result.get('spend_category'),
                'manual_expenses': manual_expenses,
            }
            forecast_result = await run_forecast(transaction_payload, upcoming_known_expenses)
            if forecast_result.get('status') == 'ok':
                recommendation = await run_recommendation(forecast_result)

        return {
            'sms_result': sms_result,
            'forecast': forecast_result,
            'recommendation': recommendation,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={'error': f'Pipeline failed: {exc}'})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.backend.main:app', host='0.0.0.0', port=8000, reload=True)
