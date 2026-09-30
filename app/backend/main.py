"""FastAPI app: serves the frontend and exposes the three pipeline stages.

    python -m app.backend.main        # http://127.0.0.1:8000
"""

import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.dont_write_bytecode = True  # never write __pycache__ into the research modules

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import forecast_core
from .config import FRONTEND_DIR, PRIORITIES, SAMPLE_SMS, SPEND_TO_EXPENSE_CATEGORY
from .forecast_service import ForecastError
from .planner_service import PlannerError
from .sms_service import SMSError, split_messages

USER_ERRORS = (SMSError, ForecastError, PlannerError)


class Services:
    """Lazily loaded model services. Loading happens on a background thread
    so the page is reachable while TensorFlow and the forecast model warm up."""

    def __init__(self):
        self.sms = None
        self.forecast = None
        self.planner = None
        self.load_errors: Dict[str, str] = {}
        self._started = False
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._load, name='model-loader', daemon=True).start()

    def _load(self):
        # The planner goes first: it must import its own `train` module
        # before sms_parsing puts a different one on the path.
        for name, factory in (('planner', self._make_planner), ('forecast', self._make_forecast),
                              ('sms', self._make_sms)):
            try:
                setattr(self, name, factory())
            except Exception as exc:
                self.load_errors[name] = f'{type(exc).__name__}: {exc}'

    @staticmethod
    def _make_planner():
        from .planner_service import SavingsPlannerService
        return SavingsPlannerService()

    @staticmethod
    def _make_forecast():
        from .forecast_service import ForecastService
        return ForecastService()

    @staticmethod
    def _make_sms():
        from .sms_service import SMSService
        return SMSService()

    def status(self) -> Dict[str, Any]:
        modules = {}
        for name in ('sms', 'forecast', 'planner'):
            service = getattr(self, name)
            if service is not None:
                info = service.status()
                info['state'] = 'ready' if info['ready'] else 'unavailable'
            elif name in self.load_errors:
                info = {'ready': False, 'state': 'unavailable', 'message': self.load_errors[name]}
            else:
                info = {'ready': False, 'state': 'loading', 'message': 'Loading model ...'}
            modules[name] = info
        return {
            'modules': modules,
            'loading': any(info['state'] == 'loading' for info in modules.values()),
        }

    def require(self, name: str):
        service = getattr(self, name)
        if service is None:
            message = self.load_errors.get(name) or 'This model is still loading. Try again in a few seconds.'
            raise ServiceUnavailable(message)
        return service


class ServiceUnavailable(RuntimeError):
    pass


services = Services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    services.start()
    yield


app = FastAPI(title='Personalized Financial Planning', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')


@app.exception_handler(ServiceUnavailable)
async def unavailable_handler(_: Request, exc: ServiceUnavailable):
    return JSONResponse(status_code=503, content={'error': str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = '.'.join(str(part) for part in first.get('loc', []) if part != 'body')
    return JSONResponse(status_code=400, content={'error': f"Invalid request: {field} {first.get('msg', '')}".strip()})


async def user_error_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=400, content={'error': str(exc)})


for _error_type in USER_ERRORS:
    app.add_exception_handler(_error_type, user_error_handler)


@app.exception_handler(Exception)
async def unexpected_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={'error': f'{type(exc).__name__}: {exc}'})


# -- request bodies ---------------------------------------------------------

class SMSRequest(BaseModel):
    text: str = ''
    messages: Optional[List[str]] = None


class LedgerEntry(BaseModel):
    category: str
    amount: float


class EventEntry(BaseModel):
    event_type: str
    estimated_cost: float
    importance: Optional[str] = None
    planning_months: Optional[float] = None


class ForecastRequest(BaseModel):
    user_id: str
    monthly_income: Optional[float] = None
    ledger: List[LedgerEntry] = Field(default_factory=list)
    events: List[EventEntry] = Field(default_factory=list)


class GoalEntry(BaseModel):
    name: str = ''
    target_amount: float
    current_savings: float = 0.0
    deadline_months: int
    priority: str = 'medium'


class PlanRequest(BaseModel):
    monthly_income: float
    monthly_expense: float
    goals: List[GoalEntry]
    actual_savings: List[float] = Field(default_factory=list)


class AllocationEntry(BaseModel):
    goal: str
    amount: float


class WhatIfRequest(BaseModel):
    available_amount: float
    recommended_savings: float
    alternative_savings: float
    allocations: List[AllocationEntry]


# -- routes -----------------------------------------------------------------

@app.get('/')
def index():
    return FileResponse(FRONTEND_DIR / 'index.html', headers={'Cache-Control': 'no-store'})


@app.get('/api/status')
def status():
    return services.status()


@app.get('/api/meta')
def meta():
    forecast = services.require('forecast')
    return {
        'profiles': forecast.profiles(),
        'categories': forecast.categories,
        'event_types': [
            {
                'event_type': event_type,
                'categories': categories,
                'importance': forecast_core.infer_event_importance(event_type),
            }
            for event_type, categories in sorted(forecast_core.EVENT_CATEGORY_MAP.items())
        ],
        'sample_sms': SAMPLE_SMS,
        'priorities': PRIORITIES,
        'spend_to_expense_category': SPEND_TO_EXPENSE_CATEGORY,
    }


@app.post('/api/sms/analyze')
def sms_analyze(body: SMSRequest):
    messages = body.messages if body.messages is not None else split_messages(body.text)
    results = services.require('sms').analyze(messages)
    return {
        'results': results,
        'summary': {
            'messages': len(results),
            'transactions': sum(1 for item in results if item['is_transaction']),
            'expenses': sum(1 for item in results if item['counts_as_expense']),
            'expense_total': round(sum(item['amount'] for item in results if item['counts_as_expense']), 2),
        },
    }


@app.post('/api/forecast')
def forecast(body: ForecastRequest):
    return services.require('forecast').forecast(
        user_id=body.user_id,
        monthly_income=body.monthly_income,
        ledger=[entry.model_dump() for entry in body.ledger],
        events=[entry.model_dump() for entry in body.events],
    )


@app.post('/api/plan')
def plan(body: PlanRequest):
    return services.require('planner').plan(
        monthly_income=body.monthly_income,
        monthly_expense=body.monthly_expense,
        goals=[goal.model_dump() for goal in body.goals],
        actual_savings=body.actual_savings,
    )


@app.post('/api/plan/what-if')
def plan_what_if(body: WhatIfRequest):
    return services.require('planner').what_if(
        available_amount=body.available_amount,
        recommended_savings=body.recommended_savings,
        allocations=[entry.model_dump() for entry in body.allocations],
        alternative_savings=body.alternative_savings,
    )


def run():
    import uvicorn

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8000'))
    print(f'Personalized Financial Planning -> http://{host}:{port}  (models load in the background)')
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    run()
