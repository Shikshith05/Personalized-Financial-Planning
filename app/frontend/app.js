/* Personalized Financial Planning — guided demo of the three-stage pipeline. */

const { esc } = Charts;
const $ = (id) => document.getElementById(id);

const STEPS = [
  { id: 'profile', label: 'Profile', sub: 'Pick a spending history' },
  { id: 'sms', label: 'Understand', sub: 'Classify bank SMS' },
  { id: 'forecast', label: 'Forecast', sub: 'Next month by category' },
  { id: 'plan', label: 'Plan & explain', sub: 'Savings, and why' },
];

const MODULE_LABELS = { sms: 'SMS models', forecast: 'Forecaster', planner: 'Savings planner' };
const BLEND_COLORS = {
  'RL policy suggestion': '--series-1',
  'Your recent saving behaviour': '--series-2',
  'Amount needed for deadlines': '--series-3',
};
const GOAL_COLORS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5', '--series-6', '--series-7', '--series-8'];

const state = {
  status: null,
  meta: null,
  step: 'profile',
  reached: new Set(['profile']),
  profile: null,
  smsResults: [],
  nextSmsId: 1,
  income: null,
  events: [],
  forecast: null,
  forecastStale: true,
  goals: [],
  goalsEdited: false,
  actualSavings: [],
  plan: null,
};

// -- formatting ---------------------------------------------------------------

const inr0 = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
const compact = new Intl.NumberFormat('en-IN', { notation: 'compact', maximumFractionDigits: 1 });
const money = (value) => (value == null || Number.isNaN(Number(value)) ? '—' : inr0.format(Number(value)));
const moneyCompact = (value) => `₹${compact.format(value)}`;
const pct = (value, digits = 0) => `${Number(value).toFixed(digits)}%`;
const catLabel = (category) => String(category).replace(/_/g, ' ');
const roundTo = (value, step) => Math.max(step, Math.round(value / step) * step);

// -- API ----------------------------------------------------------------------

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new Error('Cannot reach the server. Is `python -m app.backend.main` still running?');
  }
  let payload = null;
  try { payload = await response.json(); } catch (error) { /* non-JSON error body */ }
  if (!response.ok) {
    throw new Error((payload && payload.error) || `Request failed (${response.status})`);
  }
  return payload;
}

function setBusy(statusId, message) {
  $(statusId).innerHTML = message ? `<span class="spinner"></span>${esc(message)}` : '';
}

function showBanner(message, isError = false) {
  const banner = $('banner');
  banner.hidden = !message;
  banner.textContent = message || '';
  banner.classList.toggle('error', isError);
}

// -- status & boot --------------------------------------------------------------

function renderStatus() {
  const modules = state.status ? state.status.modules : {};
  $('moduleStatus').innerHTML = Object.keys(MODULE_LABELS).map((name) => {
    const info = modules[name] || { state: 'loading', message: 'Loading model ...' };
    const word = { ready: 'ready', loading: 'loading', unavailable: 'unavailable' }[info.state];
    return `<span class="status-pill ${info.state}" title="${esc(info.message)}">
      <span class="status-dot"></span>${esc(MODULE_LABELS[name])}: ${word}</span>`;
  }).join('');
}

function moduleState(name) {
  return state.status && state.status.modules[name] ? state.status.modules[name] : { state: 'loading', ready: false };
}

async function pollStatus() {
  try {
    state.status = await api('/api/status');
    renderStatus();
    if (!state.meta && moduleState('forecast').state !== 'loading') await loadMeta();
    const problems = Object.entries(state.status.modules)
      .filter(([, info]) => info.state === 'unavailable')
      .map(([name, info]) => `${MODULE_LABELS[name]}: ${info.message}`);
    if (problems.length) showBanner(problems.join('  ·  '));
    else if ($('banner').textContent && !$('banner').classList.contains('error')) showBanner('');
    if (state.status.loading) setTimeout(pollStatus, 1500);
  } catch (error) {
    showBanner(error.message, true);
    setTimeout(pollStatus, 3000);
  }
}

async function loadMeta() {
  try {
    state.meta = await api('/api/meta');
  } catch (error) {
    showBanner(error.message, true);
    return;
  }
  renderProfiles();
  $('sampleChips').innerHTML = state.meta.sample_sms.map((sample, index) => (
    `<button class="chip" type="button" data-sample="${index}">${esc(sample.label)}</button>`
  )).join('') + '<button class="chip" type="button" data-sample="all">All samples</button>';
  $('eventType').innerHTML = state.meta.event_types.map((item) => (
    `<option value="${esc(item.event_type)}">${esc(item.event_type)}</option>`
  )).join('');
  syncEventImportance();
}

// -- navigation -----------------------------------------------------------------

function renderStepper() {
  $('stepper').innerHTML = STEPS.map((step, index) => {
    const active = step.id === state.step;
    const done = state.reached.has(step.id) && !active;
    return `<button class="step-tab ${active ? 'active' : ''} ${done ? 'done' : ''}" data-go="${step.id}"
        ${state.reached.has(step.id) ? '' : 'disabled'} ${active ? 'aria-current="step"' : ''}>
      <span class="num">${index + 1}</span>
      <span><span class="label">${step.label}</span><br><span class="sub">${step.sub}</span></span>
    </button>`;
  }).join('');
}

function goTo(step) {
  if (step !== 'profile' && !state.profile) return;
  state.step = step;
  state.reached.add(step);
  STEPS.forEach(({ id }) => { $(`step-${id}`).hidden = id !== step; });
  renderStepper();
  Charts.hideTip();
  window.scrollTo({ top: 0 });

  if (step === 'sms') renderLedger();
  if (step === 'forecast') enterForecast();
  if (step === 'plan') enterPlan();
}

// -- step 1: profiles -------------------------------------------------------------

function renderProfiles() {
  if (!state.meta) {
    $('profileGrid').innerHTML = '<p class="inline-status"><span class="spinner"></span>Loading the forecaster and its profiles ...</p>';
    return;
  }
  $('profileGrid').innerHTML = state.meta.profiles.map((profile) => {
    const selected = state.profile && state.profile.user_id === profile.user_id;
    const share = profile.avg_monthly_spend / profile.monthly_income * 100;
    return `<button class="profile-card ${selected ? 'selected' : ''}" data-profile="${esc(profile.user_id)}"
        aria-pressed="${selected}">
      <div class="profile-top">
        <div>
          <div class="profile-name">${esc(profile.occupation)}, ${profile.age}</div>
          <div class="profile-meta">${esc(profile.city)} · ${esc(profile.marital_status)} · household of ${profile.family_size}</div>
        </div>
        <span class="tag accent">${esc(profile.financial_personality)}</span>
      </div>
      <div class="profile-nums">
        <div><div class="mini-label">Monthly income</div><div class="mini-value">${money(profile.monthly_income)}</div></div>
        <div><div class="mini-label">Typical spend</div><div class="mini-value">${money(profile.avg_monthly_spend)} <span class="hint">(${pct(share)})</span></div></div>
      </div>
      <div class="profile-meta">Spends most on ${profile.top_categories.map((c) => esc(catLabel(c.category))).join(', ')}
        · ${esc(profile.user_id)}</div>
    </button>`;
  }).join('');
}

function selectProfile(userId) {
  const profile = state.meta.profiles.find((item) => item.user_id === userId);
  if (!profile) return;
  const changed = !state.profile || state.profile.user_id !== userId;
  state.profile = profile;
  if (changed) {
    state.income = profile.monthly_income;
    state.events = profile.calendar_events.map((event) => ({ ...event }));
    state.forecast = null;
    state.forecastStale = true;
    state.goals = [];
    state.goalsEdited = false;
    state.actualSavings = [];
    state.plan = null;
    state.reached = new Set(['profile']);
    $('forecastOutput').innerHTML = '';
    $('planOutput').innerHTML = '';
    $('forecastNext').disabled = true;
  }
  $('profileNext').disabled = false;
  renderProfiles();
  renderStepper();
}

// -- step 2: SMS ------------------------------------------------------------------

function includedLedger() {
  return state.smsResults.filter((item) => item.counts_as_expense && item.include);
}

function markForecastStale() {
  state.forecastStale = true;
  state.plan = null;
  state.actualSavings = [];
}

async function analyzeSms() {
  const textValue = $('smsInput').value.trim();
  if (!textValue) {
    setBusy('smsStatus', '');
    $('smsStatus').textContent = 'Paste at least one SMS, or pick a sample.';
    return;
  }
  if (!moduleState('sms').ready) {
    $('smsStatus').textContent = moduleState('sms').state === 'loading'
      ? 'The SMS models are still loading (about 30 seconds after start-up).'
      : 'The SMS models are unavailable.';
    return;
  }
  $('analyzeBtn').disabled = true;
  setBusy('smsStatus', 'Classifying ...');
  try {
    const payload = await api('/api/sms/analyze', { text: textValue });
    const fresh = payload.results.map((result) => ({ ...result, id: state.nextSmsId++, include: result.counts_as_expense }));
    state.smsResults = [...fresh, ...state.smsResults];
    $('smsInput').value = '';
    const { messages, transactions } = payload.summary;
    $('smsStatus').textContent = `${messages} message${messages === 1 ? '' : 's'} read, ${transactions} transaction${transactions === 1 ? '' : 's'} found.`;
    if (fresh.some((item) => item.include)) markForecastStale();
    renderSmsResults();
    renderLedger();
  } catch (error) {
    $('smsStatus').textContent = error.message;
  } finally {
    $('analyzeBtn').disabled = false;
  }
}

function smsCategoryTag(item) {
  if (!item.is_transaction) return `<span class="tag">${esc(item.sms_category)} · not a transaction</span>`;
  if (item.direction === 'in') return '<span class="tag good">Transaction · money in</span>';
  if (item.direction === 'none') return '<span class="tag warn">Transaction · failed</span>';
  return '<span class="tag accent">Transaction · money out</span>';
}

const METHOD_LABELS = {
  ml_model: 'CNN+GRU model',
  upi_lookup: 'known merchant',
  merchant_name: 'merchant keyword',
  txn_type: 'transaction type',
  heuristic: 'name heuristic',
  unknown: 'no match',
};

function renderSmsResults() {
  if (!state.smsResults.length) {
    $('smsResults').innerHTML = '';
    return;
  }
  const categories = state.meta ? state.meta.categories : [];
  $('smsResults').innerHTML = state.smsResults.map((item) => {
    const fields = [];
    if (item.is_transaction) {
      fields.push(['Amount', money(item.amount)]);
      fields.push(['Merchant / payee', item.merchant || '—']);
      fields.push(['Type', item.transaction_type || '—']);
      fields.push(['Bank', item.bank || '—']);
      fields.push(['Date', item.date || '—']);
      fields.push(['Spend category', item.spend_category
        ? `${item.spend_category} (${pct(item.spend_confidence)} · ${METHOD_LABELS[item.spend_method] || item.spend_method})`
        : '—']);
    }
    let footer = '';
    if (item.counts_as_expense) {
      footer = `<div class="sms-foot">
        <label><input type="checkbox" data-sms-include="${item.id}" ${item.include ? 'checked' : ''}> Add to ledger</label>
        <label>as
          <select data-sms-category="${item.id}">
            ${categories.map((c) => `<option value="${esc(c)}" ${c === item.expense_category ? 'selected' : ''}>${esc(catLabel(c))}</option>`).join('')}
          </select>
        </label>
        <span class="spacer"></span>
        <button class="btn ghost small" data-sms-remove="${item.id}">Remove</button>
      </div>`;
    } else {
      const reason = !item.is_transaction ? 'Ignored: no money moved.'
        : item.direction === 'in' ? 'Not counted as spending.'
        : item.direction === 'none' ? 'Ignored: the transaction failed.'
        : 'No amount found, so it is not added to the ledger.';
      footer = `<div class="sms-foot"><span class="hint">${reason}</span><span class="spacer"></span>
        <button class="btn ghost small" data-sms-remove="${item.id}">Remove</button></div>`;
    }
    return `<div class="card sms-card">
      <div class="sms-head">${smsCategoryTag(item)}
        <span class="hint">SMS classifier confidence ${pct(item.sms_confidence, 1)}</span></div>
      <div class="sms-text">${esc(item.text)}</div>
      ${fields.length ? `<div class="kv-grid">${fields.map(([label, value]) => (
        `<div class="kv"><div class="mini-label">${label}</div><div class="mini-value">${esc(value)}</div></div>`
      )).join('')}</div>` : ''}
      ${footer}
    </div>`;
  }).join('');
}

function renderLedger() {
  const items = includedLedger();
  const totals = {};
  items.forEach((item) => { totals[item.expense_category] = (totals[item.expense_category] || 0) + item.amount; });
  const total = items.reduce((sum, item) => sum + item.amount, 0);
  const rows = Object.entries(totals).sort((a, b) => b[1] - a[1]);

  $('ledgerSummary').innerHTML = items.length
    ? `<div class="ledger-total">${money(total)}</div>
       <div class="hint">${items.length} transaction${items.length === 1 ? '' : 's'} across ${rows.length} categor${rows.length === 1 ? 'y' : 'ies'}</div>
       <ul class="ledger-list">${rows.map(([category, amount]) => (
         `<li><span>${esc(catLabel(category))}</span><span class="num">${money(amount)}</span></li>`
       )).join('')}</ul>`
    : `<div class="ledger-total">${money(0)}</div>
       <div class="hint">Nothing yet. You can also continue without SMS to forecast from history alone.</div>`;
}

// -- step 3: forecast -------------------------------------------------------------

function syncEventImportance() {
  const item = state.meta && state.meta.event_types.find((entry) => entry.event_type === $('eventType').value);
  if (item) $('eventImportance').value = item.importance;
}

function renderEvents() {
  $('eventMonthLabel').textContent = state.profile ? state.profile.forecast_month.label : 'the forecast month';
  $('eventList').innerHTML = state.events.length
    ? state.events.map((event, index) => `<span class="event-chip">
        ${esc(event.title || event.event_type)} · ${money(event.estimated_cost)} · ${esc(event.importance)}
        <button class="icon-btn" data-event-remove="${index}" aria-label="Remove ${esc(event.event_type)}">×</button>
      </span>`).join('')
    : '<span class="hint">Nothing on the calendar. Add an event to see how the forecast reacts.</span>';
}

function enterForecast() {
  $('incomeInput').value = Math.round(state.income);
  renderEvents();
  if (state.forecastStale || !state.forecast) runForecast();
  else renderForecast();
}

async function runForecast() {
  const income = Number($('incomeInput').value);
  if (!Number.isFinite(income) || income <= 0) {
    $('forecastStatus').textContent = 'Enter a monthly income greater than zero.';
    return;
  }
  state.income = income;
  $('forecastBtn').disabled = true;
  setBusy('forecastStatus', 'Fitting ARIMA per category and running the Random Forest ...');
  try {
    state.forecast = await api('/api/forecast', {
      user_id: state.profile.user_id,
      monthly_income: income,
      ledger: includedLedger().map((item) => ({ category: item.expense_category, amount: item.amount })),
      events: state.events.map(({ event_type, estimated_cost, importance, planning_months }) => (
        { event_type, estimated_cost, importance, planning_months }
      )),
    });
    state.forecastStale = false;
    state.plan = null;
    state.actualSavings = [];
    if (!state.goalsEdited) state.goals = [];
    setBusy('forecastStatus', '');
    renderForecast();
  } catch (error) {
    $('forecastStatus').textContent = error.message;
  } finally {
    $('forecastBtn').disabled = false;
  }
}

function renderForecast() {
  const f = state.forecast;
  if (!f) return;
  $('forecastNext').disabled = false;

  const diff = f.total_predicted_expense - f.total_avg_6m;
  const diffPct = f.total_avg_6m ? (diff / f.total_avg_6m) * 100 : 0;
  const surplus = f.projected_surplus;
  const metrics = f.model.metrics;

  $('forecastOutput').innerHTML = `
    <div class="tiles">
      <div class="tile hero">
        <div class="tile-label">Forecast spend, ${esc(f.target.label)}</div>
        <div class="tile-value">${money(f.total_predicted_expense)}</div>
        <div class="tile-note"><span class="delta ${diff > 0 ? 'up-bad' : 'down-good'}">${diff > 0 ? '▲' : '▼'} ${pct(Math.abs(diffPct), 1)}</span>
          vs 6-month average of ${money(f.total_avg_6m)}</div>
      </div>
      <div class="tile">
        <div class="tile-label">Monthly income</div>
        <div class="tile-value">${money(f.monthly_income)}</div>
        <div class="tile-note">Forecast spend is ${pct(f.total_predicted_expense / f.monthly_income * 100)} of income</div>
      </div>
      <div class="tile">
        <div class="tile-label">Left after expenses</div>
        <div class="tile-value">${money(surplus)}</div>
        <div class="tile-note">${surplus > 0 ? 'Available for the savings plan' : 'Nothing left to save this month'}</div>
      </div>
      <div class="tile">
        <div class="tile-label">Added from SMS</div>
        <div class="tile-value">${money(f.sms_added_total)}</div>
        <div class="tile-note">${f.sms_added_total > 0 ? 'Included in the latest month of history' : 'No SMS spending in the ledger'}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Total monthly spending</div>
      <div class="card-sub">24 months of history, then the forecast for ${esc(f.target.label)}.</div>
      <div class="legend">
        <span class="legend-item"><span class="swatch line" style="background:var(--series-1)"></span>Actual spend</span>
        <span class="legend-item"><span class="swatch ring" style="border-color:var(--series-1)"></span>${f.model.fallback ? '3-month average' : 'Hybrid forecast'}</span>
        ${f.model.fallback ? '' : '<span class="legend-item"><span class="swatch line" style="background:var(--series-2)"></span>ARIMA only, 3 months ahead</span>'}
        <span class="legend-item"><span class="swatch line" style="background:var(--muted)"></span>Income</span>
      </div>
      <div class="chart" id="trendChart"></div>
    </div>

    <div class="card">
      <div class="card-title">Forecast by category</div>
      <div class="card-sub">Where the ${esc(f.target.label)} spending is expected to go.</div>
      <div class="legend">
        <span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>Forecast</span>
        <span class="legend-item"><span class="swatch tick" style="background:var(--ink)"></span>6-month average</span>
      </div>
      <div class="chart" id="categoryChart"></div>
      <details class="table-view">
        <summary>Show as table</summary>
        <div class="table-scroll"><table class="data">
          <thead><tr><th>Category</th><th>Forecast</th><th>ARIMA only</th><th>6-month avg</th><th>Last month</th><th>From SMS</th><th>Event</th></tr></thead>
          <tbody>${f.categories.map((row) => `<tr>
            <td>${esc(catLabel(row.category))}</td>
            <td class="num">${money(row.predicted)}</td><td class="num">${money(row.arima)}</td>
            <td class="num">${money(row.avg_6m)}</td><td class="num">${money(row.last_month)}</td>
            <td class="num">${row.sms_added ? money(row.sms_added) : '—'}</td>
            <td>${row.event_type ? esc(row.event_type) : '—'}</td></tr>`).join('')}</tbody>
        </table></div>
      </details>
      <div class="model-note">${f.model.fallback
        ? esc(f.model.message || 'Hybrid model not available; showing a 3-month average.')
        : `Model: ${esc(f.model.name)}${metrics ? ` · held-out test (last 3 months, all users): R² ${metrics.r2.toFixed(3)}, MAE ${money(metrics.mae)}, MAPE ${pct(metrics.mape, 1)}` : ''}`}</div>
    </div>`;

  drawForecastCharts();
}

function drawForecastCharts() {
  const f = state.forecast;
  if (!f || !$('trendChart')) return;

  const n = f.history.length;
  const futureLabels = [];
  let { year, month } = f.target;
  for (let i = 0; i < 3; i += 1) {
    futureLabels.push(`${new Date(year, month - 1, 1).toLocaleString('en', { month: 'short' })} ${String(year).slice(2)}`);
    month += 1;
    if (month > 12) { month = 1; year += 1; }
  }
  const labels = [...f.history.map((row) => row.label), ...futureLabels];
  const pad = (values, offset) => labels.map((_, i) => (i >= offset && i < offset + values.length ? values[i - offset] : null));
  const lastActual = f.history[n - 1].total;

  const series = [
    { name: 'Income', color: '--muted', width: 1.5, values: pad(f.history.map((row) => row.income), 0) },
    { name: 'Actual spend', color: '--series-1', values: pad(f.history.map((row) => row.total), 0) },
  ];
  if (!f.model.fallback) {
    series.push({ name: 'ARIMA only', color: '--series-2', values: pad([lastActual, ...f.arima_trend], n - 1) });
  }
  series.push({
    name: f.model.fallback ? '3-month average' : 'Hybrid forecast', color: '--series-1', dashed: true,
    values: pad([lastActual, f.total_predicted_expense], n - 1), markers: [n], ring: true,
  });

  Charts.lineChart($('trendChart'), {
    labels, series, yMin: 0, formatY: moneyCompact,
    ariaLabel: `Monthly spending history and forecast of ${money(f.total_predicted_expense)} for ${f.target.label}`,
    formatTip: (i) => {
      const rows = [];
      if (i < n) {
        rows.push(['Actual spend', f.history[i].total], ['Income', f.history[i].income]);
      } else {
        if (i === n) rows.push([f.model.fallback ? '3-month average' : 'Hybrid forecast', f.total_predicted_expense]);
        if (!f.model.fallback) rows.push(['ARIMA only', f.arima_trend[i - n]]);
      }
      return `<b>${esc(labels[i])}</b>${rows.map(([label, value]) => (
        `<div class="tt-row"><span>${label}</span><span>${money(value)}</span></div>`)).join('')}`;
    },
  });

  Charts.barChart($('categoryChart'), {
    rows: f.categories.map((row) => ({
      label: catLabel(row.category),
      value: row.predicted,
      ref: row.avg_6m,
      tip: `<b>${esc(catLabel(row.category))}</b>
        <div class="tt-row"><span>Forecast</span><span>${money(row.predicted)}</span></div>
        <div class="tt-row"><span>6-month average</span><span>${money(row.avg_6m)}</span></div>
        <div class="tt-row"><span>Last month</span><span>${money(row.last_month)}</span></div>
        ${row.sms_added ? `<div class="tt-row"><span>of which from SMS</span><span>${money(row.sms_added)}</span></div>` : ''}
        ${row.event_type ? `<div class="tt-row"><span>Calendar</span><span>${esc(row.event_type)}</span></div>` : ''}`,
    })),
    format: money,
    ariaLabel: 'Forecast spending by category',
  });
}

// -- step 4: plan -----------------------------------------------------------------

function defaultGoals() {
  const f = state.forecast;
  const surplus = f && f.projected_surplus > 0 ? f.projected_surplus : (state.income || 50000) * 0.15;
  const emergency = roundTo(surplus * 5, 1000);
  const laptop = roundTo(surplus * 2.5, 1000);
  return [
    { name: 'Emergency fund', target_amount: emergency, current_savings: roundTo(emergency * 0.25, 500), deadline_months: 12, priority: 'high' },
    { name: 'New laptop', target_amount: laptop, current_savings: roundTo(laptop * 0.1, 500), deadline_months: 8, priority: 'medium' },
    { name: 'Goa trip', target_amount: roundTo(surplus * 1.2, 1000), current_savings: 0, deadline_months: 6, priority: 'low' },
  ];
}

function renderGoals() {
  const priorities = state.meta ? state.meta.priorities : ['low', 'medium', 'high', 'critical'];
  $('goalTable').innerHTML = state.goals.map((goal, index) => `
    <div class="goal-row">
      <div class="goal-name"><label for="g-name-${index}">Goal</label>
        <input id="g-name-${index}" data-goal="${index}" data-field="name" type="text" maxlength="40" value="${esc(goal.name)}"></div>
      <div><label for="g-target-${index}">Target (₹)</label>
        <input id="g-target-${index}" data-goal="${index}" data-field="target_amount" type="number" min="1" step="1000" value="${goal.target_amount}"></div>
      <div><label for="g-saved-${index}">Saved so far (₹)</label>
        <input id="g-saved-${index}" data-goal="${index}" data-field="current_savings" type="number" min="0" step="500" value="${goal.current_savings}"></div>
      <div><label for="g-deadline-${index}">Deadline (months)</label>
        <input id="g-deadline-${index}" data-goal="${index}" data-field="deadline_months" type="number" min="1" max="120" step="1" value="${goal.deadline_months}"></div>
      <div><label for="g-priority-${index}">Priority</label>
        <select id="g-priority-${index}" data-goal="${index}" data-field="priority">
          ${priorities.map((p) => `<option value="${p}" ${p === goal.priority ? 'selected' : ''}>${p[0].toUpperCase()}${p.slice(1)}</option>`).join('')}
        </select></div>
      <button class="icon-btn" data-goal-remove="${index}" aria-label="Remove ${esc(goal.name)}" ${state.goals.length === 1 ? 'disabled' : ''}>×</button>
    </div>`).join('');
  $('addGoalBtn').disabled = state.goals.length >= 8;
}

function enterPlan() {
  if (!state.forecast || state.forecastStale) { goTo('forecast'); return; }
  if (!state.goals.length) state.goals = defaultGoals();
  renderGoals();
  if (!state.plan) runPlan();
  else renderPlan();
}

async function runPlan() {
  if (!moduleState('planner').ready) {
    $('planStatus').textContent = moduleState('planner').message || 'The savings planner is not available.';
    return;
  }
  $('planBtn').disabled = true;
  setBusy('planStatus', 'Asking the policy ...');
  try {
    state.plan = await api('/api/plan', {
      monthly_income: state.forecast.monthly_income,
      monthly_expense: state.forecast.total_predicted_expense,
      goals: state.goals,
      actual_savings: state.actualSavings,
    });
    setBusy('planStatus', '');
    renderPlan();
  } catch (error) {
    $('planStatus').textContent = error.message;
  } finally {
    $('planBtn').disabled = false;
  }
}

function historyTable(plan) {
  if (!plan.history.length) return '';
  return `<div class="table-scroll"><table class="data">
    <thead><tr><th>Month</th><th>Recommended</th><th>Actually saved</th><th>Gap</th></tr></thead>
    <tbody>${plan.history.map((row) => `<tr><td>Month ${row.month}</td>
      <td class="num">${money(row.recommended)}</td><td class="num">${money(row.actual)}</td>
      <td class="num">${row.gap > 0.5 ? `${money(row.gap)} short` : row.gap < -0.5 ? `${money(-row.gap)} extra` : 'on target'}</td></tr>`).join('')}</tbody>
  </table></div>`;
}

function outcomeLine(outcome) {
  const name = `<b>${esc(outcome.goal)}</b>`;
  const plural = (count) => `${count} month${count === 1 ? '' : 's'}`;
  if (outcome.status === 'on_track') {
    return `<li><span class="tag good">✓ On track</span> ${name} is fully funded in ${plural(outcome.months_needed)}
      (deadline: ${plural(outcome.deadline_months)}).</li>`;
  }
  if (outcome.status === 'late') {
    return `<li><span class="tag warn">! Late</span> ${name} is funded in ${plural(outcome.months_needed)},
      after its ${plural(outcome.deadline_months)} deadline.</li>`;
  }
  return `<li><span class="tag bad">✕ Short</span> ${name} reaches ${pct(outcome.final_progress)} and is still
    ${money(outcome.shortfall)} short when the plan ends.</li>`;
}

function renderPlan() {
  const plan = state.plan;
  const out = $('planOutput');
  if (!plan) { out.innerHTML = ''; return; }

  if (plan.status === 'unavailable') {
    out.innerHTML = `<div class="banner error">${esc(plan.message)}</div>`;
    return;
  }
  if (plan.status === 'no_surplus') {
    out.innerHTML = `<div class="card"><span class="tag warn">! No surplus</span>
      <p style="margin-top:8px">${esc(plan.message)}</p>
      <p class="hint" style="margin-top:6px">Income ${money(plan.monthly_income)} · forecast expenses ${money(plan.monthly_expense)}.
      Go back to the forecast to raise the income or remove planned events.</p></div>`;
    return;
  }

  if (plan.finished) {
    out.innerHTML = `<div class="card">
      <span class="tag ${plan.all_goals_completed ? 'good' : 'warn'}">${plan.all_goals_completed ? '✓ All goals funded' : 'Plan period ended'}</span>
      <h2 style="margin-top:10px">${plan.all_goals_completed ? 'Every goal reached its target.' : `The ${plan.horizon}-month plan has ended.`}</h2>
      <div style="margin-top:12px">${plan.goals.map((goal) => `<div class="goal-meter">
        <div class="goal-meter-head"><b>${esc(goal.name)}</b><span class="num">${money(goal.current_savings)} of ${money(goal.target_amount)}</span></div>
        <div class="meter"><span class="now" style="width:${goal.progress}%"></span></div></div>`).join('')}</div>
      <div style="margin-top:12px">${historyTable(plan)}</div>
      <div class="row"><button class="btn primary" id="replanBtn">Plan again from month 1</button></div>
    </div>`;
    return;
  }

  const blendTotal = plan.blend.reduce((sum, item) => sum + item.contribution, 0) || 1;
  const adaptive = plan.history.length > 0;

  out.innerHTML = `
    <div class="card">
      <div class="reco">
        <div>
          <div class="tile-label">Month ${plan.month} of ${plan.horizon} · recommended saving</div>
          <div class="reco-amount">${money(plan.recommended_savings)}</div>
          <div class="reco-sub">${pct(plan.recommended_rate * 100)} of the ${money(plan.available_amount)} left after
            forecast expenses of ${money(plan.monthly_expense)}.</div>
        </div>
        <div>
          <div class="card-title">Why this amount</div>
          <div class="hint">${adaptive
            ? 'Now that there is a saving history, your own behaviour carries the most weight.'
            : 'First month: no saving history yet, so the policy and your deadlines decide.'}</div>
          <div class="stackbar" role="img" aria-label="Blend of the recommendation">
            ${plan.blend.map((item) => `<span style="flex:${Math.max(item.contribution, 0) / blendTotal};background:var(${BLEND_COLORS[item.name] || '--series-4'})"
              data-tip="${esc(`<b>${item.name}</b><div class="tt-row"><span>Contributes</span><span>${money(item.contribution)}</span></div>`)}"></span>`).join('')}
          </div>
          <div class="blend-rows">
            ${plan.blend.map((item) => `<div class="blend-row">
              <span class="swatch" style="background:var(${BLEND_COLORS[item.name] || '--series-4'})"></span>
              <div><b>${esc(item.name)}</b><div class="blend-math">${pct(item.weight * 100)} weight × ${money(item.value)} — ${esc(item.reason)}</div></div>
              <div class="num"><b>${money(item.contribution)}</b></div>
            </div>`).join('')}
          </div>
          ${plan.was_capped ? `<p class="hint" style="margin-top:8px">The blend came to ${money(plan.blended_total)} and was capped at the money available.</p>` : ''}
        </div>
      </div>
    </div>

    <div class="grid-2 stack-gap">
      <div class="card">
        <div class="card-title">What moved the RL policy</div>
        <div class="card-sub">The policy alone suggests saving ${pct(plan.policy_rate * 100, 1)} of what is left.
          Each bar shows how far one input pushed that rate, in percentage points, compared with a neutral value.</div>
        <div class="legend">
          <span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>Pushes saving up</span>
          <span class="legend-item"><span class="swatch" style="background:var(--negative)"></span>Pushes saving down</span>
        </div>
        <div class="chart" id="driverChart"></div>
      </div>
      <div class="card">
        <div class="card-title">Where the money goes</div>
        <div class="card-sub">The recommended amount is split by urgency: priority, how much is left and how near the deadline is.</div>
        <div class="legend">
          <span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>Saved so far</span>
          <span class="legend-item"><span class="swatch" style="background:var(--series-1-soft)"></span>Added this month</span>
        </div>
        ${plan.goals.map((goal) => `<div class="goal-meter">
          <div class="goal-meter-head"><span><b>${esc(goal.name)}</b> <span class="tag">${esc(goal.priority)}</span></span>
            <span class="num"><b>+${money(goal.allocation)}</b></span></div>
          <div class="meter"><span class="now" style="width:${goal.progress}%"></span><span class="add" style="width:${Math.max(goal.progress_after - goal.progress, 0)}%"></span></div>
          <div class="goal-meter-foot"><span>${pct(goal.progress)} → ${pct(goal.progress_after)} of ${money(goal.target_amount)}</span>
            <span>${goal.remaining > 0 ? `${goal.deadline_months > 0 ? `${goal.deadline_months} month${goal.deadline_months === 1 ? '' : 's'} left` : 'overdue'} · needs ${money(goal.required_per_month)}/month` : 'funded'}</span></div>
        </div>`).join('')}
      </div>
    </div>

    <div class="grid-2 stack-gap">
      <div class="card whatif">
        <div class="card-title">What if you saved less?</div>
        <div class="card-sub">Drag to see what a smaller amount would cost each goal this month.</div>
        <input type="range" id="whatIfRange" min="0" max="${Math.round(plan.recommended_savings)}"
          step="${Math.max(50, roundTo(plan.recommended_savings / 100, 50))}" value="${Math.round(plan.counterfactual.alternative_savings)}"
          aria-label="Alternative savings amount">
        <div class="whatif-scale"><span>${money(0)}</span><span>Recommended ${money(plan.recommended_savings)}</span></div>
        <div class="whatif-result" id="whatIfResult"></div>
      </div>
      <div class="card">
        <div class="card-title">If you follow the plan every month</div>
        <div class="card-sub">Simulated goal progress when each month's recommendation is saved in full.</div>
        <div class="legend">${plan.goals.map((goal, index) => (
          `<span class="legend-item"><span class="swatch line" style="background:var(${GOAL_COLORS[index]})"></span>${esc(goal.name)}</span>`)).join('')}</div>
        <div class="chart" id="projectionChart"></div>
        <ul class="outcome-list">${plan.projection.outcomes.map(outcomeLine).join('')}</ul>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Month ${plan.month} is over. What did you actually save?</div>
      <div class="card-sub">The planner adapts: next month's recommendation leans on what you really saved, not just on the target.</div>
      <div class="log-row">
        <div><label class="field-label" for="actualInput">Amount saved (₹)</label>
          <input id="actualInput" type="number" min="0" step="500" value="${Math.round(plan.recommended_savings)}"></div>
        <button class="btn primary" id="logBtn">Log and plan month ${plan.month + 1}</button>
        <button class="btn small" data-quick="0.5">Saved half</button>
        <button class="btn small" data-quick="1">Saved it all</button>
        ${adaptive ? '<button class="btn ghost small" id="replanBtn">Restart from month 1</button>' : ''}
      </div>
      <div style="margin-top:12px">${historyTable(plan)}</div>
      <details class="table-view">
        <summary>Show the planner's explanation in plain text</summary>
        <pre class="plain-explanation">${esc(plan.explanation_text)}</pre>
      </details>
    </div>`;

  renderWhatIf(plan.counterfactual);
  drawPlanCharts();
}

function drawPlanCharts() {
  const plan = state.plan;
  if (!plan || plan.status !== 'ok' || plan.finished || !$('driverChart')) return;

  Charts.divergingChart($('driverChart'), {
    rows: plan.drivers.map((driver) => ({
      label: driver.feature,
      value: driver.influence_pp,
      tip: `<b>${esc(driver.feature)}</b><div>${esc(driver.description)}</div>
        <div class="tt-row"><span>Effect on savings rate</span><span>${driver.influence_pp >= 0 ? '+' : ''}${driver.influence_pp.toFixed(2)} pp</span></div>
        <div class="tt-row"><span>Input value (normalised)</span><span>${driver.state_value.toFixed(2)}</span></div>`,
    })),
    format: (value) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(1)}`,
    ariaLabel: 'Effect of each input on the policy savings rate, in percentage points',
  });

  const months = plan.projection.months;
  const labels = months.map((row, index) => (index === 0 ? 'Now' : `M${row.month}`));
  Charts.lineChart($('projectionChart'), {
    labels,
    height: 220,
    yMin: 0,
    series: plan.goals.map((goal, index) => ({
      name: goal.name, color: GOAL_COLORS[index], values: months.map((row) => row.progress[goal.name]),
    })),
    formatY: (value) => `${value}%`,
    ariaLabel: 'Projected goal progress by month',
    formatTip: (i) => `<b>${i === 0 ? 'Now' : `After month ${months[i].month}`}</b>
      ${months[i].savings != null ? `<div class="tt-row"><span>Saved that month</span><span>${money(months[i].savings)}</span></div>` : ''}
      ${plan.goals.map((goal) => `<div class="tt-row"><span>${esc(goal.name)}</span><span>${pct(months[i].progress[goal.name])}</span></div>`).join('')}`,
  });
}

function renderWhatIf(result) {
  const target = $('whatIfResult');
  if (!target) return;
  if (result.savings_difference < 1) {
    target.innerHTML = `<b>${money(result.alternative_savings)}</b> is the recommended amount. Every goal gets its full share.`;
    return;
  }
  target.innerHTML = `Saving <b>${money(result.alternative_savings)}</b> instead of ${money(result.recommended_savings)}
    puts <b>${money(result.savings_difference)} less</b> toward your goals this month:
    <ul>${result.goal_impacts.filter((impact) => impact.contribution_loss > 0.5).map((impact) => (
      `<li>${esc(impact.goal)} gets ${money(impact.contribution_loss)} less (${money(impact.alternative_amount)} instead of ${money(impact.recommended_amount)})</li>`
    )).join('')}</ul>`;
}

let whatIfTimer = null;
function scheduleWhatIf(value) {
  clearTimeout(whatIfTimer);
  whatIfTimer = setTimeout(async () => {
    const plan = state.plan;
    if (!plan || plan.status !== 'ok' || plan.finished) return;
    try {
      const result = await api('/api/plan/what-if', {
        available_amount: plan.available_amount,
        recommended_savings: plan.recommended_savings,
        alternative_savings: Number(value),
        allocations: plan.allocations.map(({ goal, amount }) => ({ goal, amount })),
      });
      if (state.plan === plan) renderWhatIf(result);
    } catch (error) {
      $('whatIfResult').textContent = error.message;
    }
  }, 120);
}

function logActual() {
  const value = Number($('actualInput').value);
  if (!Number.isFinite(value) || value < 0) {
    $('planStatus').textContent = 'Enter the amount you saved (0 or more).';
    return;
  }
  state.actualSavings = [...state.actualSavings, value];
  runPlan();
}

function restartPlan() {
  state.actualSavings = [];
  state.plan = null;
  runPlan();
}

// -- events ---------------------------------------------------------------------

document.addEventListener('click', (event) => {
  const target = event.target.closest('button, [data-profile]');
  if (!target) return;

  if (target.dataset.go) { goTo(target.dataset.go); return; }
  if (target.dataset.profile) { selectProfile(target.dataset.profile); return; }

  if (target.dataset.sample !== undefined) {
    const samples = state.meta.sample_sms;
    const lines = target.dataset.sample === 'all' ? samples.map((s) => s.text) : [samples[Number(target.dataset.sample)].text];
    const input = $('smsInput');
    input.value = [input.value.trim(), ...lines].filter(Boolean).join('\n');
    input.focus();
    return;
  }
  if (target.dataset.smsRemove) {
    const id = Number(target.dataset.smsRemove);
    const removed = state.smsResults.find((item) => item.id === id);
    state.smsResults = state.smsResults.filter((item) => item.id !== id);
    if (removed && removed.include) markForecastStale();
    renderSmsResults();
    renderLedger();
    return;
  }
  if (target.dataset.eventRemove !== undefined) {
    state.events.splice(Number(target.dataset.eventRemove), 1);
    renderEvents();
    runForecast();
    return;
  }
  if (target.dataset.goalRemove !== undefined) {
    state.goals.splice(Number(target.dataset.goalRemove), 1);
    state.goalsEdited = true;
    renderGoals();
    return;
  }
  if (target.dataset.quick) {
    $('actualInput').value = Math.round(state.plan.recommended_savings * Number(target.dataset.quick));
    return;
  }

  switch (target.id) {
    case 'profileNext': goTo('sms'); break;
    case 'analyzeBtn': analyzeSms(); break;
    case 'clearSmsBtn': $('smsInput').value = ''; $('smsStatus').textContent = ''; break;
    case 'smsNext': goTo('forecast'); break;
    case 'addEventBtn': {
      const cost = Number($('eventCost').value);
      if (!Number.isFinite(cost) || cost <= 0) {
        $('forecastStatus').textContent = 'Enter an estimated cost for the event.';
        return;
      }
      state.events.push({ event_type: $('eventType').value, estimated_cost: cost,
        importance: $('eventImportance').value, planning_months: 1 });
      $('eventCost').value = '';
      renderEvents();
      runForecast();
      break;
    }
    case 'forecastBtn': runForecast(); break;
    case 'forecastNext': goTo('plan'); break;
    case 'addGoalBtn':
      state.goals.push({ name: `Goal ${state.goals.length + 1}`, target_amount: 20000, current_savings: 0, deadline_months: 6, priority: 'medium' });
      state.goalsEdited = true;
      renderGoals();
      break;
    case 'planBtn': restartPlan(); break;
    case 'logBtn': logActual(); break;
    case 'replanBtn': restartPlan(); break;
    case 'restartBtn': window.location.reload(); break;
    default: break;
  }
});

document.addEventListener('change', (event) => {
  const target = event.target;
  if (target.dataset.smsInclude) {
    const item = state.smsResults.find((entry) => entry.id === Number(target.dataset.smsInclude));
    item.include = target.checked;
    markForecastStale();
    renderLedger();
  } else if (target.dataset.smsCategory) {
    const item = state.smsResults.find((entry) => entry.id === Number(target.dataset.smsCategory));
    item.expense_category = target.value;
    markForecastStale();
    renderLedger();
  } else if (target.dataset.goal !== undefined) {
    const goal = state.goals[Number(target.dataset.goal)];
    const field = target.dataset.field;
    goal[field] = field === 'name' || field === 'priority' ? target.value : Number(target.value);
    state.goalsEdited = true;
  } else if (target.id === 'eventType') {
    syncEventImportance();
  } else if (target.id === 'incomeInput') {
    runForecast();
  }
});

document.addEventListener('input', (event) => {
  if (event.target.id === 'whatIfRange') scheduleWhatIf(event.target.value);
});

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  if (event.target.id === 'smsInput' && (event.ctrlKey || event.metaKey)) analyzeSms();
  if (event.target.id === 'eventCost') $('addEventBtn').click();
  if (event.target.id === 'actualInput') logActual();
});

document.addEventListener('pointermove', (event) => {
  const segment = event.target.closest('[data-tip]');
  if (segment) Charts.showTip(segment.dataset.tip, event.clientX, event.clientY);
  else if (event.target.closest('.stackbar') === null && !event.target.closest('.chart')) Charts.hideTip();
});

let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.step === 'forecast') drawForecastCharts();
    if (state.step === 'plan') drawPlanCharts();
  }, 150);
});

// -- start ----------------------------------------------------------------------

renderStepper();
renderProfiles();
STEPS.forEach(({ id }) => { $(`step-${id}`).hidden = id !== state.step; });
pollStatus();
