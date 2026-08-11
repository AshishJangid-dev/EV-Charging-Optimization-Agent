"""
ChargeWise AI – Electric Vehicle Charging Optimization System
=============================================================
A multi-agent AI application powered by IBM watsonx.ai Granite Models.
Agents:
  1. Charging Pattern Analysis Agent
  2. Charging Schedule Optimization Agent
  3. Energy Demand Prediction Agent
  4. Cost Optimization Insights Agent
"""

import os
import json
from flask import Flask, render_template_string, request, jsonify
import requests

# ─────────────────────────────────────────────
# Flask App Initialization
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# IBM watsonx.ai Credentials
# Environment variables take priority; hardcoded values are used as fallback.
# ─────────────────────────────────────────────
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY",    "YUWYBehwNf5X-N4ncvYdNHT_UCaoR8WIm8AasikBE9d0")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "4386a850-12d1-4d49-8be9-668de86d3b06")
WATSONX_URL        = os.environ.get("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")

# ─────────────────────────────────────────────
# IBM watsonx.ai – Generate IAM Access Token
# ─────────────────────────────────────────────
def get_iam_token():
    """Exchange IBM API key for a short-lived IAM bearer token."""
    url  = "https://iam.cloud.ibm.com/identity/token"
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WATSONX_API_KEY,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, data=data, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("access_token", "")

# ─────────────────────────────────────────────
# IBM watsonx.ai – Core Generation Function
# Called by ALL agents
# ─────────────────────────────────────────────
def generate_response(prompt: str) -> str:
    """
    Send a prompt to IBM watsonx.ai Granite model and return the generated text.
    Model: ibm/granite-13b-instruct-v2
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return (
            "⚠️  IBM watsonx.ai credentials are not configured. "
            "Please set WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL "
            "as environment variables and restart the application."
        )

    try:
        token = get_iam_token()
        endpoint = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model_id": "ibm/granite-13b-instruct-v2",
            "project_id": WATSONX_PROJECT_ID,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 900,
                "min_new_tokens": 60,
                "repetition_penalty": 1.1,
                "temperature": 0.7,
            },
        }
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result["results"][0]["generated_text"].strip()
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error communicating with IBM watsonx.ai: {exc}"


# ═══════════════════════════════════════════════════════════
# AGENT 1 – Charging Pattern Analysis Agent
# ═══════════════════════════════════════════════════════════
def charging_pattern_agent(data: dict) -> str:
    """
    Analyzes historical EV charging station data.
    Uses IBM Granite to generate intelligent charging pattern analysis.
    """
    prompt = f"""You are an expert EV Charging Pattern Analysis AI Agent.
Analyze the following EV charging station data and provide a comprehensive analysis.

Station ID        : {data.get('station_id', 'N/A')}
Location          : {data.get('location', 'N/A')}
Charging History  : {data.get('history', 'N/A')}
Session Duration  : {data.get('duration', 'N/A')}
Time of Charging  : {data.get('time_of_charging', 'N/A')}
Energy Consumed   : {data.get('energy_kwh', 'N/A')} kWh
Vehicles Charged  : {data.get('vehicles_count', 'N/A')}
Record Period     : {data.get('record_period', 'N/A')}
Additional Notes  : {data.get('notes', 'N/A')}

Provide a detailed analysis covering:
1. Charging Demand Summary
2. Peak Usage Hours identification
3. Station Utilization Rate analysis
4. Daily, Weekly & Monthly Charging Trends
5. AI-generated explanation of overall charging behavior patterns
6. Actionable recommendations for station operators

Format your response with clear section headers and bullet points."""
    # ── IBM watsonx.ai call ──
    return generate_response(prompt)


# ═══════════════════════════════════════════════════════════
# AGENT 2 – Charging Schedule Optimization Agent
# ═══════════════════════════════════════════════════════════
def charging_schedule_agent(data: dict) -> str:
    """
    Recommends optimized charging schedules.
    Uses IBM Granite to generate personalized optimization strategies.
    """
    prompt = f"""You are an expert EV Charging Schedule Optimization AI Agent.
Based on the user's preferences and constraints, generate an optimized charging schedule.

Preferred Charging Time : {data.get('preferred_time', 'N/A')}
Vehicle Type            : {data.get('vehicle_type', 'N/A')}
Battery Capacity        : {data.get('battery_capacity', 'N/A')} kWh
Charging Priority       : {data.get('priority', 'N/A')}
Charging Duration Needed: {data.get('duration_needed', 'N/A')} hours
Available Slots         : {data.get('available_slots', 'N/A')}
User Flexibility        : {data.get('flexibility', 'N/A')}
Grid Constraints        : {data.get('grid_constraints', 'N/A')}

Generate an optimized charging schedule with:
1. Recommended Charging Time Slots (specific hours)
2. Load Balancing Suggestions
3. Queue Reduction Recommendations
4. Station Utilization Improvement Tips
5. Estimated Waiting Time Reduction
6. Smart Scheduling Strategy (off-peak, pre-booking, etc.)
7. Priority-based charging allocation

Format response with clear headings and actionable time-slot recommendations."""
    # ── IBM watsonx.ai call ──
    return generate_response(prompt)


# ═══════════════════════════════════════════════════════════
# AGENT 3 – Energy Demand Prediction Agent
# ═══════════════════════════════════════════════════════════
def energy_demand_prediction_agent(data: dict) -> str:
    """
    Predicts future EV charging demand using historical data.
    Uses IBM Granite for intelligent demand forecasting.
    """
    prompt = f"""You are an expert EV Energy Demand Prediction AI Agent.
Using the provided historical charging data, predict future demand and provide planning insights.

Historical Charging Records : {data.get('historical_records', 'N/A')}
Daily Charging Sessions     : {data.get('daily_sessions', 'N/A')}
Seasonal Trends             : {data.get('seasonal_trends', 'N/A')}
Upcoming Special Events     : {data.get('special_events', 'N/A')}
User Behavior Patterns      : {data.get('user_behavior', 'N/A')}
Current Station Occupancy   : {data.get('occupancy', 'N/A')}
Forecast Horizon            : {data.get('forecast_horizon', 'Next 7 days')}

Provide a comprehensive demand forecast including:
1. Predicted Charging Demand (hourly/daily/weekly)
2. Expected Peak Charging Periods
3. Future Station Utilization Projections
4. Demand Growth Trends (short & long term)
5. Capacity Planning Recommendations
6. AI-generated explanation of demand forecast methodology
7. Risk factors and uncertainties in the forecast
8. Recommended infrastructure investments

Use specific numbers and percentages where appropriate."""
    # ── IBM watsonx.ai call ──
    return generate_response(prompt)


# ═══════════════════════════════════════════════════════════
# AGENT 4 – Cost Optimization Insights Agent
# ═══════════════════════════════════════════════════════════
def cost_optimization_agent(data: dict) -> str:
    """
    Provides AI-powered recommendations to minimize electricity costs.
    Uses IBM Granite to generate intelligent cost optimization insights.
    """
    prompt = f"""You are an expert EV Charging Cost Optimization AI Agent.
Analyze the charging cost parameters and provide actionable cost-saving recommendations.

Electricity Tariff          : {data.get('tariff', 'N/A')} per kWh
Time-of-Use Pricing         : {data.get('tou_pricing', 'N/A')}
Charging Duration           : {data.get('duration', 'N/A')} hours
Total Energy Consumption    : {data.get('energy_kwh', 'N/A')} kWh
Current Monthly Cost        : {data.get('monthly_cost', 'N/A')}
Charging Preferences        : {data.get('preferences', 'N/A')}
User Schedule Flexibility   : {data.get('flexibility', 'N/A')}
Renewable Energy Available  : {data.get('renewable', 'N/A')}

Generate comprehensive cost optimization recommendations:
1. Low-Cost Charging Schedules (specific time windows)
2. Off-Peak Charging Recommendations with estimated savings
3. Monthly Charging Cost Estimate (before & after optimization)
4. Estimated Electricity Savings (% and currency amount)
5. Smart Tariff Utilization Strategies
6. Renewable Energy Integration opportunities
7. Dynamic Pricing response strategies
8. Long-term cost reduction roadmap

Include specific dollar/percentage savings estimates where possible."""
    # ── IBM watsonx.ai call ──
    return generate_response(prompt)


# ═══════════════════════════════════════════════════════════
# AGENT ORCHESTRATOR
# Routes requests to the appropriate specialized agent
# ═══════════════════════════════════════════════════════════
def orchestrator(agent_name: str, data: dict) -> str:
    """
    Intelligently routes user requests to the appropriate AI agent.
    Supported agents: pattern | schedule | demand | cost
    """
    routing = {
        "pattern":  charging_pattern_agent,
        "schedule": charging_schedule_agent,
        "demand":   energy_demand_prediction_agent,
        "cost":     cost_optimization_agent,
    }
    agent_fn = routing.get(agent_name)
    if not agent_fn:
        return f"❌ Unknown agent: '{agent_name}'. Choose from: pattern, schedule, demand, cost."
    return agent_fn(data)


# ═══════════════════════════════════════════════════════════
# HTML TEMPLATES (render_template_string – no separate files)
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# BASE LAYOUT  (shared sidebar + nav)
# ─────────────────────────────────────────────
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ChargeWise AI – {{ page_title }}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --sidebar-w: 270px;
    --brand-1: #00b4d8;
    --brand-2: #0077b6;
    --brand-3: #023e8a;
    --accent:  #48cae4;
    --green:   #52b788;
    --yellow:  #f4a261;
    --dark-bg: #0d1b2a;
    --card-bg: #112240;
    --text-light: #ccd6f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2f4a 100%);
    color: var(--text-light);
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  #sidebar {
    position: fixed; top: 0; left: 0;
    width: var(--sidebar-w); height: 100vh;
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
    border-right: 1px solid rgba(0,180,216,.18);
    display: flex; flex-direction: column;
    z-index: 1000; overflow-y: auto;
    transition: transform .3s ease;
  }
  .sidebar-brand {
    padding: 1.4rem 1.2rem 1rem;
    border-bottom: 1px solid rgba(0,180,216,.15);
  }
  .sidebar-brand h5 {
    font-size: 1.05rem; font-weight: 700;
    color: var(--brand-1); letter-spacing: .5px;
  }
  .sidebar-brand p { font-size: .72rem; color: #6c8ebf; margin-top: .15rem; }
  .sidebar-brand .logo-icon {
    font-size: 2rem; color: var(--brand-1);
    display: block; margin-bottom: .4rem;
  }
  .sidebar-section {
    padding: .55rem 1rem .2rem;
    font-size: .68rem; font-weight: 700;
    color: #4a6fa5; text-transform: uppercase; letter-spacing: 1px;
  }
  .nav-link-sb {
    display: flex; align-items: center; gap: .7rem;
    padding: .62rem 1.2rem; color: #8fa8cc;
    text-decoration: none; font-size: .87rem; font-weight: 500;
    border-left: 3px solid transparent;
    transition: all .2s;
  }
  .nav-link-sb i { font-size: 1.05rem; }
  .nav-link-sb:hover, .nav-link-sb.active {
    color: #fff; background: rgba(0,180,216,.1);
    border-left-color: var(--brand-1);
  }
  .sidebar-footer {
    margin-top: auto; padding: 1rem 1.2rem;
    border-top: 1px solid rgba(0,180,216,.1);
    font-size: .72rem; color: #4a6fa5; text-align: center;
  }

  /* ── Main content ── */
  #main-content {
    margin-left: var(--sidebar-w);
    min-height: 100vh;
    padding: 0;
  }
  .topbar {
    background: rgba(13,27,42,.85);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(0,180,216,.15);
    padding: .75rem 1.8rem;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  .topbar-title { font-size: 1.05rem; font-weight: 600; color: #fff; }
  .topbar-badge {
    font-size: .7rem; padding: .28rem .75rem;
    border-radius: 20px;
    background: rgba(0,180,216,.15);
    border: 1px solid rgba(0,180,216,.35);
    color: var(--brand-1);
  }
  .page-content { padding: 1.8rem 2rem; }

  /* ── Cards ── */
  .ev-card {
    background: var(--card-bg);
    border: 1px solid rgba(0,180,216,.2);
    border-radius: 14px; padding: 1.4rem;
    height: 100%;
    transition: transform .2s, box-shadow .2s;
  }
  .ev-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,180,216,.15);
  }
  .ev-card .card-icon {
    font-size: 2.2rem; margin-bottom: .8rem;
  }
  .ev-card h6 { font-size: .78rem; color: #4a6fa5; text-transform: uppercase; letter-spacing: .8px; }
  .ev-card .stat-val { font-size: 1.7rem; font-weight: 700; color: #fff; line-height: 1.1; }
  .ev-card .stat-label { font-size: .78rem; color: #8fa8cc; margin-top: .2rem; }

  .feature-card {
    background: linear-gradient(135deg, #0a1e36 0%, #112240 100%);
    border: 1px solid rgba(0,180,216,.25);
    border-radius: 16px; padding: 1.8rem;
    height: 100%;
    transition: transform .25s, box-shadow .25s;
    cursor: pointer; text-decoration: none; display: block; color: inherit;
  }
  .feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 35px rgba(0,180,216,.2);
    border-color: var(--brand-1);
    color: inherit;
  }
  .feature-card .fc-icon {
    width: 56px; height: 56px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin-bottom: 1rem;
  }
  .feature-card h5 { font-size: 1rem; font-weight: 700; color: #fff; margin-bottom: .5rem; }
  .feature-card p { font-size: .83rem; color: #8fa8cc; line-height: 1.6; }
  .badge-agent {
    font-size: .65rem; padding: .25rem .6rem; border-radius: 20px;
    background: rgba(0,180,216,.15); color: var(--brand-1);
    border: 1px solid rgba(0,180,216,.3); font-weight: 600;
  }

  /* ── Forms ── */
  .form-label { color: #8fa8cc; font-size: .85rem; font-weight: 500; margin-bottom: .3rem; }
  .form-control, .form-select {
    background: #0a1628; border: 1px solid rgba(0,180,216,.25);
    color: #ccd6f6; border-radius: 8px;
    transition: border-color .2s, box-shadow .2s;
  }
  .form-control:focus, .form-select:focus {
    background: #0a1628; color: #fff;
    border-color: var(--brand-1);
    box-shadow: 0 0 0 3px rgba(0,180,216,.15);
  }
  .form-control::placeholder { color: #4a6fa5; }
  .form-select option { background: #0d1b2a; }
  .btn-primary-ev {
    background: linear-gradient(135deg, var(--brand-2), var(--brand-1));
    border: none; border-radius: 8px;
    color: #fff; font-weight: 600; font-size: .9rem;
    padding: .65rem 1.6rem;
    transition: opacity .2s, transform .15s;
  }
  .btn-primary-ev:hover { opacity: .88; transform: translateY(-1px); color: #fff; }
  .btn-primary-ev:disabled { opacity: .5; }

  /* ── Result box ── */
  .result-box {
    background: #07111e;
    border: 1px solid rgba(0,180,216,.25);
    border-radius: 12px; padding: 1.5rem;
    white-space: pre-wrap; word-break: break-word;
    font-size: .88rem; line-height: 1.75;
    color: #ccd6f6; min-height: 120px;
    display: none;
  }
  .result-box.visible { display: block; }
  .result-header {
    display: flex; align-items: center; gap: .6rem;
    margin-bottom: .75rem; padding-bottom: .6rem;
    border-bottom: 1px solid rgba(0,180,216,.15);
    font-size: .8rem; font-weight: 600; color: var(--brand-1);
    text-transform: uppercase; letter-spacing: .8px;
  }

  /* ── Section headers ── */
  .section-header {
    padding-bottom: .8rem; margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(0,180,216,.15);
  }
  .section-header h4 { font-size: 1.2rem; font-weight: 700; color: #fff; }
  .section-header p { font-size: .85rem; color: #8fa8cc; margin-top: .25rem; }

  /* ── Gradient text ── */
  .gradient-text {
    background: linear-gradient(90deg, var(--brand-1), #90e0ef);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* ── Spinner ── */
  .spinner-border-sm { width: 1rem; height: 1rem; }

  /* ── Chart containers ── */
  .chart-container { position: relative; height: 200px; }

  /* ── Mobile ── */
  @media (max-width: 768px) {
    #sidebar { transform: translateX(-100%); }
    #sidebar.show { transform: translateX(0); }
    #main-content { margin-left: 0; }
    .page-content { padding: 1rem; }
  }

  /* ── About page ── */
  .about-card {
    background: #0a1e36; border: 1px solid rgba(0,180,216,.2);
    border-radius: 14px; padding: 1.5rem;
  }
  .step-badge {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-2), var(--brand-1));
    display: flex; align-items: center; justify-content: center;
    font-size: .85rem; font-weight: 700; flex-shrink: 0;
  }

  /* scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #0a1628; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
</head>
<body>

<!-- ── SIDEBAR ── -->
<nav id="sidebar">
  <div class="sidebar-brand">
    <span class="logo-icon"><i class="bi bi-lightning-charge-fill"></i></span>
    <h5>ChargeWise AI</h5>
    <p>EV Charging Optimization System</p>
  </div>

  <div class="sidebar-section mt-2">Navigation</div>
  <a href="/" class="nav-link-sb {{ 'active' if active_page=='home' }}">
    <i class="bi bi-house-door-fill"></i> Home Dashboard
  </a>

  <div class="sidebar-section mt-1">AI Agents</div>
  <a href="/pattern" class="nav-link-sb {{ 'active' if active_page=='pattern' }}">
    <i class="bi bi-graph-up-arrow"></i> Pattern Analysis
  </a>
  <a href="/schedule" class="nav-link-sb {{ 'active' if active_page=='schedule' }}">
    <i class="bi bi-calendar2-check-fill"></i> Schedule Optimizer
  </a>
  <a href="/demand" class="nav-link-sb {{ 'active' if active_page=='demand' }}">
    <i class="bi bi-bar-chart-fill"></i> Demand Prediction
  </a>
  <a href="/cost" class="nav-link-sb {{ 'active' if active_page=='cost' }}">
    <i class="bi bi-currency-dollar"></i> Cost Optimization
  </a>

  <div class="sidebar-section mt-1">Info</div>
  <a href="/about" class="nav-link-sb {{ 'active' if active_page=='about' }}">
    <i class="bi bi-info-circle-fill"></i> About
  </a>

  <div class="sidebar-footer">
    <i class="bi bi-cpu-fill me-1"></i>Powered by IBM Granite Models<br/>
    <span style="color:#2a4a6f">IBM watsonx.ai</span>
  </div>
</nav>

<!-- ── MAIN CONTENT ── -->
<div id="main-content">
  <div class="topbar">
    <button class="btn btn-sm d-md-none me-2" onclick="document.getElementById('sidebar').classList.toggle('show')"
            style="background:rgba(0,180,216,.15);border:1px solid rgba(0,180,216,.3);color:#fff;">
      <i class="bi bi-list"></i>
    </button>
    <span class="topbar-title"><i class="bi bi-lightning-charge-fill me-2" style="color:var(--brand-1)"></i>{{ page_title }}</span>
    <span class="topbar-badge"><i class="bi bi-cpu me-1"></i>IBM Granite · watsonx.ai</span>
  </div>

  <div class="page-content">
    {% block content %}{% endblock %}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

# ─────────────────────────────────────────────
# HOME PAGE TEMPLATE
# ─────────────────────────────────────────────
HOME_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
<!-- Hero -->
<div class="mb-4">
  <h2 class="fw-bold mb-1" style="font-size:1.7rem;">
    Welcome to <span class="gradient-text">ChargeWise AI</span>
  </h2>
  <p style="color:#8fa8cc;font-size:.92rem;max-width:660px;">
    An Agentic AI system powered by <strong style="color:var(--brand-1)">IBM watsonx.ai Granite Models</strong>
    that intelligently analyzes EV charging patterns, optimizes schedules,
    predicts energy demand, and reduces electricity costs for a sustainable future.
  </p>
</div>

<!-- KPI Cards -->
<div class="row g-3 mb-4">
  <div class="col-6 col-md-3">
    <div class="ev-card text-center">
      <div class="card-icon" style="color:#00b4d8"><i class="bi bi-lightning-charge-fill"></i></div>
      <div class="stat-val">4</div>
      <div class="stat-label">AI Agents Active</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="ev-card text-center">
      <div class="card-icon" style="color:#52b788"><i class="bi bi-ev-front-fill"></i></div>
      <div class="stat-val">IBM</div>
      <div class="stat-label">Granite Foundation Model</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="ev-card text-center">
      <div class="card-icon" style="color:#f4a261"><i class="bi bi-battery-charging"></i></div>
      <div class="stat-val">30%</div>
      <div class="stat-label">Avg. Cost Savings</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="ev-card text-center">
      <div class="card-icon" style="color:#9b72cf"><i class="bi bi-globe-americas"></i></div>
      <div class="stat-val">100%</div>
      <div class="stat-label">AI-Driven Insights</div>
    </div>
  </div>
</div>

<!-- Charts Row -->
<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="ev-card">
      <h6 class="mb-3"><i class="bi bi-bar-chart-fill me-2" style="color:var(--brand-1)"></i>Sample Charging Demand (24h)</h6>
      <div class="chart-container"><canvas id="demandChart"></canvas></div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="ev-card">
      <h6 class="mb-3"><i class="bi bi-pie-chart-fill me-2" style="color:#52b788"></i>Station Utilization Breakdown</h6>
      <div class="chart-container"><canvas id="utilChart"></canvas></div>
    </div>
  </div>
</div>

<!-- Feature Cards -->
<div class="section-header">
  <h4><i class="bi bi-grid-3x3-gap-fill me-2" style="color:var(--brand-1)"></i>AI Agent Suite</h4>
  <p>Select an agent below to begin your EV charging optimization journey</p>
</div>
<div class="row g-3">
  <div class="col-md-6 col-lg-3">
    <a href="/pattern" class="feature-card">
      <div class="fc-icon" style="background:rgba(0,180,216,.12);">
        <i class="bi bi-graph-up-arrow" style="color:#00b4d8"></i>
      </div>
      <span class="badge-agent mb-2">Agent 1</span>
      <h5>Pattern Analysis</h5>
      <p>Analyze historical charging behavior, peak demand, and station utilization using IBM Granite AI.</p>
    </a>
  </div>
  <div class="col-md-6 col-lg-3">
    <a href="/schedule" class="feature-card">
      <div class="fc-icon" style="background:rgba(82,183,136,.12);">
        <i class="bi bi-calendar2-check-fill" style="color:#52b788"></i>
      </div>
      <span class="badge-agent mb-2" style="color:#52b788;border-color:rgba(82,183,136,.3);background:rgba(82,183,136,.1);">Agent 2</span>
      <h5>Schedule Optimizer</h5>
      <p>Get AI-generated optimized charging schedules balancing demand, congestion, and grid efficiency.</p>
    </a>
  </div>
  <div class="col-md-6 col-lg-3">
    <a href="/demand" class="feature-card">
      <div class="fc-icon" style="background:rgba(244,162,97,.12);">
        <i class="bi bi-bar-chart-fill" style="color:#f4a261"></i>
      </div>
      <span class="badge-agent mb-2" style="color:#f4a261;border-color:rgba(244,162,97,.3);background:rgba(244,162,97,.1);">Agent 3</span>
      <h5>Demand Prediction</h5>
      <p>Predict future charging demand with AI-powered forecasting for smarter capacity planning.</p>
    </a>
  </div>
  <div class="col-md-6 col-lg-3">
    <a href="/cost" class="feature-card">
      <div class="fc-icon" style="background:rgba(155,114,207,.12);">
        <i class="bi bi-currency-dollar" style="color:#9b72cf"></i>
      </div>
      <span class="badge-agent mb-2" style="color:#9b72cf;border-color:rgba(155,114,207,.3);background:rgba(155,114,207,.1);">Agent 4</span>
      <h5>Cost Optimization</h5>
      <p>Minimize electricity costs with smart tariff strategies and off-peak charging recommendations.</p>
    </a>
  </div>
</div>

<!-- IBM Watsonx Banner -->
<div class="mt-4 p-3 rounded-3" style="background:rgba(0,180,216,.07);border:1px solid rgba(0,180,216,.2);">
  <div class="d-flex align-items-center gap-3 flex-wrap">
    <i class="bi bi-cpu-fill" style="font-size:2rem;color:var(--brand-1)"></i>
    <div>
      <div style="font-weight:700;color:#fff;">Powered by IBM watsonx.ai Granite Models</div>
      <div style="font-size:.8rem;color:#8fa8cc;">
        All four agents use IBM's enterprise-grade Granite foundation models via the watsonx.ai API
        for responsible, explainable, and high-quality AI generation.
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
// Demand chart
const dCtx = document.getElementById('demandChart').getContext('2d');
new Chart(dCtx, {
  type: 'line',
  data: {
    labels: ['0','2','4','6','8','10','12','14','16','18','20','22'],
    datasets: [{
      label: 'kWh Consumed',
      data: [12,8,5,4,10,28,40,38,35,45,52,30],
      borderColor: '#00b4d8', backgroundColor: 'rgba(0,180,216,.12)',
      borderWidth: 2, tension: 0.4, fill: true, pointRadius: 3
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#8fa8cc', font: { size: 11 } } } },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#6c8ebf' } },
      y: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#6c8ebf' } }
    }
  }
});
// Utilization pie
const uCtx = document.getElementById('utilChart').getContext('2d');
new Chart(uCtx, {
  type: 'doughnut',
  data: {
    labels: ['Occupied', 'Available', 'Reserved', 'Maintenance'],
    datasets: [{
      data: [52, 28, 14, 6],
      backgroundColor: ['#00b4d8','#52b788','#f4a261','#9b72cf'],
      borderColor: '#0d1b2a', borderWidth: 3
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { color: '#8fa8cc', font: { size: 11 }, padding: 12 } } }
  }
});
</script>
""")

# ─────────────────────────────────────────────
# PATTERN ANALYSIS PAGE
# ─────────────────────────────────────────────
PATTERN_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
<div class="section-header">
  <h4><i class="bi bi-graph-up-arrow me-2" style="color:#00b4d8"></i>Charging Pattern Analysis</h4>
  <p>Agent 1 · Enter your charging station data to receive AI-powered behavioral analysis from IBM Granite</p>
</div>
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ev-card">
      <h6 class="mb-3" style="color:#00b4d8;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-input-cursor-text me-2"></i>Station Data Input
      </h6>
      <form id="patternForm">
        <div class="mb-3">
          <label class="form-label">Charging Station ID</label>
          <input type="text" class="form-control" name="station_id" placeholder="e.g. EVCS-007-NYC"/>
        </div>
        <div class="mb-3">
          <label class="form-label">Station Location</label>
          <input type="text" class="form-control" name="location" placeholder="e.g. Manhattan, NY – Parking Lot B"/>
        </div>
        <div class="mb-3">
          <label class="form-label">Charging History Summary</label>
          <textarea class="form-control" name="history" rows="3"
            placeholder="e.g. 150 sessions last month, avg 45 min per session..."></textarea>
        </div>
        <div class="row g-2 mb-3">
          <div class="col-6">
            <label class="form-label">Avg Session Duration</label>
            <input type="text" class="form-control" name="duration" placeholder="e.g. 45 minutes"/>
          </div>
          <div class="col-6">
            <label class="form-label">Peak Charging Time</label>
            <input type="text" class="form-control" name="time_of_charging" placeholder="e.g. 7AM–9AM, 5PM–7PM"/>
          </div>
        </div>
        <div class="row g-2 mb-3">
          <div class="col-6">
            <label class="form-label">Energy Consumed (kWh)</label>
            <input type="text" class="form-control" name="energy_kwh" placeholder="e.g. 4500 kWh/month"/>
          </div>
          <div class="col-6">
            <label class="form-label">Vehicles Charged</label>
            <input type="text" class="form-control" name="vehicles_count" placeholder="e.g. 320 vehicles"/>
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label">Record Period</label>
          <select class="form-select" name="record_period">
            <option value="Last 7 days">Last 7 days</option>
            <option value="Last 30 days" selected>Last 30 days</option>
            <option value="Last 3 months">Last 3 months</option>
            <option value="Last 6 months">Last 6 months</option>
            <option value="Last 12 months">Last 12 months</option>
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Additional Notes</label>
          <textarea class="form-control" name="notes" rows="2" placeholder="Any relevant context..."></textarea>
        </div>
        <button type="submit" class="btn btn-primary-ev w-100" id="patternBtn">
          <i class="bi bi-cpu-fill me-2"></i>Analyze with IBM Granite AI
        </button>
      </form>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ev-card h-100 d-flex flex-column">
      <h6 class="mb-3" style="color:#00b4d8;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-stars me-2"></i>AI Analysis Result
      </h6>
      <div id="patternLoading" class="text-center py-4 d-none">
        <div class="spinner-border" style="color:var(--brand-1);width:2.5rem;height:2.5rem;"></div>
        <p class="mt-3" style="color:#8fa8cc;font-size:.85rem;">IBM Granite is analyzing your data…</p>
      </div>
      <div id="patternResult" class="result-box flex-grow-1"></div>
      <div id="patternPlaceholder" class="text-center py-5" style="color:#4a6fa5;">
        <i class="bi bi-graph-up" style="font-size:3rem;"></i>
        <p class="mt-2" style="font-size:.85rem;">Submit your station data to receive AI-powered pattern analysis</p>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
document.getElementById('patternForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('patternBtn');
  const loading = document.getElementById('patternLoading');
  const result = document.getElementById('patternResult');
  const placeholder = document.getElementById('patternPlaceholder');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing…';
  loading.classList.remove('d-none');
  result.classList.remove('visible');
  placeholder.style.display = 'none';
  const data = Object.fromEntries(new FormData(this).entries());
  try {
    const resp = await fetch('/api/agent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({agent: 'pattern', data})
    });
    const json = await resp.json();
    result.innerHTML = '<div class="result-header"><i class="bi bi-cpu-fill"></i>IBM Granite Analysis</div>' + escapeHtml(json.result);
    result.classList.add('visible');
  } catch(err) {
    result.innerHTML = '❌ Request failed: ' + err.message;
    result.classList.add('visible');
  } finally {
    loading.classList.add('d-none');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-cpu-fill me-2"></i>Analyze with IBM Granite AI';
  }
});
function escapeHtml(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }
</script>
""")

# ─────────────────────────────────────────────
# SCHEDULE OPTIMIZER PAGE
# ─────────────────────────────────────────────
SCHEDULE_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
<div class="section-header">
  <h4><i class="bi bi-calendar2-check-fill me-2" style="color:#52b788"></i>Charging Schedule Optimizer</h4>
  <p>Agent 2 · Provide your charging requirements and receive an AI-optimized schedule from IBM Granite</p>
</div>
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ev-card">
      <h6 class="mb-3" style="color:#52b788;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-sliders me-2"></i>Charging Preferences
      </h6>
      <form id="scheduleForm">
        <div class="mb-3">
          <label class="form-label">Vehicle Type</label>
          <select class="form-select" name="vehicle_type">
            <option>Tesla Model 3</option>
            <option>Tesla Model Y</option>
            <option>Chevy Bolt EV</option>
            <option>Nissan Leaf</option>
            <option>Ford F-150 Lightning</option>
            <option>Rivian R1T</option>
            <option>BMW iX</option>
            <option>Hyundai IONIQ 6</option>
            <option>Fleet EV (Mixed)</option>
          </select>
        </div>
        <div class="row g-2 mb-3">
          <div class="col-6">
            <label class="form-label">Battery Capacity (kWh)</label>
            <input type="text" class="form-control" name="battery_capacity" placeholder="e.g. 75"/>
          </div>
          <div class="col-6">
            <label class="form-label">Duration Needed (hrs)</label>
            <input type="text" class="form-control" name="duration_needed" placeholder="e.g. 3"/>
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label">Preferred Charging Time</label>
          <input type="text" class="form-control" name="preferred_time" placeholder="e.g. After 10 PM, Early morning"/>
        </div>
        <div class="mb-3">
          <label class="form-label">Charging Priority</label>
          <select class="form-select" name="priority">
            <option>High – Need full charge ASAP</option>
            <option>Medium – Flexible by a few hours</option>
            <option>Low – Overnight/next-day is fine</option>
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Available Charging Slots</label>
          <input type="text" class="form-control" name="available_slots" placeholder="e.g. 6 slots open between 8PM–6AM"/>
        </div>
        <div class="mb-3">
          <label class="form-label">User Flexibility</label>
          <select class="form-select" name="flexibility">
            <option>Very flexible – anytime overnight</option>
            <option>Moderately flexible – within 4 hrs</option>
            <option>Minimal flexibility – specific window</option>
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Grid Constraints / Notes</label>
          <textarea class="form-control" name="grid_constraints" rows="2"
            placeholder="e.g. Grid capacity limit 150 kW, solar available 9AM–3PM…"></textarea>
        </div>
        <button type="submit" class="btn btn-primary-ev w-100" id="scheduleBtn"
          style="background:linear-gradient(135deg,#1b6b45,#52b788);">
          <i class="bi bi-calendar-check-fill me-2"></i>Generate Optimized Schedule
        </button>
      </form>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ev-card h-100 d-flex flex-column">
      <h6 class="mb-3" style="color:#52b788;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-stars me-2"></i>Optimized Schedule
      </h6>
      <div id="scheduleLoading" class="text-center py-4 d-none">
        <div class="spinner-border" style="color:#52b788;width:2.5rem;height:2.5rem;"></div>
        <p class="mt-3" style="color:#8fa8cc;font-size:.85rem;">IBM Granite is generating your schedule…</p>
      </div>
      <div id="scheduleResult" class="result-box flex-grow-1" style="border-color:rgba(82,183,136,.25)"></div>
      <div id="schedulePlaceholder" class="text-center py-5" style="color:#4a6fa5;">
        <i class="bi bi-calendar3" style="font-size:3rem;"></i>
        <p class="mt-2" style="font-size:.85rem;">Fill in your preferences to get an AI-generated charging schedule</p>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
document.getElementById('scheduleForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('scheduleBtn');
  const loading = document.getElementById('scheduleLoading');
  const result = document.getElementById('scheduleResult');
  const placeholder = document.getElementById('schedulePlaceholder');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Optimizing…';
  loading.classList.remove('d-none');
  result.classList.remove('visible');
  placeholder.style.display = 'none';
  const data = Object.fromEntries(new FormData(this).entries());
  try {
    const resp = await fetch('/api/agent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({agent: 'schedule', data})
    });
    const json = await resp.json();
    result.innerHTML = '<div class="result-header" style="color:#52b788"><i class="bi bi-cpu-fill"></i>IBM Granite Schedule</div>' + escapeHtml(json.result);
    result.classList.add('visible');
  } catch(err) {
    result.innerHTML = '❌ Request failed: ' + err.message;
    result.classList.add('visible');
  } finally {
    loading.classList.add('d-none');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-calendar-check-fill me-2"></i>Generate Optimized Schedule';
  }
});
function escapeHtml(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }
</script>
""")

# ─────────────────────────────────────────────
# DEMAND PREDICTION PAGE
# ─────────────────────────────────────────────
DEMAND_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
<div class="section-header">
  <h4><i class="bi bi-bar-chart-fill me-2" style="color:#f4a261"></i>Energy Demand Prediction</h4>
  <p>Agent 3 · Forecast future EV charging demand with IBM Granite AI-powered predictive analytics</p>
</div>
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ev-card">
      <h6 class="mb-3" style="color:#f4a261;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-database-fill me-2"></i>Historical Data Input
      </h6>
      <form id="demandForm">
        <div class="mb-3">
          <label class="form-label">Historical Charging Records</label>
          <textarea class="form-control" name="historical_records" rows="3"
            placeholder="e.g. Jan: 4200 kWh, Feb: 4800 kWh, Mar: 5100 kWh…"></textarea>
        </div>
        <div class="row g-2 mb-3">
          <div class="col-6">
            <label class="form-label">Daily Sessions (avg)</label>
            <input type="text" class="form-control" name="daily_sessions" placeholder="e.g. 85 sessions/day"/>
          </div>
          <div class="col-6">
            <label class="form-label">Current Occupancy (%)</label>
            <input type="text" class="form-control" name="occupancy" placeholder="e.g. 72%"/>
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label">Seasonal Trends</label>
          <select class="form-select" name="seasonal_trends">
            <option>Summer – higher demand (vacations, heat)</option>
            <option>Winter – moderate demand (cold batteries)</option>
            <option>Spring/Fall – baseline demand</option>
            <option>Year-round consistent demand</option>
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Upcoming Special Events</label>
          <input type="text" class="form-control" name="special_events"
            placeholder="e.g. Music festival nearby this weekend, holiday Monday"/>
        </div>
        <div class="mb-3">
          <label class="form-label">User Behavior Notes</label>
          <textarea class="form-control" name="user_behavior" rows="2"
            placeholder="e.g. 60% commuters charge Mon-Fri mornings, 40% weekend leisure…"></textarea>
        </div>
        <div class="mb-3">
          <label class="form-label">Forecast Horizon</label>
          <select class="form-select" name="forecast_horizon">
            <option>Next 24 hours</option>
            <option>Next 7 days</option>
            <option>Next 30 days</option>
            <option>Next 3 months</option>
            <option>Next 12 months</option>
          </select>
        </div>
        <button type="submit" class="btn btn-primary-ev w-100" id="demandBtn"
          style="background:linear-gradient(135deg,#b54e10,#f4a261);">
          <i class="bi bi-bar-chart-line-fill me-2"></i>Predict Demand with IBM AI
        </button>
      </form>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ev-card h-100 d-flex flex-column">
      <h6 class="mb-3" style="color:#f4a261;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-stars me-2"></i>Demand Forecast
      </h6>
      <div id="demandLoading" class="text-center py-4 d-none">
        <div class="spinner-border" style="color:#f4a261;width:2.5rem;height:2.5rem;"></div>
        <p class="mt-3" style="color:#8fa8cc;font-size:.85rem;">IBM Granite is forecasting demand…</p>
      </div>
      <div id="demandResult" class="result-box flex-grow-1" style="border-color:rgba(244,162,97,.25)"></div>
      <div id="demandPlaceholder" class="text-center py-5" style="color:#4a6fa5;">
        <i class="bi bi-graph-up" style="font-size:3rem;color:#f4a261;opacity:.4"></i>
        <p class="mt-2" style="font-size:.85rem;">Provide historical data to generate demand forecasts</p>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
document.getElementById('demandForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('demandBtn');
  const loading = document.getElementById('demandLoading');
  const result = document.getElementById('demandResult');
  const placeholder = document.getElementById('demandPlaceholder');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Forecasting…';
  loading.classList.remove('d-none');
  result.classList.remove('visible');
  placeholder.style.display = 'none';
  const data = Object.fromEntries(new FormData(this).entries());
  try {
    const resp = await fetch('/api/agent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({agent: 'demand', data})
    });
    const json = await resp.json();
    result.innerHTML = '<div class="result-header" style="color:#f4a261"><i class="bi bi-cpu-fill"></i>IBM Granite Forecast</div>' + escapeHtml(json.result);
    result.classList.add('visible');
  } catch(err) {
    result.innerHTML = '❌ Request failed: ' + err.message;
    result.classList.add('visible');
  } finally {
    loading.classList.add('d-none');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-bar-chart-line-fill me-2"></i>Predict Demand with IBM AI';
  }
});
function escapeHtml(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }
</script>
""")

# ─────────────────────────────────────────────
# COST OPTIMIZATION PAGE
# ─────────────────────────────────────────────
COST_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
<div class="section-header">
  <h4><i class="bi bi-currency-dollar me-2" style="color:#9b72cf"></i>Cost Optimization Insights</h4>
  <p>Agent 4 · Get AI-powered electricity cost reduction strategies from IBM Granite</p>
</div>
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ev-card">
      <h6 class="mb-3" style="color:#9b72cf;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-coin me-2"></i>Cost & Tariff Details
      </h6>
      <form id="costForm">
        <div class="row g-2 mb-3">
          <div class="col-6">
            <label class="form-label">Electricity Tariff ($/kWh)</label>
            <input type="text" class="form-control" name="tariff" placeholder="e.g. 0.18"/>
          </div>
          <div class="col-6">
            <label class="form-label">Energy Consumed (kWh)</label>
            <input type="text" class="form-control" name="energy_kwh" placeholder="e.g. 350 kWh/month"/>
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label">Time-of-Use Pricing Structure</label>
          <textarea class="form-control" name="tou_pricing" rows="2"
            placeholder="e.g. Peak: $0.28/kWh 2PM-8PM | Off-peak: $0.10/kWh 10PM-6AM | Mid: $0.18/kWh rest"></textarea>
        </div>
        <div class="row g-2 mb-3">
          <div class="col-6">
            <label class="form-label">Charging Duration (hrs/day)</label>
            <input type="text" class="form-control" name="duration" placeholder="e.g. 4 hrs/day"/>
          </div>
          <div class="col-6">
            <label class="form-label">Current Monthly Cost ($)</label>
            <input type="text" class="form-control" name="monthly_cost" placeholder="e.g. $95"/>
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label">Charging Preferences</label>
          <textarea class="form-control" name="preferences" rows="2"
            placeholder="e.g. Commuter vehicle, need 80% charge ready by 7AM weekdays…"></textarea>
        </div>
        <div class="mb-3">
          <label class="form-label">Schedule Flexibility</label>
          <select class="form-select" name="flexibility">
            <option>Very flexible – optimize freely overnight</option>
            <option>Moderate – can shift by 2-3 hrs</option>
            <option>Low – limited to specific windows</option>
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Renewable Energy Available?</label>
          <select class="form-select" name="renewable">
            <option>No renewable energy</option>
            <option>Solar panels on-site</option>
            <option>Grid green energy tariff</option>
            <option>Battery storage system</option>
            <option>Solar + Battery storage</option>
          </select>
        </div>
        <button type="submit" class="btn btn-primary-ev w-100" id="costBtn"
          style="background:linear-gradient(135deg,#5e35b1,#9b72cf);">
          <i class="bi bi-lightbulb-fill me-2"></i>Optimize Costs with IBM AI
        </button>
      </form>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ev-card h-100 d-flex flex-column">
      <h6 class="mb-3" style="color:#9b72cf;text-transform:uppercase;font-size:.75rem;letter-spacing:.8px;">
        <i class="bi bi-stars me-2"></i>Cost Optimization Report
      </h6>
      <div id="costLoading" class="text-center py-4 d-none">
        <div class="spinner-border" style="color:#9b72cf;width:2.5rem;height:2.5rem;"></div>
        <p class="mt-3" style="color:#8fa8cc;font-size:.85rem;">IBM Granite is computing savings…</p>
      </div>
      <div id="costResult" class="result-box flex-grow-1" style="border-color:rgba(155,114,207,.25)"></div>
      <div id="costPlaceholder" class="text-center py-5" style="color:#4a6fa5;">
        <i class="bi bi-piggy-bank" style="font-size:3rem;color:#9b72cf;opacity:.4"></i>
        <p class="mt-2" style="font-size:.85rem;">Enter your tariff details to receive AI cost-saving recommendations</p>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
document.getElementById('costForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('costBtn');
  const loading = document.getElementById('costLoading');
  const result = document.getElementById('costResult');
  const placeholder = document.getElementById('costPlaceholder');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Optimizing…';
  loading.classList.remove('d-none');
  result.classList.remove('visible');
  placeholder.style.display = 'none';
  const data = Object.fromEntries(new FormData(this).entries());
  try {
    const resp = await fetch('/api/agent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({agent: 'cost', data})
    });
    const json = await resp.json();
    result.innerHTML = '<div class="result-header" style="color:#9b72cf"><i class="bi bi-cpu-fill"></i>IBM Granite Insights</div>' + escapeHtml(json.result);
    result.classList.add('visible');
  } catch(err) {
    result.innerHTML = '❌ Request failed: ' + err.message;
    result.classList.add('visible');
  } finally {
    loading.classList.add('d-none');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-lightbulb-fill me-2"></i>Optimize Costs with IBM AI';
  }
});
function escapeHtml(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }
</script>
""")

# ─────────────────────────────────────────────
# ABOUT PAGE
# ─────────────────────────────────────────────
ABOUT_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
<div class="section-header">
  <h4><i class="bi bi-info-circle-fill me-2" style="color:var(--brand-1)"></i>About ChargeWise AI</h4>
  <p>Architecture, technology stack, and IBM watsonx.ai integration details</p>
</div>

<!-- Overview -->
<div class="row g-3 mb-4">
  <div class="col-md-12">
    <div class="about-card">
      <h5 class="mb-2" style="color:#fff;"><i class="bi bi-lightning-charge-fill me-2" style="color:var(--brand-1)"></i>What is ChargeWise AI?</h5>
      <p style="color:#8fa8cc;font-size:.9rem;line-height:1.8;">
        ChargeWise AI is an <strong style="color:#fff">Agentic AI Electric Vehicle Charging Optimization System</strong>
        that demonstrates enterprise-grade AI integration using <strong style="color:var(--brand-1)">IBM watsonx.ai Granite Models</strong>.
        The system helps EV users and charging station operators analyze charging behavior, optimize schedules,
        predict energy demand, and reduce electricity costs through intelligent, context-aware AI recommendations.
        Every insight is generated in real-time by IBM Granite foundation models — not by static rules or pre-programmed logic.
      </p>
    </div>
  </div>
</div>

<!-- Four Agents -->
<div class="section-header mt-2">
  <h4 style="font-size:1rem;"><i class="bi bi-diagram-3-fill me-2" style="color:var(--brand-1)"></i>Four-Agent Architecture</h4>
</div>
<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="about-card h-100">
      <div class="d-flex gap-3 align-items-start">
        <div class="step-badge">1</div>
        <div>
          <div style="color:#00b4d8;font-weight:700;margin-bottom:.3rem;">Charging Pattern Analysis Agent</div>
          <p style="color:#8fa8cc;font-size:.85rem;line-height:1.7;margin:0;">
            Ingests historical charging station data (session duration, energy consumed, vehicle counts, time slots)
            and uses IBM Granite to produce comprehensive behavioral analysis — including peak demand identification,
            utilization rates, and trend explanations.
          </p>
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="about-card h-100">
      <div class="d-flex gap-3 align-items-start">
        <div class="step-badge" style="background:linear-gradient(135deg,#1b6b45,#52b788)">2</div>
        <div>
          <div style="color:#52b788;font-weight:700;margin-bottom:.3rem;">Charging Schedule Optimization Agent</div>
          <p style="color:#8fa8cc;font-size:.85rem;line-height:1.7;margin:0;">
            Accepts user vehicle preferences, battery capacity, priority levels, and grid constraints.
            IBM Granite generates personalized time-slot recommendations, load-balancing strategies,
            queue reduction plans, and smart pre-booking advisories.
          </p>
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="about-card h-100">
      <div class="d-flex gap-3 align-items-start">
        <div class="step-badge" style="background:linear-gradient(135deg,#b54e10,#f4a261)">3</div>
        <div>
          <div style="color:#f4a261;font-weight:700;margin-bottom:.3rem;">Energy Demand Prediction Agent</div>
          <p style="color:#8fa8cc;font-size:.85rem;line-height:1.7;margin:0;">
            Takes historical charging records, seasonal patterns, special events, and occupancy data.
            IBM Granite performs intelligent forecasting to predict future demand, identify capacity bottlenecks,
            and recommend infrastructure investments with detailed trend projections.
          </p>
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="about-card h-100">
      <div class="d-flex gap-3 align-items-start">
        <div class="step-badge" style="background:linear-gradient(135deg,#5e35b1,#9b72cf)">4</div>
        <div>
          <div style="color:#9b72cf;font-weight:700;margin-bottom:.3rem;">Cost Optimization Insights Agent</div>
          <p style="color:#8fa8cc;font-size:.85rem;line-height:1.7;margin:0;">
            Processes electricity tariffs, time-of-use pricing structures, consumption data, and renewable
            energy availability. IBM Granite produces actionable cost-saving schedules, off-peak strategies,
            estimated savings figures, and long-term energy cost roadmaps.
          </p>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Orchestrator + Tech Stack -->
<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="about-card h-100">
      <h5 class="mb-3" style="color:#fff;font-size:.95rem;">
        <i class="bi bi-share-fill me-2" style="color:var(--brand-1)"></i>Agent Orchestrator
      </h5>
      <p style="color:#8fa8cc;font-size:.85rem;line-height:1.7;">
        A central <code style="color:var(--brand-1);background:rgba(0,180,216,.1);padding:.1rem .35rem;border-radius:4px;">orchestrator()</code>
        function routes each user request to the correct specialized agent based on the selected feature.
        The routing map (pattern → schedule → demand → cost) keeps agent logic fully decoupled while
        allowing the system to scale to additional agents without modifying existing ones.
      </p>
      <div class="mt-3 p-2 rounded" style="background:#07111e;font-size:.78rem;color:#48cae4;font-family:monospace;">
        orchestrator("pattern", data) → charging_pattern_agent()<br/>
        orchestrator("schedule", data) → charging_schedule_agent()<br/>
        orchestrator("demand", data) → energy_demand_prediction_agent()<br/>
        orchestrator("cost", data) → cost_optimization_agent()
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="about-card h-100">
      <h5 class="mb-3" style="color:#fff;font-size:.95rem;">
        <i class="bi bi-cpu-fill me-2" style="color:var(--brand-1)"></i>IBM watsonx.ai Integration
      </h5>
      <ul style="color:#8fa8cc;font-size:.85rem;line-height:1.9;padding-left:1.1rem;margin:0;">
        <li><strong style="color:#fff">Model:</strong> ibm/granite-13b-instruct-v2</li>
        <li><strong style="color:#fff">Auth:</strong> IAM token exchange (API key → bearer token)</li>
        <li><strong style="color:#fff">Endpoint:</strong> watsonx.ai text generation REST API</li>
        <li><strong style="color:#fff">Config:</strong> Via environment variables (WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL)</li>
        <li><strong style="color:#fff">Function:</strong> <code style="color:var(--brand-1);background:rgba(0,180,216,.1);padding:.1rem .3rem;border-radius:4px;">generate_response(prompt)</code> — used by all 4 agents</li>
        <li><strong style="color:#fff">Parameters:</strong> greedy decoding, 900 max tokens, temp 0.7</li>
      </ul>
    </div>
  </div>
</div>

<!-- Tech stack -->
<div class="about-card mb-4">
  <h5 class="mb-3" style="color:#fff;font-size:.95rem;">
    <i class="bi bi-stack me-2" style="color:var(--brand-1)"></i>Technology Stack
  </h5>
  <div class="row g-3">
    <div class="col-6 col-md-3 text-center">
      <div style="font-size:2rem;color:#00b4d8;"><i class="bi bi-cpu-fill"></i></div>
      <div style="font-weight:700;color:#fff;font-size:.85rem;margin-top:.3rem;">IBM watsonx.ai</div>
      <div style="color:#6c8ebf;font-size:.75rem;">Granite Foundation Models</div>
    </div>
    <div class="col-6 col-md-3 text-center">
      <div style="font-size:2rem;color:#52b788;"><i class="bi bi-server"></i></div>
      <div style="font-weight:700;color:#fff;font-size:.85rem;margin-top:.3rem;">Python + Flask</div>
      <div style="color:#6c8ebf;font-size:.75rem;">Web Framework & API</div>
    </div>
    <div class="col-6 col-md-3 text-center">
      <div style="font-size:2rem;color:#f4a261;"><i class="bi bi-bootstrap-fill"></i></div>
      <div style="font-weight:700;color:#fff;font-size:.85rem;margin-top:.3rem;">Bootstrap 5</div>
      <div style="color:#6c8ebf;font-size:.75rem;">Responsive UI Framework</div>
    </div>
    <div class="col-6 col-md-3 text-center">
      <div style="font-size:2rem;color:#9b72cf;"><i class="bi bi-bar-chart-fill"></i></div>
      <div style="font-weight:700;color:#fff;font-size:.85rem;margin-top:.3rem;">Chart.js</div>
      <div style="color:#6c8ebf;font-size:.75rem;">Data Visualisation</div>
    </div>
  </div>
</div>

<!-- Use Cases -->
<div class="about-card">
  <h5 class="mb-3" style="color:#fff;font-size:.95rem;">
    <i class="bi bi-award-fill me-2" style="color:#f4a261"></i>Ideal For
  </h5>
  <div class="row g-2">
    {% for badge in ['IBM Hackathons', 'IBM SkillsBuild Showcases', 'College Demonstrations',
                     'EV & Smart Mobility Presentations', 'AI Exhibitions',
                     'Smart Grid & Sustainable Energy Showcases', 'Portfolio Projects – Agentic AI with IBM Granite',
                     'IBM TechXchange Community', 'Clean Energy Innovation Challenges'] %}
    <div class="col-auto">
      <span style="font-size:.78rem;padding:.3rem .75rem;border-radius:20px;
                   background:rgba(0,180,216,.08);border:1px solid rgba(0,180,216,.2);
                   color:#8fa8cc;">{{ badge }}</span>
    </div>
    {% endfor %}
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", "")


# ═══════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template_string(
        HOME_TEMPLATE,
        page_title="Home Dashboard",
        active_page="home",
    )

@app.route("/pattern")
def pattern():
    return render_template_string(
        PATTERN_TEMPLATE,
        page_title="Charging Pattern Analysis",
        active_page="pattern",
    )

@app.route("/schedule")
def schedule():
    return render_template_string(
        SCHEDULE_TEMPLATE,
        page_title="Schedule Optimizer",
        active_page="schedule",
    )

@app.route("/demand")
def demand():
    return render_template_string(
        DEMAND_TEMPLATE,
        page_title="Demand Prediction",
        active_page="demand",
    )

@app.route("/cost")
def cost():
    return render_template_string(
        COST_TEMPLATE,
        page_title="Cost Optimization",
        active_page="cost",
    )

@app.route("/about")
def about():
    return render_template_string(
        ABOUT_TEMPLATE,
        page_title="About ChargeWise AI",
        active_page="about",
    )


# ─────────────────────────────────────────────
# API ENDPOINT – All agent calls go through here
# POST /api/agent  { "agent": "pattern|schedule|demand|cost", "data": {...} }
# ─────────────────────────────────────────────
@app.route("/api/agent", methods=["POST"])
def api_agent():
    payload = request.get_json(force=True, silent=True) or {}
    agent_name = payload.get("agent", "")
    data       = payload.get("data", {})

    if not agent_name:
        return jsonify({"error": "Missing 'agent' field"}), 400

    # ── Route to the orchestrator (IBM watsonx.ai is invoked inside) ──
    result = orchestrator(agent_name, data)
    return jsonify({"result": result})


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ChargeWise AI – EV Charging Optimization System")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print(f"  WATSONX_API_KEY    : {'✓ set' if WATSONX_API_KEY    else '✗ NOT SET'}")
    print(f"  WATSONX_PROJECT_ID : {'✓ set' if WATSONX_PROJECT_ID else '✗ NOT SET'}")
    print(f"  WATSONX_URL        : {WATSONX_URL}")
    print("=" * 60)
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
