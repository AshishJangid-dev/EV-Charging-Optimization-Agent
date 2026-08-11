# EV-Charging-Optimization-Agent


app.py
├── IBM watsonx.ai Layer
│   ├── get_iam_token()          – exchanges API key for bearer token
│   └── generate_response(prompt) – calls ibm/granite-13b-instruct-v2
│
├── Four Specialized Agents
│   ├── charging_pattern_agent()          – Agent 1
│   ├── charging_schedule_agent()         – Agent 2
│   ├── energy_demand_prediction_agent()  – Agent 3
│   └── cost_optimization_agent()         – Agent 4
│
├── orchestrator(agent_name, data)  – routes to correct agent
│
├── HTML Templates (render_template_string)
│   ├── BASE_TEMPLATE    – sidebar, topbar, CSS, Bootstrap 5
│   ├── HOME_TEMPLATE    – dashboard + Chart.js charts
│   ├── PATTERN_TEMPLATE – Agent 1 form + result
│   ├── SCHEDULE_TEMPLATE – Agent 2 form + result
│   ├── DEMAND_TEMPLATE  – Agent 3 form + result
│   ├── COST_TEMPLATE    – Agent 4 form + result
│   └── ABOUT_TEMPLATE   – architecture documentation
│
└── Flask Routes
    ├── GET  /           → Home Dashboard
    ├── GET  /pattern    → Charging Pattern Analysis
    ├── GET  /schedule   → Schedule Optimizer
    ├── GET  /demand     → Demand Prediction
    ├── GET  /cost       → Cost Optimization
    ├── GET  /about      → About page
    └── POST /api/agent  → Agent API (JSON in/out)
