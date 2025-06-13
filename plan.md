Below is a **“prompt-driven build plan”** you can hand to any competent coding-LLM (GPT-4o, DeepSeek-Coder, etc.). It’s organized in *sprints*; each sprint ends with tests the model must pass before moving on. Copy-paste the bold bits (or tweak them) as system-level or user-level prompts when you chat with the code model.

---

## 🏁 Sprint 0 — Repo Scaffold

**System Prompt:**

> *You are an expert Python architect. Create a Poetry-managed repo called `btc_garch`, ready for Docker and CI.*

1. **Generate repo tree**

   ```
   btc_garch/
   ├── btc_garch/
   │   ├── __init__.py
   │   ├── data/
   │   ├── modeling/
   │   ├── api/
   │   ├── dashboard/
   │   └── utils/
   ├── tests/
   ├── pyproject.toml
   ├── .env.example
   ├── .dockerignore
   └── Dockerfile
   ```
2. **Poetry `pyproject.toml`** with these deps:

   ```
   arch, pandas, numpy, yfinance, matplotlib, plotly,
   prefect, fastapi, uvicorn[standard], duckdb, pydantic,
   mlflow, optuna
   ```
3. **Dockerfile** using `python:3.10-slim`, running `poetry install`.

**Acceptance Test:**
`pytest -q` should show 0 tests collected, exit 0.
`docker build .` must succeed.

---

## 🏁 Sprint 1 — Data Layer

**User Prompt:**

> *Write `data/ingest.py` that*
>
> 1. *downloads BTC-USD candles (daily & 1-h) via yfinance*
> 2. *stores raw CSV in `/data/raw/`*
> 3. *writes Parquet to DuckDB `/data/btc.duckdb`*
>    *Add pydantic models for typed I/O.*

**Tests:**

* `python -m btc_garch.data.ingest --start 2023-01-01` creates both files.
* DuckDB table `btc_prices` has non-null `close` and spans requested dates.

---

## 🏁 Sprint 2 — Volatility Modeling Core

**User Prompt:**

> *Implement `modeling/garch.py` with a class `BTCGarchModel` that:*
> • *fetches returns from DuckDB*
> • *fits GARCH(1,1) (Student-t)* using `arch`\*
> • *serializes the fitted model via pickle + MLflow (artifact path `models/garch/YYYYMMDD`)*
> • *outputs a Pandas DataFrame of one-step-ahead σ<sub>t+1</sub> forecasts.*

**Tests:**

* `BTCGarchModel.fit().summary()` contains “Student’s t”.
* Forecast DataFrame length == input length – burn-in.

---

## 🏁 Sprint 3 — Hyper-param Search

**User Prompt:**

> *Create `modeling/search.py` that runs Optuna to scan (p, q) ∈ \[1-3]×\[1-3] and distribution type {normal, t}.*
> • *Objective = AIC.*
> • *Log trials to MLflow experiment “garch\_search”.*
> *Return best params dict.*

**Tests:**

* Running search produces ≥9 trials.
* `mlflow ui` lists experiment with best AIC field.

---

## 🏁 Sprint 4 — REST API

**User Prompt:**

> *Build FastAPI app `api/main.py` exposing:*
>
> * `GET /forecast` → latest σ forecast JSON
> * `POST /backtest` with dates → JSON of VaR breaches count
>   *Mount MLflow model for inference inside the endpoint.*

**Tests:**

* `uvicorn btc_garch.api.main:app` runs.
* Curl to `/forecast` returns `{"sigma": float}` within 100 ms.

---

## 🏁 Sprint 5 — Streamlit Dashboard

**User Prompt:**

> *Create Streamlit app `dashboard/app.py` that:*
>
> 1. *Plots daily BTC price & conditional σ on twin axes.*
> 2. *Displays a VaR traffic-light widget (green / red for last 30 days).*
> 3. *Calls `/forecast` endpoint every minute.*

**Tests:**

* `streamlit run dashboard/app.py` shows chart and updates without crash.

---

## 🏁 Sprint 6 — Orchestration & CI/CD

**User Prompt:**

> *Write `prefect_flow.py` that:*
> • *ingests data, retrains model weekly, pushes artifact to S3 (mock).*
> • *Deploys flow via Prefect Cloud (use env vars for token).*
> *Add GitHub Actions workflow:*
>
> * *`push` → lint (`ruff .`), pytest, Docker build, image push to GHCR*
> * *on `main` success → `flyctl deploy`*

**Tests:**

* `prefect run` completes all tasks locally.
* `gh workflow run ci.yml --dry-run` exits 0.

---

## 🏁 Sprint 7 — Docs & Polish

**User Prompt:**

> *Generate Quarto docs under `docs/` covering:*
>
> * *Project overview*
> * *How GARCH works*
> * *API usage examples (curl & Python)*
>   *Set up `quarto publish gh-pages` in CI.*

---

### Tips for Working with the Coding LLM

1. **Chunk prompts**—one module at a time.
2. **Pin context**—paste the latest file tree or key code when asking for edits.
3. **Enforce tests early**—the LLM **loves** green check-marks.
4. **Use `# TODO:` comments**—then prompt: “Fill in every TODO, keep other code unchanged.”
5. **Regenerate selectively**—if only one function fails, show the failing test and ask for a patch, not a rewrite.

Follow these sprints and prompt templates, and your code LLM will crank out a production-ready Bitcoin-GARCH stack while you sip coffee and watch the volatility spikes. Happy prompting!
