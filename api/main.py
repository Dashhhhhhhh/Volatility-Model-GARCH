"""FastAPI application for BTC GARCH model API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
import pandas as pd
import numpy as np
from pydantic import BaseModel
from datetime import date
import os

# Assuming the GARCH model and data ingestion scripts are in the parent directory
# Adjust the path as necessary if your project structure is different
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modeling.garch import BTCGarchModel # Assuming BTCGarchModel can be imported
import duckdb

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./data/mlruns.db")
DUCKDB_PATH = os.getenv("DATABASE_URL", "duckdb://./data/btc.duckdb").replace("duckdb://", "")

# Global variable to cache the loaded model
_cached_model = None
_model_load_time = None


app = FastAPI(
    title="BTC GARCH API",
    description="API for Bitcoin volatility predictions using GARCH models",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BacktestRequest(BaseModel):
    start_date: date
    end_date: date

def load_latest_model_from_mlflow():
    """Loads the latest GARCH model from MLflow artifacts with caching."""
    global _cached_model, _model_load_time
    
    # Simple cache: if model was loaded less than 1 hour ago, reuse it
    import time
    current_time = time.time()
    if _cached_model is not None and _model_load_time is not None:
        if current_time - _model_load_time < 3600:  # 1 hour
            return _cached_model
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Get the latest run from the Default experiment
        experiment = client.get_experiment_by_name("Default")
        if experiment is None:
            raise HTTPException(status_code=503, detail="No MLflow experiment found.")
        
        # Get all runs from the experiment, sorted by start time (latest first)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise HTTPException(status_code=503, detail="No runs found in MLflow experiment.")
        
        latest_run = runs[0]
        
        # Find the model artifact path - it should be under models/garch/{date}/*.pkl
        artifacts = client.list_artifacts(latest_run.info.run_id, "models")
        model_artifact_path = None
        
        for artifact in artifacts:
            if artifact.path.startswith("models/garch"):
                # Get nested artifacts in the garch directory
                garch_artifacts = client.list_artifacts(latest_run.info.run_id, artifact.path)
                for garch_artifact in garch_artifacts:
                    if garch_artifact.is_dir:
                        # This should be the date directory
                        date_artifacts = client.list_artifacts(latest_run.info.run_id, garch_artifact.path)
                        for date_artifact in date_artifacts:
                            if date_artifact.path.endswith(".pkl"):
                                model_artifact_path = date_artifact.path
                                break
                    if model_artifact_path:
                        break
            if model_artifact_path:
                break
        
        if not model_artifact_path:
            raise HTTPException(status_code=503, detail="No model pickle file found in MLflow artifacts.")
        
        # Download and load the model
        model_uri = f"runs:/{latest_run.info.run_id}/{model_artifact_path}"
        local_path = mlflow.artifacts.download_artifacts(model_uri)
          # Load the pickled model
        import pickle
        with open(local_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Cache the model
        _cached_model = loaded_model
        _model_load_time = current_time
        
        return loaded_model

    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        raise HTTPException(status_code=503, detail=f"Model could not be loaded from MLflow: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "BTC GARCH API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/forecast")
async def get_forecast():
    """
    Returns the latest one-step-ahead volatility (sigma) forecast.
    Loads the latest GARCH model from MLflow and uses it to predict.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        # Load the fitted GARCH model from MLflow
        fitted_model = load_latest_model_from_mlflow()

        # Generate a one-step-ahead forecast
        # The arch library's forecast method generates out-of-sample forecasts
        forecast_result = fitted_model.forecast(horizon=1, reindex=False)
          # Extract the volatility forecast (sigma = sqrt(variance))
        # The forecast result has the structure: mean, variance columns
        variance_forecast = float(forecast_result.variance.iloc[-1, 0])
        sigma_forecast = float(np.sqrt(variance_forecast))

        return {"sigma": sigma_forecast}
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions from model loading
    except Exception as e:
        print(f"Error during forecast generation: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Performs a simplified backtest for VaR breaches.
    This endpoint loads a GARCH model and returns simulated VaR breach counts.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        # Load the GARCH model
        fitted_model = load_latest_model_from_mlflow()

        # Fetch historical data for the requested period from DuckDB
        con = duckdb.connect(DUCKDB_PATH, read_only=True)
        query = """
        SELECT date, close 
        FROM btc_prices 
        WHERE date >= ? AND date <= ? AND interval = '1d'
        ORDER BY date
        """
        prices_df = con.execute(query, [request.start_date, request.end_date]).fetchdf()
        con.close()
        
        if prices_df.empty:
            raise HTTPException(status_code=404, detail="No price data found for the specified period.")
          # Calculate log returns for the period
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
        log_returns = (np.log(prices_df['close']) - np.log(prices_df['close'].shift(1))) * 100
        actual_returns = log_returns.dropna()
        
        if len(actual_returns) < 2:
            raise HTTPException(status_code=400, detail="Insufficient data for backtesting.")

        # Simplified VaR calculation using the model's conditional volatility
        # This is a demonstration - in practice, you'd want a more sophisticated approach
        conditional_vol = fitted_model.conditional_volatility
        
        # Use the mean conditional volatility as a proxy for VaR calculation
        # In practice, you'd use a confidence level (e.g., 99%) and the distribution
        mean_volatility = conditional_vol.mean()
        var_threshold = -1.96 * mean_volatility  # 95% VaR (simplified)
        
        # Count breaches (when actual returns are worse than VaR)
        breaches = (actual_returns < var_threshold).sum()
        
        return {
            "start_date": request.start_date,
            "end_date": request.end_date,
            "var_breaches_count": int(breaches),
            "total_observations": len(actual_returns),
            "breach_rate": float(breaches / len(actual_returns)) if len(actual_returns) > 0 else 0.0
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
