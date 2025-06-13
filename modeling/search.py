import optuna
import mlflow
from arch import arch_model
import pandas as pd
import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT_NAME = "garch_search" # As per plan.md for Sprint 3
DUCKDB_PATH = os.getenv("DATABASE_URL", "duckdb:///./data/btc_garch.db").replace("duckdb://", "")


def get_returns_from_db(db_path: str) -> pd.Series:
    """Fetches daily log returns from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    # Assuming 'btc_prices' table with 'close' and 'date' columns
    # And that data is already sorted by date
    prices = con.execute("SELECT date, close FROM btc_prices ORDER BY date").fetchdf()
    con.close()
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.set_index('date')
    log_returns = (pd.np.log(prices['close']) - pd.np.log(prices['close'].shift(1))) * 100
    return log_returns.dropna()

def objective(trial: optuna.Trial, returns: pd.Series) -> float:
    """Optuna objective function to minimize AIC for GARCH model."""
    p = trial.suggest_int("p", 1, 3)
    q = trial.suggest_int("q", 1, 3)
    dist = trial.suggest_categorical("dist", ["normal", "t"])

    mlflow.start_run(nested=True)
    mlflow.log_params({"p": p, "q": q, "dist": dist})

    try:
        model = arch_model(returns, vol="Garch", p=p, o=0, q=q, dist=dist)
        results = model.fit(disp="off")
        aic = results.aic
        mlflow.log_metric("aic", aic)
    except Exception as e:
        print(f"Error fitting model with p={p}, q={q}, dist={dist}: {e}")
        mlflow.log_metric("aic", float('inf')) # Penalize failed trials
        mlflow.end_run(status="FAILED")
        # Optuna will handle this by trying other parameters or raising an error if too many trials fail.
        # For robustness, we might return a very high AIC or re-raise depending on desired behavior.
        # Returning a high AIC allows Optuna to continue searching.
        return float('inf')


    mlflow.end_run()
    return aic

def run_hyperparameter_search(returns: pd.Series, n_trials: int = 100) -> dict:
    """Runs Optuna hyperparameter search and logs to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, returns), n_trials=n_trials)

    best_params = study.best_params
    best_aic = study.best_value

    print(f"Best AIC: {best_aic}")
    print(f"Best params: {best_params}")

    # Log best trial to MLflow under the main run (optional, could be a separate run)
    with mlflow.start_run(run_name="best_garch_model_search_results") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("best_aic", best_aic)
        # Log the study object or a summary if desired
        # optuna.visualization.plot_optimization_history(study).write_image("optuna_history.png")
        # mlflow.log_artifact("optuna_history.png")

    return best_params

if __name__ == "__main__":
    print("Fetching returns...")
    # Ensure DUCKDB_PATH is correct and the database/table exists
    # This part assumes that Sprint 1 (data ingestion) has been run
    # and the btc_garch.db contains the btc_prices table.
    try:
        returns_data = get_returns_from_db(DUCKDB_PATH)
        if returns_data.empty:
            print(f"No data returned from {DUCKDB_PATH}. Ensure 'btc_prices' table exists and has data.")
        else:
            print(f"Successfully fetched {len(returns_data)} return entries.")
            print("Starting hyperparameter search...")
            best_parameters = run_hyperparameter_search(returns_data, n_trials=int(os.getenv("OPTIMIZATION_TRIALS", 10))) # Reduced for quick testing, plan asks for 100
            print("Hyperparameter search finished.")
            print(f"Best parameters found: {best_parameters}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Please ensure your DuckDB database is correctly set up at '{DUCKDB_PATH}' and contains the 'btc_prices' table.")
        print("You might need to run the data ingestion script from Sprint 1.")

