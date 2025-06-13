import pytest
import pandas as pd
import numpy as np
import mlflow
import os
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add project root to sys.path to allow for correct module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from modeling.search import run_hyperparameter_search, MLFLOW_EXPERIMENT_NAME, get_returns_from_db

# Use a temporary directory for MLflow runs during testing
TEST_MLFLOW_DIR = "./test_mlruns"
TEST_DB_PATH = "./test_data/test_btc_garch.db" # Not actually used if get_returns_from_db is mocked

@pytest.fixture(scope="module")
def setup_teardown_mlflow():
    """Setup a temporary MLflow tracking URI and clean up after tests."""
    original_tracking_uri = mlflow.get_tracking_uri()
    test_tracking_uri = f"sqlite:///{TEST_MLFLOW_DIR}/mlruns.db"
    mlflow.set_tracking_uri(test_tracking_uri)

    if not os.path.exists(TEST_MLFLOW_DIR):
        os.makedirs(TEST_MLFLOW_DIR)

    yield test_tracking_uri # Provide the URI to tests if needed

    # Teardown: Remove the temporary MLflow directory and reset tracking URI
    shutil.rmtree(TEST_MLFLOW_DIR, ignore_errors=True)
    mlflow.set_tracking_uri(original_tracking_uri)

@pytest.fixture
def sample_returns_data() -> pd.Series:
    """Provides a sample Pandas Series of returns for testing."""
    np.random.seed(42)
    return pd.Series(np.random.randn(200) * 0.01 + 0.0005, name="log_returns")

@patch("modeling.search.get_returns_from_db")
@patch.dict(os.environ, {"MLFLOW_TRACKING_URI": f"sqlite:///{TEST_MLFLOW_DIR}/mlruns.db", "OPTIMIZATION_TRIALS": "3"})
def test_run_hyperparameter_search_completes_trials_and_logs_to_mlflow(
    mock_get_returns: MagicMock,
    sample_returns_data: pd.Series,
    setup_teardown_mlflow # Ensure MLflow is set up for this test
):
    """Test that hyperparameter search runs, completes trials, and logs to MLflow."""
    mock_get_returns.return_value = sample_returns_data
    n_test_trials = 3 # Should match OPTIMIZATION_TRIALS from patch.dict    # Override MLFLOW_EXPERIMENT_NAME for testing to avoid conflicts if run in parallel or repeatedly
    test_experiment_name = f"{MLFLOW_EXPERIMENT_NAME}_test_{np.random.randint(10000)}"
    
    with patch("modeling.search.MLFLOW_EXPERIMENT_NAME", test_experiment_name):
        best_params = run_hyperparameter_search(returns=sample_returns_data, n_trials=n_test_trials)

    assert isinstance(best_params, dict)
    assert "p" in best_params
    assert "q" in best_params
    assert "dist" in best_params

    # Verify MLflow experiment and runs
    experiment = mlflow.get_experiment_by_name(test_experiment_name)
    assert experiment is not None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    # n_test_trials for individual hyperparameter evaluations + 1 for the summary "best_garch_model_search_results" run
    assert len(runs) >= n_test_trials + 1 

    # Check that individual trials have an AIC metric
    trial_runs = runs[runs["tags.mlflow.runName"] != "best_garch_model_search_results"] 
    assert len(trial_runs) >= n_test_trials
    for _, row in trial_runs.iterrows():
        assert "metrics.aic" in row
        assert pd.notna(row["metrics.aic"])

    # Check that the best model run has the best_aic metric
    best_model_run = runs[runs["tags.mlflow.runName"] == "best_garch_model_search_results"]
    assert len(best_model_run) == 1
    assert "metrics.best_aic" in best_model_run.columns
    assert pd.notna(best_model_run.iloc[0]["metrics.best_aic"])
    # Check that best params are logged
    assert f"params.p" in best_model_run.columns
    assert f"params.q" in best_model_run.columns
    assert f"params.dist" in best_model_run.columns

@patch.dict(os.environ, {"MLFLOW_TRACKING_URI": f"sqlite:///{TEST_MLFLOW_DIR}/mlruns.db", "OPTIMIZATION_TRIALS": "1"})
@patch("modeling.search.arch_model") # Mock arch_model to simulate fitting errors
@patch("modeling.search.get_returns_from_db")
def test_hyperparameter_search_handles_model_fit_errors(
    mock_get_returns: MagicMock,
    mock_arch_model: MagicMock,
    sample_returns_data: pd.Series,
    setup_teardown_mlflow
):
    """Test that Optuna search handles errors during model fitting gracefully."""
    mock_get_returns.return_value = sample_returns_data
      # Configure the mock arch_model().fit() to raise an exception
    mock_model_instance = MagicMock()
    mock_model_instance.fit.side_effect = Exception("Simulated fitting error")
    mock_arch_model.return_value = mock_model_instance
    
    n_test_trials = 1 # Run only one trial that is expected to fail
    test_experiment_name = f"{MLFLOW_EXPERIMENT_NAME}_error_test_{np.random.randint(10000)}"

    with patch("modeling.search.MLFLOW_EXPERIMENT_NAME", test_experiment_name):
        # Expect the study to still complete, with the trial(s) marked as failed or having high AIC
        best_params = run_hyperparameter_search(returns=sample_returns_data, n_trials=n_test_trials)

    # Even if trials fail, run_hyperparameter_search should return the best found so far
    # (Optuna handles this; if all fail, it might raise or return default/initial params)
    # For this test, we are more interested in the MLflow logging of failures.
    assert isinstance(best_params, dict) 

    experiment = mlflow.get_experiment_by_name(test_experiment_name)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # 1 trial run + 1 best_garch_model_search_results run
    assert len(runs) >= n_test_trials + 1

    trial_runs = runs[runs["tags.mlflow.runName"] != "best_garch_model_search_results"]
    assert len(trial_runs) >= n_test_trials
    # Check that the failed trial has a very high AIC and status FAILED
    failed_trial_run = trial_runs.iloc[0]
    assert failed_trial_run["metrics.aic"] == float('inf')
    assert failed_trial_run["status"] == "FAILED"

    # The "best_garch_model_search_results" run should still exist and log the (poor) best result
    best_model_run = runs[runs["tags.mlflow.runName"] == "best_garch_model_search_results"]
    assert len(best_model_run) == 1
    assert pd.notna(best_model_run.iloc[0]["metrics.best_aic"])

