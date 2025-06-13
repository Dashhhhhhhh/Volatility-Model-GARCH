import pytest
import pandas as pd
import numpy as np
from modeling.garch import BTCGarchModel
import duckdb
import os
import shutil
import mlflow

# Test data setup
TEST_DB_PATH = 'test_btc_garch.duckdb'
TEST_TABLE_NAME = 'test_prices'
N_SAMPLES = 100

@pytest.fixture(scope="module")
def setup_test_db():
    """Sets up a temporary DuckDB database with sample price data for testing."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    con = duckdb.connect(TEST_DB_PATH)
    # Create a simple date range
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='D'))
    # Simulate some price data (e.g., a random walk)
    prices = 100 + np.cumsum(np.random.randn(N_SAMPLES) * 0.5) 
    prices_df = pd.DataFrame({'date': dates, 'close': prices})
    
    con.execute(f"CREATE TABLE {TEST_TABLE_NAME} AS SELECT * FROM prices_df")
    con.close()
    
    yield TEST_DB_PATH # Provide the path to the test DB
    
    # Teardown: Remove the test database
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

@pytest.fixture
def btc_model(setup_test_db, tmp_path):
    """Provides an instance of BTCGarchModel initialized with the test DB."""
    # Use tmp_path for MLflow to avoid directory conflicts
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    
    # Set MLflow tracking URI and ensure clean state
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    
    # End any existing MLflow runs
    try:
        mlflow.end_run()
    except:
        pass
    
    # Create and set experiment
    try:
        experiment_id = mlflow.create_experiment("test_experiment")
        mlflow.set_experiment("test_experiment")
    except mlflow.exceptions.MlflowException:
        # Experiment already exists
        mlflow.set_experiment("test_experiment")
    
    model = BTCGarchModel(db_path=setup_test_db, table_name=TEST_TABLE_NAME)
    yield model
    
    # Clean up MLflow runs
    try:
        mlflow.end_run()
    except:
        pass

def test_btc_garch_model_fit_summary_contains_students_t(btc_model):
    """Tests that the fitted model summary contains 'Student\'s t'."""
    fitted_model = btc_model.fit()
    summary_text = fitted_model.summary().as_text()
    assert "Student's t" in summary_text, "Model summary should indicate Student's t distribution."

def test_btc_garch_model_forecast_length(btc_model):
    """Tests that the forecast DataFrame length is input length - burn_in."""
    burn_in_value = 5
    
    # Fit the model to get the returns_series length
    btc_model.fit()
    assert btc_model.returns_series is not None, "Returns series should be populated after fit."
    input_length = len(btc_model.returns_series)
    
    forecast_df = btc_model.forecast(burn_in=burn_in_value)
    
    expected_length = input_length - burn_in_value
    assert len(forecast_df) == expected_length, \
        f"Forecast DataFrame length ({len(forecast_df)}) should be input length ({input_length}) - burn_in ({burn_in_value}) = {expected_length}."

def test_btc_garch_model_forecast_output_type(btc_model):
    """Tests that the forecast method returns a Pandas DataFrame."""
    btc_model.fit()
    forecast_output = btc_model.forecast()
    assert isinstance(forecast_output, pd.DataFrame), "Forecast output should be a Pandas DataFrame."
    assert 'sigma_t_plus_1' in forecast_output.columns, "Forecast DataFrame should have 'sigma_t_plus_1' column."

def test_btc_garch_model_fit_with_provided_returns(btc_model):
    """Tests fitting the model with a pre-calculated returns series."""
    # Generate some sample returns data
    sample_returns = pd.Series(np.random.randn(N_SAMPLES -1) * 0.01, 
                               index=pd.date_range(start='2023-01-02', periods=N_SAMPLES-1, freq='D'))
    
    fitted_model = btc_model.fit(returns_data=sample_returns)
    assert fitted_model is not None, "Model should fit successfully with provided returns."
    summary_text = fitted_model.summary().as_text()
    assert "Student's t" in summary_text, "Model summary should indicate Student's t distribution even with provided returns."
    assert len(btc_model.returns_series) == len(sample_returns), "Model should use the provided returns series."

def test_btc_garch_model_invalid_burn_in(btc_model):
    """Tests that an invalid burn_in value raises a ValueError."""
    btc_model.fit()
    with pytest.raises(ValueError, match="burn_in must be non-negative"):
        btc_model.forecast(burn_in=-1)

def test_btc_garch_model_burn_in_larger_than_forecasts(btc_model):
    """Tests that if burn_in is >= forecast length, an empty DataFrame is returned."""
    btc_model.fit()
    # Get the length of potential forecasts (which is same as returns_series length for h=1)
    num_forecasts = len(btc_model.returns_series)
    forecast_df = btc_model.forecast(burn_in=num_forecasts) # burn_in equal to length
    assert forecast_df.empty, "Forecast DataFrame should be empty if burn_in >= number of forecasts."
    assert 'sigma_t_plus_1' in forecast_df.columns # Still expect the column

    forecast_df_larger = btc_model.forecast(burn_in=num_forecasts + 5) # burn_in larger than length
    assert forecast_df_larger.empty, "Forecast DataFrame should be empty if burn_in > number of forecasts."


@pytest.mark.parametrize("price_column_name", ["close", "custom_price_col"])
def test_btc_garch_model_different_price_column(price_column_name):
    """Tests model with different price column names."""
    test_db_path = f'test_custom_{price_column_name}.duckdb'
    
    # Clean up any existing database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # Create DB with a custom column name
    con = duckdb.connect(test_db_path)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='D'))
    prices = 100 + np.cumsum(np.random.randn(N_SAMPLES) * 0.5)
    prices_df = pd.DataFrame({'date': dates, price_column_name: prices})
    con.execute(f"CREATE TABLE {TEST_TABLE_NAME} AS SELECT * FROM prices_df")
    con.close()

    # Setup MLflow for this test
    mlflow_dir = f"test_mlruns_custom_{price_column_name}"
    if os.path.exists(mlflow_dir):
        shutil.rmtree(mlflow_dir)
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{os.path.abspath(mlflow_dir)}")
    
    # Create experiment explicitly
    try:
        mlflow.create_experiment(f"test_custom_{price_column_name}")
        mlflow.set_experiment(f"test_custom_{price_column_name}")
    except mlflow.exceptions.MlflowException:
        mlflow.set_experiment(f"test_custom_{price_column_name}")
    
    model = BTCGarchModel(db_path=test_db_path, table_name=TEST_TABLE_NAME, price_column=price_column_name)
    
    try:
        fitted_model = model.fit()
        assert fitted_model is not None
        assert "Student's t" in fitted_model.summary().as_text()
        forecast_df = model.forecast(burn_in=1)
        assert not forecast_df.empty
    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        if os.path.exists(mlflow_dir):
            shutil.rmtree(mlflow_dir)

def test_data_fetching_errors(tmp_path):
    """Tests error handling for data fetching issues."""
    empty_db_path = tmp_path / "empty.duckdb"
    
    # Test with non-existent database file
    model_no_db = BTCGarchModel(db_path=str(empty_db_path), table_name="any_table")
    with pytest.raises(ValueError, match="Database file .* does not exist"):
        model_no_db._fetch_and_prepare_data()
    
    # Create empty database for further tests
    conn = duckdb.connect(str(empty_db_path))
    conn.close()
    
    # Test with non-existent table
    model_no_table = BTCGarchModel(db_path=str(empty_db_path), table_name="non_existent_table")
    with pytest.raises(duckdb.CatalogException): # DuckDB raises CatalogException for missing table
        model_no_table._fetch_and_prepare_data() # Accessing protected member for specific test

    # Test with table missing required columns
    conn = duckdb.connect(str(empty_db_path))
    conn.execute("CREATE TABLE wrong_cols (id INTEGER, value REAL)")
    conn.close()
    model_wrong_cols = BTCGarchModel(db_path=str(empty_db_path), table_name="wrong_cols", price_column='price')
    with pytest.raises(ValueError, match="Expected 'date' and 'price' columns"):
        model_wrong_cols._fetch_and_prepare_data()

    # Test with insufficient data for returns
    conn = duckdb.connect(str(empty_db_path))
    conn.execute("DROP TABLE IF EXISTS wrong_cols")
    # Create DataFrame and insert it properly
    one_row_df = pd.DataFrame({'date': [pd.to_datetime('2023-01-01')], 'close': [100]})
    conn.register('one_row_temp', one_row_df)
    conn.execute("CREATE TABLE one_row_table AS SELECT * FROM one_row_temp")
    conn.close()
    model_one_row = BTCGarchModel(db_path=str(empty_db_path), table_name="one_row_table")
    with pytest.raises(ValueError, match="Not enough valid price points to calculate returns"):
        model_one_row._fetch_and_prepare_data()

    # Test with data that results in empty returns (all NaNs)
    conn = duckdb.connect(str(empty_db_path))
    conn.execute("DROP TABLE IF EXISTS one_row_table")
    all_nan_prices_df = pd.DataFrame({
        'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=3, freq='D')),
        'close': [np.nan, np.nan, np.nan]
    })
    conn.register('nan_temp', all_nan_prices_df)
    conn.execute("CREATE TABLE nan_table AS SELECT * FROM nan_temp")
    conn.close()
    model_nan_returns = BTCGarchModel(db_path=str(empty_db_path), table_name="nan_table")
    with pytest.raises(ValueError, match="Not enough valid price points to calculate returns"):
        model_nan_returns._fetch_and_prepare_data()


def test_fit_error_insufficient_returns(btc_model):
    """Tests ValueError if returns series is too short for GARCH fitting."""
    short_returns = pd.Series(np.random.randn(5) * 0.01) # Only 5 data points
    with pytest.raises(ValueError, match="Returns series too short .* for GARCH model fitting"):
        btc_model.fit(returns_data=short_returns)

