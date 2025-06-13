import pandas as pd
import numpy as np
import duckdb
from arch import arch_model
import mlflow
import pickle
import os
from datetime import datetime


class BTCGarchModel:
    """
    A class to fetch Bitcoin price data, fit a GARCH(1,1) model with Student-t
    distribution, serialize the model, and generate volatility forecasts.
    """
    def __init__(self, db_path='data/btc.duckdb', table_name='btc_prices_daily', price_column='close'):
        """
        Initializes the BTCGarchModel.

        Args:
            db_path (str): Path to the DuckDB database file.
            table_name (str): Name of the table containing price data.
            price_column (str): Name of the column with closing prices.
        """        
        self.db_path = db_path
        self.table_name = table_name
        self.price_column = price_column
        self.model_spec = None  # Stores the arch_model object before fitting
        self.fitted_model_result = None  # Stores the result from model.fit()
        self.returns_series = None  # Stores the returns series used for fitting    def _fetch_and_prepare_data(self):
        """
        Fetches prices from DuckDB, calculates scaled log returns, and stores them.
        Raises ValueError if data fetching or processing fails.
        """
        # Check if database file exists
        if not os.path.exists(self.db_path):
            raise ValueError(f"Database file '{self.db_path}' does not exist.")
            
        con = None
        try:
            con = duckdb.connect(database=self.db_path, read_only=True)
            # Ensure column name is quoted if it contains special chars or is a keyword
            query = f'SELECT date, "{self.price_column}" FROM {self.table_name} ORDER BY date ASC'
            prices_df = con.execute(query).fetchdf()
        finally:
            if con:
                con.close()

        if prices_df.empty:
            raise ValueError(f"No data fetched from DuckDB table '{self.table_name}'. Check table name and data presence.")
        
        if 'date' not in prices_df.columns or self.price_column not in prices_df.columns:
            raise ValueError(f"Expected 'date' and '{self.price_column}' columns in fetched data from table '{self.table_name}'.")

        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
        
        # Calculate log returns, scaled by 100 for GARCH modeling
        # Ensure price_column is float for calculation
        valid_prices = prices_df[self.price_column].astype(float).dropna()
        if len(valid_prices) < 2:
            raise ValueError("Not enough valid price points to calculate returns (need at least 2).")

        log_returns = 100 * np.log(valid_prices / valid_prices.shift(1))
        self.returns_series = log_returns.dropna()
        
        if self.returns_series.empty:
            raise ValueError("Returns series is empty after calculation. Check for NaNs or insufficient data in price series.")
        
        if np.isinf(self.returns_series).any() or np.isnan(self.returns_series).any():
            raise ValueError("Infinite or NaN values found in returns series. Check price data for zeros, negative values, or gaps.")

    def fit(self, returns_data=None):
        """
        Fits a GARCH(1,1) model with Student-t distribution to the returns.
        Serializes the fitted model using pickle and logs it with MLflow.

        Args:
            returns_data (pd.Series, optional): Pre-calculated returns series. 
                                                If None, data will be fetched from DuckDB.

        Returns:
            arch.univariate.base.ARCHModelResult: The fitted model result object.
        
        Raises:
            TypeError: If returns_data is not a pandas Series.
            ValueError: If no returns data is available or suitable for fitting.
        """
        if returns_data is not None:
            if not isinstance(returns_data, pd.Series):
                raise TypeError("returns_data must be a pandas Series.")
            self.returns_series = returns_data.dropna() # Ensure no NaNs in provided returns
        elif self.returns_series is None:
            self._fetch_and_prepare_data()

        if self.returns_series is None or self.returns_series.empty:
            raise ValueError("No returns data available to fit the model.")
        if len(self.returns_series) < 10: # GARCH models need sufficient data
             raise ValueError(f"Returns series too short ({len(self.returns_series)} points) for GARCH model fitting. Need at least 10 points.")


        # Define the GARCH(1,1) model with Student-t distribution
        self.model_spec = arch_model(self.returns_series, vol='Garch', p=1, q=1, dist='StudentsT')
        
        # Fit the model
        self.fitted_model_result = self.model_spec.fit(disp='off', show_warning=False)        # Serialize model with MLflow
        today_str = datetime.now().strftime("%Y%m%d")
        mlflow_artifact_subdir = f"models/garch/{today_str}"
        
        # Ensure MLflow experiment exists
        try:
            mlflow.get_experiment_by_name("Default")
        except:
            mlflow.create_experiment("Default")
        
        with mlflow.start_run():
            mlflow.log_param("model_type", "GARCH(1,1)")
            mlflow.log_param("distribution", "StudentsT")
            mlflow.log_param("p_lag", 1)
            mlflow.log_param("q_lag", 1)
            mlflow.log_param("input_returns_length", len(self.returns_series))
            
            # Get summary text
            summary_str = self.fitted_model_result.summary().as_text()
            
            # Create temporary files for artifacts
            summary_filename = "model_summary.txt"
            model_pickle_filename = "btc_garch_1_1_student_t.pkl"
            
            # Save summary as text file
            with open(summary_filename, "w") as f:
                f.write(summary_str)
            mlflow.log_artifact(summary_filename, artifact_path=mlflow_artifact_subdir)

            # Save and log the model
            with open(model_pickle_filename, "wb") as f:
                pickle.dump(self.fitted_model_result, f)
            mlflow.log_artifact(model_pickle_filename, artifact_path=mlflow_artifact_subdir)
            
            # Log metrics
            mlflow.log_metric("aic", self.fitted_model_result.aic)
            mlflow.log_metric("bic", self.fitted_model_result.bic)
            mlflow.log_metric("log_likelihood", self.fitted_model_result.loglikelihood)
            
            # Clean up temporary files
            if os.path.exists(summary_filename):
                os.remove(summary_filename)
            if os.path.exists(model_pickle_filename):
                os.remove(model_pickle_filename)

        return self.fitted_model_result

    def forecast(self, burn_in=1):
        """
        Outputs a Pandas DataFrame of one-step-ahead sigma_t+1 forecasts.
        The length of the forecast DataFrame is input length – burn_in.
        'input length' refers to the length of the returns series used for fitting.

        Args:
            burn_in (int): Number of initial forecasts to discard. Default is 1.

        Returns:
            pd.DataFrame: DataFrame with a single column 'sigma_t_plus_1' 
                          containing the one-step-ahead volatility forecasts.
        
        Raises:
            ValueError: If the model has not been fitted, or if burn_in is invalid.
        """
        if self.fitted_model_result is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if self.returns_series is None or self.returns_series.empty:
            raise ValueError("Returns series is not available. Ensure model was fitted with data.")

        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        
        # Get conditional volatility from the fitted model (in-sample estimates)
        # This gives us the estimated volatility for each time period in the sample
        conditional_volatility = self.fitted_model_result.conditional_volatility
        
        if burn_in >= len(conditional_volatility):
            return pd.DataFrame(columns=['sigma_t_plus_1']) # Return empty if burn_in too large

        # Apply burn-in period
        final_sigma_forecasts = conditional_volatility.iloc[burn_in:]
        
        # Create DataFrame with proper column name
        forecast_df = pd.DataFrame(final_sigma_forecasts)
        forecast_df.columns = ['sigma_t_plus_1']
        
        return forecast_df

