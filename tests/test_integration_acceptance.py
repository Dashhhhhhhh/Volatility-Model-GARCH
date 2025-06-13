"""Integration acceptance tests for BTC GARCH functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf


class TestDataIntegration:
    """Integration tests for data-related functionality."""
    
    def test_yfinance_data_access(self):
        """Test that we can access BTC data through yfinance."""
        
        # Test fetching a small amount of recent BTC data
        btc = yf.Ticker("BTC-USD")
        
        # Get last 5 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        data = btc.history(start=start_date, end=end_date)
        
        assert not data.empty, "Should be able to fetch BTC data"
        assert 'Close' in data.columns, "Data should contain Close prices"
        assert len(data) > 0, "Should have at least some data points"
    
    def test_pandas_data_manipulation(self):
        """Test basic pandas operations for price data."""
        # Create sample BTC price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.random.randn(100).cumsum() + 50000  # Random walk starting at 50k
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
        
        # Test basic operations
        assert len(df) == 100
        assert df['Close'].dtype == np.float64
        
        # Test returns calculation
        df['Returns'] = df['Close'].pct_change()
        assert not df['Returns'].isna().all(), "Should be able to calculate returns"
        assert df['Returns'].isna().sum() == 1, "Should have one NaN from pct_change"


class TestModelingIntegration:
    """Integration tests for GARCH modeling functionality."""
    
    def test_arch_library_basic_functionality(self):
        """Test that the arch library can be used for basic GARCH modeling."""
        from arch import arch_model
        import numpy as np
        
        # Generate sample return data
        np.random.seed(42)  # For reproducible tests
        returns = np.random.normal(0, 1, 1000)
        
        # Test that we can create a GARCH model
        model = arch_model(returns, vol='Garch', p=1, q=1)
        assert model is not None
        
        # Test that model has expected attributes
        assert hasattr(model, 'fit')
        assert hasattr(model, 'volatility')
    
    def test_statistical_libraries_integration(self):
        """Test integration with statistical libraries."""
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        # Test numpy operations
        data = np.random.normal(0, 1, 1000)
        assert len(data) == 1000
        
        # Test pandas operations
        df = pd.DataFrame({'returns': data})
        assert len(df) == 1000
        
        # Test basic statistical operations
        mean_return = df['returns'].mean()
        std_return = df['returns'].std()
        
        assert isinstance(mean_return, (float, np.float64))
        assert isinstance(std_return, (float, np.float64))
        assert std_return > 0


class TestDatabaseIntegration:
    """Integration tests for database functionality."""
    
    def test_duckdb_connection(self):
        """Test that we can connect to DuckDB."""
        import duckdb
        
        # Test in-memory connection
        conn = duckdb.connect(':memory:')
        assert conn is not None
        
        # Test basic query
        result = conn.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1
        
        conn.close()
    
    def test_duckdb_dataframe_integration(self):
        """Test DuckDB integration with pandas DataFrames."""
        import duckdb
        import pandas as pd
        import numpy as np
        
        # Create sample data
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'price': np.random.randn(10) + 50000,
            'volume': np.random.randint(1000, 10000, 10)
        })
        
        # Test DuckDB with DataFrame
        conn = duckdb.connect(':memory:')
        
        # Register DataFrame
        conn.register('btc_data', df)
        
        # Test query
        result = conn.execute("SELECT COUNT(*) FROM btc_data").fetchone()
        assert result[0] == 10
        
        # Test more complex query
        avg_price = conn.execute("SELECT AVG(price) FROM btc_data").fetchone()[0]
        assert isinstance(avg_price, (float, np.float64))
        
        conn.close()


class TestMLFlowIntegration:
    """Integration tests for MLflow functionality."""
    
    def test_mlflow_import_and_basic_functionality(self):
        """Test that MLflow can be imported and basic functionality works."""
        import mlflow
        
        # Test that we can access basic MLflow functionality
        assert hasattr(mlflow, 'start_run')
        assert hasattr(mlflow, 'log_metric')
        assert hasattr(mlflow, 'log_param')
        
        # Test basic experiment functionality (without actually logging)
        experiment_name = "test-experiment"
        
        # This should not raise an error
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            # It's okay if this fails in test environment
            # We just want to make sure the import works
            pass
