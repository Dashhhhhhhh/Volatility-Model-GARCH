"""
Tests for the BTC data ingestion module.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from btc_garch.data.ingest import BTCDataIngester, IngestionConfig, PriceRecord


class TestPriceRecord:
    """Test the PriceRecord Pydantic model."""
    
    def test_valid_price_record(self):
        """Test creating a valid price record."""
        record = PriceRecord(
            date=datetime(2023, 1, 1),
            open=20000.0,
            high=21000.0,
            low=19500.0,
            close=20500.0,
            volume=1000000,
            interval="1d"
        )
        assert record.open == 20000.0
        assert record.high == 21000.0
        assert record.low == 19500.0
        assert record.close == 20500.0
        assert record.volume == 1000000
        assert record.interval == "1d"
    
    def test_invalid_high_low_relationship(self):
        """Test that high must be >= low."""
        with pytest.raises(ValueError, match="High price must be >= low price"):
            PriceRecord(
                date=datetime(2023, 1, 1),
                open=20000.0,
                high=19000.0,  # High < Low
                low=19500.0,
                close=20500.0,
                volume=1000000,
                interval="1d"
            )
    
    def test_negative_prices(self):
        """Test that prices must be positive."""
        with pytest.raises(ValueError):
            PriceRecord(
                date=datetime(2023, 1, 1),
                open=-20000.0,  # Negative price
                high=21000.0,
                low=19500.0,
                close=20500.0,
                volume=1000000,
                interval="1d"
            )


class TestIngestionConfig:
    """Test the IngestionConfig Pydantic model."""
    
    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = IngestionConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        assert config.symbol == "BTC-USD"
        assert config.start_date == datetime(2023, 1, 1)
        assert config.end_date == datetime(2023, 12, 31)
        assert config.intervals == ["1d", "1h"]
    
    def test_end_date_defaults_to_now(self):
        """Test that end_date defaults to current time."""
        config = IngestionConfig(start_date=datetime(2023, 1, 1))
        assert config.end_date is not None
        assert config.end_date > config.start_date
    
    def test_invalid_date_range(self):
        """Test that end_date must be after start_date."""
        with pytest.raises(ValueError, match="End date must be after start date"):
            IngestionConfig(
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1)  # End before start
            )


class TestBTCDataIngester:
    """Test the BTCDataIngester class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = IngestionConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 7),  # Short range for testing
                raw_data_dir=temp_path / "raw",
                duckdb_path=temp_path / "test.duckdb"
            )
            yield config
    
    def test_ingester_initialization(self, temp_config):
        """Test that the ingester initializes correctly."""
        ingester = BTCDataIngester(temp_config)
        assert ingester.config == temp_config
        assert temp_config.raw_data_dir.exists()
        assert temp_config.duckdb_path.parent.exists()
    
    def test_download_data_daily(self, temp_config):
        """Test downloading daily data."""
        ingester = BTCDataIngester(temp_config)
        
        # Download a small amount of data
        data = ingester.download_data("1d")
        
        assert not data.empty
        assert "date" in data.columns
        assert "open" in data.columns
        assert "high" in data.columns
        assert "low" in data.columns
        assert "close" in data.columns
        assert "volume" in data.columns
        assert "interval" in data.columns
        assert all(data["interval"] == "1d")
    
    def test_validate_data(self, temp_config):
        """Test data validation."""
        ingester = BTCDataIngester(temp_config)
        
        # Create sample data
        sample_data = pd.DataFrame({
            "date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "open": [20000.0, 20100.0],
            "high": [21000.0, 21100.0],
            "low": [19500.0, 19600.0],
            "close": [20500.0, 20600.0],
            "volume": [1000000, 1100000],
            "interval": ["1d", "1d"]
        })
        
        records = ingester.validate_data(sample_data)
        assert len(records) == 2
        assert all(isinstance(record, PriceRecord) for record in records)
    
    def test_save_to_csv(self, temp_config):
        """Test saving data to CSV."""
        ingester = BTCDataIngester(temp_config)
        
        # Create sample data
        sample_data = pd.DataFrame({
            "date": [datetime(2023, 1, 1)],
            "open": [20000.0],
            "high": [21000.0],
            "low": [19500.0],
            "close": [20500.0],
            "volume": [1000000],
            "interval": ["1d"]
        })
        
        csv_path = ingester.save_to_csv(sample_data, "1d")
        assert csv_path.exists()
        
        # Verify content
        loaded_data = pd.read_csv(csv_path)
        assert len(loaded_data) == 1
        assert loaded_data.iloc[0]["open"] == 20000.0
    
    def test_save_to_duckdb(self, temp_config):
        """Test saving data to DuckDB."""
        ingester = BTCDataIngester(temp_config)
        
        # Create sample data
        sample_data = pd.DataFrame({
            "date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "open": [20000.0, 20100.0],
            "high": [21000.0, 21100.0],
            "low": [19500.0, 19600.0],
            "close": [20500.0, 20600.0],
            "volume": [1000000, 1100000],
            "interval": ["1d", "1d"]
        })
        
        ingester.save_to_duckdb(sample_data)
        
        # Verify data was saved
        conn = duckdb.connect(str(temp_config.duckdb_path))
        try:
            result = conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()
            assert result[0] == 2
            
            # Check that close prices are not null
            result = conn.execute("SELECT close FROM btc_prices WHERE close IS NULL").fetchall()
            assert len(result) == 0
            
            # Check date range
            result = conn.execute("SELECT MIN(date), MAX(date) FROM btc_prices").fetchone()
            assert result[0] <= datetime(2023, 1, 1)
            assert result[1] >= datetime(2023, 1, 2)
            
        finally:
            conn.close()


def test_integration_basic_ingestion():
    """Integration test for basic data ingestion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        config = IngestionConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3),  # Very short range
            intervals=["1d"],  # Only daily to speed up test
            raw_data_dir=temp_path / "raw",
            duckdb_path=temp_path / "btc.duckdb"
        )
        
        ingester = BTCDataIngester(config)
        ingester.run()
        
        # Verify CSV file was created
        csv_files = list(config.raw_data_dir.glob("*.csv"))
        assert len(csv_files) >= 1
        
        # Verify DuckDB was created and has data
        assert config.duckdb_path.exists()
        
        conn = duckdb.connect(str(config.duckdb_path))
        try:
            # Check table exists and has data
            result = conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()
            assert result[0] > 0
            
            # Check non-null close prices
            result = conn.execute("SELECT COUNT(*) FROM btc_prices WHERE close IS NOT NULL").fetchone()
            assert result[0] > 0
            
            # Check date range spans requested dates
            result = conn.execute("SELECT MIN(date), MAX(date) FROM btc_prices").fetchone()
            min_date, max_date = result
            assert min_date.date() >= datetime(2023, 1, 1).date()
            assert max_date.date() <= datetime(2023, 1, 3).date()
            
        finally:
            conn.close()


if __name__ == "__main__":
    # Run a basic test
    test_integration_basic_ingestion()
    print("Integration test passed!")
