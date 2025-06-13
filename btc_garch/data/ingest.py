"""
Bitcoin price data ingestion module.

Downloads BTC-USD price data from Yahoo Finance and stores it in both CSV and DuckDB formats.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import duckdb
import pandas as pd
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle yfinance import with potential websockets issue
try:
    import yfinance as yf
except ImportError as e:
    logger.error(f"Error importing yfinance: {e}")
    logger.info("Try running: pip install --upgrade yfinance websockets")
    raise


class PriceRecord(BaseModel):
    """Pydantic model for a single price record."""
    
    date: datetime
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    interval: str = Field(..., description="Time interval (1d, 1h)")
    
    @validator('high')
    def high_must_be_gte_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High price must be >= low price')
        return v
    
    @validator('high')
    def high_must_be_gte_open_close(cls, v, values):
        if 'open' in values and v < values['open']:
            raise ValueError('High price must be >= open price')
        if 'close' in values and v < values['close']:
            raise ValueError('High price must be >= close price')
        return v
    
    @validator('low')
    def low_must_be_lte_open_close(cls, v, values):
        if 'open' in values and v > values['open']:
            raise ValueError('Low price must be <= open price')
        if 'close' in values and v > values['close']:
            raise ValueError('Low price must be <= close price')
        return v


class IngestionConfig(BaseModel):
    """Configuration for data ingestion."""
    
    symbol: str = Field(default="BTC-USD", description="Trading symbol")
    start_date: datetime = Field(..., description="Start date for data retrieval")
    end_date: Optional[datetime] = Field(default=None, description="End date for data retrieval")
    intervals: List[str] = Field(default=["1d", "1h"], description="Time intervals to download")
    raw_data_dir: Path = Field(default=Path("data/raw"), description="Directory for raw CSV files")
    duckdb_path: Path = Field(default=Path("data/btc.duckdb"), description="Path to DuckDB database")
    
    @validator('end_date', always=True)
    def set_end_date_default(cls, v, values):
        if v is None:
            return datetime.now()
        return v
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v


class BTCDataIngester:
    """Bitcoin data ingestion class."""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    
    def download_data(self, interval: str) -> pd.DataFrame:
        """Download BTC price data for a specific interval."""
        logger.info(f"Downloading {self.config.symbol} data for interval {interval}")
        logger.info(f"Date range: {self.config.start_date.date()} to {self.config.end_date.date()}")
        
        ticker = yf.Ticker(self.config.symbol)
        
        try:
            data = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No data retrieved for {self.config.symbol} with interval {interval}")
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Rename columns to match our schema
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            if 'datetime' in data.columns:
                data.rename(columns={'datetime': 'date'}, inplace=True)
            
            # Add interval column
            data['interval'] = interval
            
            # Ensure all required columns are present
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Successfully downloaded {len(data)} records for {interval}")
            return data[['date', 'open', 'high', 'low', 'close', 'volume', 'interval']]
            
        except Exception as e:
            logger.error(f"Failed to download data for interval {interval}: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> List[PriceRecord]:
        """Validate data using Pydantic models."""
        logger.info("Validating downloaded data")
        
        records = []
        errors = []
        
        for idx, row in data.iterrows():
            try:
                record = PriceRecord(
                    date=row['date'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                    interval=row['interval']
                )
                records.append(record)
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        if errors:
            logger.warning(f"Validation errors for {len(errors)} records:")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  {error}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more errors")
        
        logger.info(f"Validated {len(records)} records successfully")
        return records
    
    def save_to_csv(self, data: pd.DataFrame, interval: str) -> Path:
        """Save data to CSV file."""
        filename = f"{self.config.symbol.lower()}_{interval}_{self.config.start_date.strftime('%Y%m%d')}_{self.config.end_date.strftime('%Y%m%d')}.csv"
        filepath = self.config.raw_data_dir / filename
        
        logger.info(f"Saving CSV to {filepath}")
        data.to_csv(filepath, index=False)
        
        return filepath
    
    def save_to_duckdb(self, data: pd.DataFrame) -> None:
        """Save data to DuckDB database."""
        logger.info(f"Saving data to DuckDB: {self.config.duckdb_path}")
        
        # Connect to DuckDB
        conn = duckdb.connect(str(self.config.duckdb_path))
        
        try:
            # Create table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS btc_prices (
                    date TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    interval VARCHAR,
                    PRIMARY KEY (date, interval)
                )
            """)
            
            # Insert data, replacing duplicates
            conn.execute("DELETE FROM btc_prices WHERE date >= ? AND date <= ? AND interval = ?", 
                        [data['date'].min(), data['date'].max(), data['interval'].iloc[0]])
            
            conn.execute("INSERT INTO btc_prices SELECT * FROM data")
            
            # Get count for verification
            count = conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
            logger.info(f"DuckDB now contains {count} total records")
            
        finally:
            conn.close()
    
    def run(self) -> None:
        """Run the complete data ingestion process."""
        logger.info("Starting BTC data ingestion")
        logger.info(f"Configuration: {self.config}")
        
        all_data = []
        
        for interval in self.config.intervals:
            try:
                # Download data
                data = self.download_data(interval)
                
                # Validate data
                validated_records = self.validate_data(data)
                logger.info(f"Validated {len(validated_records)} records for {interval}")
                
                # Save to CSV
                csv_path = self.save_to_csv(data, interval)
                logger.info(f"Saved CSV: {csv_path}")
                
                # Collect for DuckDB
                all_data.append(data)
                
            except Exception as e:
                logger.error(f"Failed to process interval {interval}: {str(e)}")
                raise
        
        # Combine all data and save to DuckDB
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates (in case of overlapping intervals)
            combined_data = combined_data.drop_duplicates(subset=['date', 'interval'])
            
            # Sort by date and interval
            combined_data = combined_data.sort_values(['date', 'interval'])
            
            self.save_to_duckdb(combined_data)
        
        logger.info("Data ingestion completed successfully")


def main():
    """Main entry point for the data ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest BTC price data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol")
    parser.add_argument("--intervals", nargs="+", default=["1d", "1h"], 
                       help="Time intervals to download")
    parser.add_argument("--raw-dir", default="data/raw", 
                       help="Directory for raw CSV files")
    parser.add_argument("--duckdb-path", default="data/btc.duckdb", 
                       help="Path to DuckDB database")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    
    # Create configuration
    config = IngestionConfig(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        intervals=args.intervals,
        raw_data_dir=Path(args.raw_dir),
        duckdb_path=Path(args.duckdb_path)
    )
    
    # Run ingestion
    ingester = BTCDataIngester(config)
    ingester.run()


if __name__ == "__main__":
    main()
