"""
Bitcoin price data ingestion script.
This is a wrapper that calls the main ingestion module.
"""

from btc_garch.data.ingest import main

if __name__ == "__main__":
    main()
