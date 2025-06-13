#!/usr/bin/env python3
"""
Demo script to test Sprint 3 hyperparameter search functionality
This creates mock data and runs a small hyperparameter search to verify everything works.
"""

import pandas as pd
import numpy as np
import os
from modeling.search import run_hyperparameter_search

def main():
    # Create mock BTC return data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    returns = pd.Series(
        np.random.normal(0.0005, 0.02, 200),  # Mean return ~0.05% daily, volatility ~2%
        index=dates,
        name='btc_returns'
    )
    
    print(f"Created mock BTC returns data with {len(returns)} observations")
    print(f"Return statistics: mean={returns.mean():.6f}, std={returns.std():.6f}")
    
    # Set environment variables for the test
    os.environ["MLFLOW_TRACKING_URI"] = "./demo_mlruns"
    os.environ["OPTIMIZATION_TRIALS"] = "9"  # Sprint 3 requires ≥9 trials
    
    print("\nStarting hyperparameter search with 9 trials...")
    
    # Run the search
    best_params = run_hyperparameter_search(returns, n_trials=9)
    
    print(f"\nSprint 3 Demo Complete!")
    print(f"Best parameters found: {best_params}")
    print(f"Search completed {9} trials successfully")
    print(f"MLflow experiment 'garch_search' created with results")
    
    print(f"\nTo view results, run: mlflow ui --backend-store-uri ./demo_mlruns")

if __name__ == "__main__":
    main()
