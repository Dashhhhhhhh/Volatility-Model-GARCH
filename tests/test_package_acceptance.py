"""Acceptance tests for the BTC GARCH package structure and imports."""

import pytest
import importlib
import sys
from pathlib import Path


class TestPackageStructure:
    """Test that the package structure is correct and modules can be imported."""
    
    def test_main_package_import(self):
        """Test that the main btc_garch package can be imported."""
        import btc_garch
        
        assert hasattr(btc_garch, '__version__')
        assert btc_garch.__version__ == "0.1.0"
        assert hasattr(btc_garch, '__author__')
        assert hasattr(btc_garch, '__description__')
    
    def test_api_module_import(self):
        """Test that the API module can be imported."""
        import api
        assert api is not None
        
        # Test that the main FastAPI app can be imported
        from api.main import app
        assert app is not None
        assert hasattr(app, 'title')
    
    def test_data_module_import(self):
        """Test that the data module can be imported."""
        import data
        assert data is not None
    
    def test_modeling_module_import(self):
        """Test that the modeling module can be imported."""
        import modeling
        assert modeling is not None
    
    def test_dashboard_module_import(self):
        """Test that the dashboard module can be imported."""
        import dashboard
        assert dashboard is not None
    
    def test_utils_module_import(self):
        """Test that the utils module can be imported."""
        import utils
        assert utils is not None
    
    def test_required_dependencies_available(self):
        """Test that all required dependencies are available."""
        required_packages = [
            'arch',
            'pandas',
            'numpy',
            'yfinance',
            'matplotlib',
            'plotly',
            'prefect',
            'fastapi',
            'uvicorn',
            'duckdb',
            'pydantic',
            'mlflow',
            'optuna'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        assert len(missing_packages) == 0, f"Missing required packages: {missing_packages}"
    
    def test_project_structure_exists(self):
        """Test that all expected directories exist."""
        project_root = Path.cwd()
        expected_dirs = [
            'api',
            'btc_garch', 
            'data',
            'modeling',
            'dashboard',
            'utils',
            'tests'
        ]
        
        missing_dirs = []
        for dir_name in expected_dirs:
            if not (project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        assert len(missing_dirs) == 0, f"Missing expected directories: {missing_dirs}"
    
    def test_configuration_files_exist(self):
        """Test that important configuration files exist."""
        project_root = Path.cwd()
        expected_files = [
            'pyproject.toml',
            'README.md',
            'Dockerfile',
            '.env.example'
        ]
        
        missing_files = []
        for file_name in expected_files:
            if not (project_root / file_name).exists():
                missing_files.append(file_name)
        
        assert len(missing_files) == 0, f"Missing expected files: {missing_files}"
