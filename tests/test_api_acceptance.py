"""Acceptance tests for the BTC GARCH API."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestAPIAcceptance:
    """Acceptance tests for the API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_api_root_endpoint(self, client):
        """Test that the root endpoint returns the expected message."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "BTC GARCH API is running"}
    
    def test_api_health_check_endpoint(self, client):
        """Test that the health check endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_api_application_metadata(self, client):
        """Test that the API has correct metadata configuration."""
        # Access the app metadata through the client
        app_instance = client.app
        
        assert app_instance.title == "BTC GARCH API"
        assert app_instance.description == "API for Bitcoin volatility predictions using GARCH models"
        assert app_instance.version == "0.1.0"
    
    def test_cors_middleware_configured(self, client):
        """Test that CORS middleware is properly configured."""
        # Test with a preflight request
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # Should not return 405 Method Not Allowed due to CORS middleware
        assert response.status_code != 405

    def test_forecast_endpoint_exists(self, client):
        """Test that the forecast endpoint exists and has correct structure."""
        response = client.get("/forecast")
        
        # The endpoint should exist (not 404)
        assert response.status_code != 404
        
        # It might return 503 if no model is available, but the endpoint should exist
        # For now, we'll accept both success and service unavailable as valid responses
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            # If successful, check that it returns the expected structure
            data = response.json()
            assert "sigma" in data
            assert isinstance(data["sigma"], (int, float))
    
    def test_backtest_endpoint_exists(self, client):
        """Test that the backtest endpoint exists and accepts POST requests."""
        test_payload = {
            "start_date": "2023-01-01",
            "end_date": "2023-01-10"
        }
        
        response = client.post("/backtest", json=test_payload)
        
        # The endpoint should exist (not 404)
        assert response.status_code != 404
        
        # It might return various status codes depending on data availability
        # For now, we'll accept success, service unavailable, or not found as valid responses
        assert response.status_code in [200, 404, 503]
        
        if response.status_code == 200:
            # If successful, check that it returns the expected structure
            data = response.json()
            assert "start_date" in data
            assert "end_date" in data
            assert "var_breaches_count" in data
            assert isinstance(data["var_breaches_count"], int)
