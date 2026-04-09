
import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict

@pytest.fixture
def valid_quote_payload() -> Dict[str, Any]:
    """Fixture providing a valid payload for CRM quote creation."""
    return {
        "customerEmailId": "customer@example.com",
        "receivedDateTime": "2024-06-01T12:00:00Z",
        "lineItems": [
            {
                "productId": "PROD-001",
                "quantity": 2,
                "price": 100.0
            },
            {
                "productId": "PROD-002",
                "quantity": 1,
                "price": 250.0
            }
        ]
    }

@pytest.fixture
def mock_crm_quote_response() -> Dict[str, Any]:
    """Fixture providing a mocked successful CRM quote creation response."""
    return {
        "success": True,
        "data": {
            "status": "PASS",
            "quoteId": "Q-123456",
            "quoteOrderId": "QO-654321",
            "timestamps": {
                "created": "2024-06-01T12:01:00Z",
                "updated": "2024-06-01T12:01:00Z"
            },
            "validationSummary": {
                "missingInputs": [],
                "errors": []
            }
        }
    }

@pytest.fixture
def client():
    """
    Fixture for a test client.
    Replace this with the actual test client for your web framework (e.g., Flask, FastAPI).
    """
    from fastapi.testclient import TestClient
    from main import app  # Replace with your actual app import
    return TestClient(app)

def test_functional_successful_crm_quote_creation(
    client,
    valid_quote_payload,
    mock_crm_quote_response
):
    """
    Functional test: Validates the end-to-end workflow for a successful CRM quote creation
    via the /crm/quote endpoint with all required fields present and valid.
    """
    # Patch the backend logic that would make any external HTTP/database calls
    # and return the mocked response instead.
    # Assume the endpoint internally calls a function like create_crm_quote(...)
    # which we will patch. Adjust the import path as needed.
    with patch("main.create_crm_quote", return_value=mock_crm_quote_response):
        response = client.post("/crm/quote", json=valid_quote_payload)
        assert response.status_code == 200, "HTTP status code is not 200"
        resp_json = response.json()
        assert resp_json.get("success") is True, "Response 'success' field is not True"
        data = resp_json.get("data", {})
        assert data.get("status") == "PASS", "Response data 'status' is not 'PASS'"
        assert "quoteId" in data, "'quoteId' missing in response data"
        assert "quoteOrderId" in data, "'quoteOrderId' missing in response data"
        assert "timestamps" in data, "'timestamps' missing in response data"
        validation_summary = data.get("validationSummary", {})
        assert validation_summary.get("missingInputs") == [], "There should be no missingInputs"
        assert validation_summary.get("errors") == [], "There should be no errors"
