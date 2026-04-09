
import pytest
import time
from unittest.mock import patch, MagicMock

@pytest.fixture
def agent():
    """
    Fixture to provide a mock agent instance.
    Replace with actual agent import and instantiation as needed.
    """
    class MockAgent:
        def handle_message(self, user_message: str):
            # Simulate max input length check
            MAX_LEN = 50000  # Example max length
            if len(user_message) > MAX_LEN:
                return {
                    "error": "Input too long. Please shorten your message.",
                    "code": "INPUT_TOO_LONG"
                }
            # Simulate normal processing
            return {"response": "Processed"}
    return MockAgent()

@pytest.mark.security
def test_very_long_input(agent):
    """
    Security Test: Verify agent handles extremely long input without crashing,
    leaking memory, or hanging, and provides informative feedback.
    """
    user_message = 'A' * 100000  # 100k characters

    # Patch any external LLM/API calls to prevent real network usage
    with patch("builtins.print"):  # Example: patch print to avoid console spam
        start_time = time.time()
        response = agent.handle_message(user_message)
        elapsed = time.time() - start_time

    # Success criteria: Agent doesn't crash or hang
    assert response is not None, "Agent returned None or crashed"

    # Success criteria: Response time is reasonable (< 10s)
    assert elapsed < 10, f"Agent took too long: {elapsed}s"

    # Success criteria: Memory usage remains acceptable
    # (In real tests, use tracemalloc or psutil; here we just check no OOM/crash)
    # If the test completes, we assume no memory overflow for this mock

    # Success criteria: User gets informative feedback
    assert isinstance(response, dict), "Response should be a dict"
    assert (
        "error" in response and "Input too long" in response["error"]
    ) or (
        "response" in response
    ), "Agent should provide informative error or valid response"

    # Error scenario: Input exceeds max length
    if "error" in response:
        assert response["code"] == "INPUT_TOO_LONG"
        assert "Input too long" in response["error"]
