
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Configuration management for CRM Quote Creation Agent.
    Handles environment variable loading, API key management,
    LLM config, domain settings, validation, error handling, and defaults.
    """

    # --- Environment Variable Loading ---
    @staticmethod
    def get_env(key: str, default=None, required: bool = False) -> str:
        value = os.getenv(key, default)
        if required and not value:
            raise ConfigError(f"Missing required environment variable: {key}")
        return value

    # --- API Key Management ---
    @property
    def AZURE_SEARCH_ENDPOINT(self) -> str:
        return self.get_env("AZURE_SEARCH_ENDPOINT", required=True)

    @property
    def AZURE_SEARCH_API_KEY(self) -> str:
        return self.get_env("AZURE_SEARCH_API_KEY", required=True)

    @property
    def AZURE_SEARCH_INDEX_NAME(self) -> str:
        return self.get_env("AZURE_SEARCH_INDEX_NAME", required=True)

    @property
    def AZURE_OPENAI_ENDPOINT(self) -> str:
        return self.get_env("AZURE_OPENAI_ENDPOINT", required=True)

    @property
    def AZURE_OPENAI_API_KEY(self) -> str:
        return self.get_env("AZURE_OPENAI_API_KEY", required=True)

    @property
    def AZURE_OPENAI_EMBEDDING_DEPLOYMENT(self) -> str:
        return self.get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", required=True)

    @property
    def AZURE_OPENAI_LLM_DEPLOYMENT(self) -> str:
        # Optional override, default to gpt-4.1-aba
        return self.get_env("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4.1-aba")

    # --- LLM Configuration ---
    @property
    def LLM_CONFIG(self) -> dict:
        return {
            "provider": "azure",
            "model": self.AZURE_OPENAI_LLM_DEPLOYMENT,
            "temperature": float(self.get_env("LLM_TEMPERATURE", 0.7)),
            "max_tokens": int(self.get_env("LLM_MAX_TOKENS", 2000)),
            "system_prompt": (
                "You are a professional CRM Quote Creation Agent. Your role is to automate the creation of quotes and quote orders in a CRM system, "
                "following strict business rules and validations. For each request: - Validate that customerEmailId is present and in a valid email format. "
                "- Validate that receivedDateTime is present and parseable as ISO-8601. - Resolve the customer in CRM using the provided email. "
                "If not found or ineligible, return a clear error. - Resolve all required fields (Contact, Account, bill-to/ship-to, currency, price list, "
                "payment terms, tax region). If any are missing, request only the minimum additional information needed. - Ensure at least one valid line item is present. "
                "If not, prompt for product/SKU, quantity, requested dates, and shipping requirements. - Do not guess or autofill any required values. "
                "- Interact with CRM APIs to create the quote and quote order, enforcing idempotency and capturing all relevant identifiers and statuses. "
                "- Return a structured response: On success, provide quoteId, quoteOrderId, status, timestamps, and any warnings. On failure, provide a concise list of missing/invalid inputs, "
                "lookup failures, API errors, and the minimal missing inputs required to continue. - If information is not found in the knowledge base, respond with a clear fallback message."
            ),
            "user_prompt_template": (
                "Please provide the following details to create a quote: customer email address and the date/time the request was received (ISO-8601). "
                "If you have product/SKU, quantity, and requested dates, please include them. I will guide you through any additional required information."
            ),
            "few_shot_examples": [
                'Input: { "customerEmailId": "jane.doe@example.com", "receivedDateTime": "2024-06-01T10:00:00Z" } Response: { "status": "FAIL", "errors": ["Missing required line item details: product/SKU, quantity, requested dates"], "missingInputs": ["product/SKU", "quantity", "requested start/end dates"], "requestId": "REQ-12345", "correlationId": "CORR-67890", "validationSummary": "Line item details required to proceed." }',
                'Input: { "customerEmailId": "john.smith@example.com", "receivedDateTime": "2024-06-01T09:30:00Z", "lineItems": [{ "product": "SKU-001", "quantity": 2, "startDate": "2024-06-10", "endDate": "2024-06-20" }] } Response: { "status": "PASS", "quoteId": "Q-10001", "quoteOrderId": "O-20001", "timestamps": { "created": "2024-06-01T09:31:00Z" }, "warnings": [], "requestId": "REQ-54321", "correlationId": "CORR-09876", "validationSummary": "Quote and order created successfully." }'
            ]
        }

    # --- Domain-Specific Settings ---
    @property
    def DOMAIN(self) -> str:
        return "general"

    @property
    def RAG_CONFIG(self) -> dict:
        return {
            "enabled": True,
            "retrieval_service": "azure_ai_search",
            "embedding_model": "text-embedding-ada-002",
            "top_k": int(self.get_env("RAG_TOP_K", 5)),
            "search_type": "vector_semantic"
        }

    @property
    def REQUIRED_CONFIG_KEYS(self) -> list:
        return [
            "customerEmailId",
            "receivedDateTime"
        ]

    # --- Validation and Error Handling ---
    @staticmethod
    def validate_config():
        required_keys = [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ]
        missing = []
        for key in required_keys:
            if not os.getenv(key):
                missing.append(key)
        if missing:
            raise ConfigError(f"Missing required API keys or environment variables: {', '.join(missing)}")

    # --- Default Values and Fallbacks ---
    @property
    def DEFAULT_CURRENCY(self) -> str:
        return self.get_env("DEFAULT_CURRENCY", "USD")

    @property
    def DEFAULT_PAYMENT_TERMS(self) -> str:
        return self.get_env("DEFAULT_PAYMENT_TERMS", "Net 30")

    @property
    def FALLBACK_RESPONSE(self) -> str:
        return (
            "The requested information is not available in the current knowledge base. "
            "Please provide additional details or contact support for further assistance."
        )

    # --- Logging Setup ---
    @staticmethod
    def setup_logging():
        logger = logging.getLogger("crm_quote_agent_config")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

# Instantiate and validate config at import time
config = Config()
try:
    config.validate_config()
except ConfigError as e:
    logger = Config.setup_logging()
    logger.error(str(e))
    raise

