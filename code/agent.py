try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 4,
 'runtime_enabled': True,
 'sanitize_pii': True}


import os
import logging
import uuid
import time as _time
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator
from email_validator import validate_email, EmailNotValidError
from dotenv import load_dotenv
from datetime import datetime
from threading import Lock

# Azure AI Search and OpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

# Observability wrappers (trace_step, etc.) are injected by the runtime, do not import manually

# Load .env if present
load_dotenv()

# Logging configuration
logger = logging.getLogger("crm_quote_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# --- Configuration Management ---

class Config:
    """Lazy configuration loader for environment variables."""
    @staticmethod
    def get_azure_search_endpoint() -> str:
        val = os.getenv("AZURE_SEARCH_ENDPOINT")
        if not val:
            raise RuntimeError("AZURE_SEARCH_ENDPOINT not set")
        return val

    @staticmethod
    def get_azure_search_api_key() -> str:
        val = os.getenv("AZURE_SEARCH_API_KEY")
        if not val:
            raise RuntimeError("AZURE_SEARCH_API_KEY not set")
        return val

    @staticmethod
    def get_azure_search_index_name() -> str:
        val = os.getenv("AZURE_SEARCH_INDEX_NAME")
        if not val:
            raise RuntimeError("AZURE_SEARCH_INDEX_NAME not set")
        return val

    @staticmethod
    def get_azure_openai_endpoint() -> str:
        val = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not val:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT not set")
        return val

    @staticmethod
    def get_azure_openai_api_key() -> str:
        val = os.getenv("AZURE_OPENAI_API_KEY")
        if not val:
            raise RuntimeError("AZURE_OPENAI_API_KEY not set")
        return val

    @staticmethod
    def get_azure_openai_embedding_deployment() -> str:
        val = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if not val:
            raise RuntimeError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not set")
        return val

    @staticmethod
    @trace_agent(agent_name='CRM Quote Creation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_azure_openai_llm_deployment() -> str:
        # Model name for LLM (e.g., "gpt-4.1-aba")
        val = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4.1-aba")
        return val

    @staticmethod
    def get_rag_top_k() -> int:
        return int(os.getenv("RAG_TOP_K", "5"))

    @staticmethod
    def validate():
        # Optional: Validate all required config keys
        try:
            Config.get_azure_search_endpoint()
            Config.get_azure_search_api_key()
            Config.get_azure_search_index_name()
            Config.get_azure_openai_endpoint()
            Config.get_azure_openai_api_key()
            Config.get_azure_openai_embedding_deployment()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

# --- Error Classes ---

class InputValidationError(Exception):
    pass

class DuplicateRequestError(Exception):
    pass

class CRMApiError(Exception):
    pass

class EntityResolutionError(Exception):
    pass

class BusinessRuleViolation(Exception):
    pass

class FallbackKnowledgeBaseError(Exception):
    pass

# --- Pydantic Models ---

class LineItemModel(BaseModel):
    product: Optional[str] = None
    quantity: Optional[int] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None

    @field_validator("quantity")
    def validate_quantity(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @field_validator("startDate", "endDate")
    def validate_dates(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except Exception:
                raise ValueError("Date must be ISO-8601 format")
        return v

class CRMQuoteRequestModel(BaseModel):
    customerEmailId: Optional[str] = None
    receivedDateTime: Optional[str] = None
    lineItems: Optional[List[LineItemModel]] = None
    requestId: Optional[str] = None
    correlationId: Optional[str] = None

    @field_validator("customerEmailId")
    def validate_email(cls, v):
        if v is not None:
            try:
                validate_email(v)
            except EmailNotValidError:
                raise ValueError("Invalid email format")
        return v

    @field_validator("receivedDateTime")
    def validate_datetime(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except Exception:
                raise ValueError("Invalid ISO-8601 datetime format")
        return v

    @model_validator(mode="after")
    def check_minimum_fields(self):
        if not self.customerEmailId:
            raise ValueError("customerEmailId is required")
        if not self.receivedDateTime:
            raise ValueError("receivedDateTime is required")
        return self

class ValidationResult(BaseModel):
    valid: bool
    errors: List[str] = []
    missing_inputs: List[str] = []
    validation_summary: Optional[str] = None

class RuleApplicationResult(BaseModel):
    passed: bool
    errors: List[str] = []
    missing_inputs: List[str] = []
    summary: Optional[str] = None

class ResolvedEntities(BaseModel):
    contactId: Optional[str] = None
    accountId: Optional[str] = None
    billTo: Optional[str] = None
    shipTo: Optional[str] = None
    currency: Optional[str] = None
    priceList: Optional[str] = None
    paymentTerms: Optional[str] = None
    taxRegion: Optional[str] = None
    missing_fields: List[str] = []

class QuoteCreationResult(BaseModel):
    success: bool
    quoteId: Optional[str] = None
    status: Optional[str] = None
    errors: List[str] = []
    warnings: List[str] = []
    timestamps: Optional[Dict[str, Any]] = None

class QuoteOrderCreationResult(BaseModel):
    success: bool
    quoteOrderId: Optional[str] = None
    status: Optional[str] = None
    errors: List[str] = []
    warnings: List[str] = []
    timestamps: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    message: str
    tips: Optional[str] = None
    requestId: Optional[str] = None
    correlationId: Optional[str] = None

class SuccessResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]

# --- Service Base ---

class ServiceBase:
    def __init__(self, audit_logger=None):
        self.audit_logger = audit_logger

# --- Audit Logging Service ---

class AuditLoggingService(ServiceBase):
    """Logs all actions, errors, and responses for traceability and compliance."""
    def log_action(self, action: str, details: Dict[str, Any]):
        try:
            logger.info(f"AUDIT: {action} | {self._mask_pii(details)}")
        except Exception as e:
            logger.error(f"Failed to log audit action: {e}")

    def _mask_pii(self, details: Dict[str, Any]) -> Dict[str, Any]:
        # Mask email addresses and other PII
        masked = {}
        for k, v in details.items():
            if isinstance(v, str) and "@" in v:
                parts = v.split("@")
                if len(parts) == 2:
                    masked[k] = parts[0][:2] + "***@" + parts[1]
                else:
                    masked[k] = "***"
            else:
                masked[k] = v
        return masked

# --- Idempotency Service ---

class IdempotencyService(ServiceBase):
    """Detect and prevent duplicate requests using requestId/correlationId, manage short-term cache."""
    _cache = {}
    _lock = Lock()
    _cache_ttl = 60 * 10  # 10 minutes

    def check_duplicate(self, requestId: Optional[str], correlationId: Optional[str]) -> bool:
        now = _time.time()
        key = f"{requestId or ''}:{correlationId or ''}"
        with self._lock:
            # Purge expired
            expired = [k for k, v in self._cache.items() if now - v > self._cache_ttl]
            for k in expired:
                del self._cache[k]
            if key in self._cache:
                return True
            self._cache[key] = now
        return False

# --- Input Processor ---

class InputProcessor(ServiceBase):
    """Parse and normalize incoming requests from email/API triggers."""
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def process_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Sanitize and normalize input
            cleaned = {k: v for k, v in request.items() if v is not None}
            return cleaned
        except Exception as e:
            raise InputValidationError(f"Failed to process input: {e}")

# --- Validation Service ---

class ValidationService(ServiceBase):
    """Validate presence and format of required fields."""
    def validate_fields(self, input_data: Dict[str, Any]) -> ValidationResult:
        errors = []
        missing = []
        summary = []
        # Email
        email = input_data.get("customerEmailId")
        if not email:
            missing.append("customerEmailId")
            errors.append("Missing required field: customerEmailId")
        else:
            try:
                validate_email(email)
            except EmailNotValidError:
                errors.append("Invalid email format")
        # receivedDateTime
        dt = input_data.get("receivedDateTime")
        if not dt:
            missing.append("receivedDateTime")
            errors.append("Missing required field: receivedDateTime")
        else:
            try:
                datetime.fromisoformat(dt.replace("Z", "+00:00"))
            except Exception:
                errors.append("Invalid ISO-8601 datetime format")
        # Line items
        line_items = input_data.get("lineItems")
        if not line_items or not isinstance(line_items, list) or len(line_items) == 0:
            missing.extend(["product/SKU", "quantity", "requested start/end dates"])
            errors.append("Missing required line item details: product/SKU, quantity, requested dates")
        else:
            for idx, item in enumerate(line_items):
                if not item.get("product"):
                    missing.append(f"lineItems[{idx}].product")
                if not item.get("quantity"):
                    missing.append(f"lineItems[{idx}].quantity")
                if not item.get("startDate"):
                    missing.append(f"lineItems[{idx}].startDate")
                if not item.get("endDate"):
                    missing.append(f"lineItems[{idx}].endDate")
        valid = len(errors) == 0
        if valid:
            summary.append("All required fields present and valid.")
        else:
            summary.append("Validation failed: " + "; ".join(errors))
        return ValidationResult(
            valid=valid,
            errors=errors,
            missing_inputs=missing,
            validation_summary=" ".join(summary)
        )

# --- Business Rule Engine ---

class BusinessRuleEngine(ServiceBase):
    """Apply business rules, eligibility checks, and decision tables."""
    def apply_rules(self, input_data: Dict[str, Any]) -> RuleApplicationResult:
        errors = []
        missing = []
        summary = []
        # Simulate CRM customer lookup
        customer_email = input_data.get("customerEmailId")
        # For demo, treat all emails as found and eligible except "notfound@example.com"
        if customer_email == "notfound@example.com":
            errors.append("Customer not found in CRM")
            summary.append("Customer not found")
            return RuleApplicationResult(
                passed=False,
                errors=errors,
                missing_inputs=["customerEmailId"],
                summary="Customer not found"
            )
        if customer_email == "ineligible@example.com":
            errors.append("Customer not eligible for quotes")
            summary.append("Customer not eligible")
            return RuleApplicationResult(
                passed=False,
                errors=errors,
                missing_inputs=["customerEmailId"],
                summary="Customer not eligible"
            )
        # All other emails pass
        summary.append("Business rules passed")
        return RuleApplicationResult(
            passed=True,
            errors=[],
            missing_inputs=[],
            summary="Business rules passed"
        )

# --- Data Resolution Service ---

class DataResolutionService(ServiceBase):
    """Resolve CRM entities (Contact, Account, bill-to, ship-to, etc.) and required fields."""
    def resolve_entities(self, input_data: Dict[str, Any]) -> ResolvedEntities:
        # Simulate CRM entity resolution
        missing = []
        # For demo, resolve everything except if email is "missingfields@example.com"
        if input_data.get("customerEmailId") == "missingfields@example.com":
            missing = ["billTo", "shipTo", "currency"]
            return ResolvedEntities(
                contactId=str(uuid.uuid4()),
                accountId=str(uuid.uuid4()),
                missing_fields=missing
            )
        # Otherwise, resolve all
        return ResolvedEntities(
            contactId=str(uuid.uuid4()),
            accountId=str(uuid.uuid4()),
            billTo="BILL-123",
            shipTo="SHIP-456",
            currency="USD",
            priceList="PL-789",
            paymentTerms="Net 30",
            taxRegion="US",
            missing_fields=[]
        )

# --- API Integration Service ---

class APIIntegrationService(ServiceBase):
    """Interact with CRM APIs to create quotes and quote orders, enforce idempotency."""
    def create_quote(self, payload: Dict[str, Any]) -> QuoteCreationResult:
        # Simulate CRM API call
        try:
            # Simulate transient error for demo
            if payload.get("simulate_api_error"):
                raise CRMApiError("Simulated CRM API error")
            quote_id = f"Q-{uuid.uuid4().hex[:8]}"
            return QuoteCreationResult(
                success=True,
                quoteId=quote_id,
                status="CREATED",
                warnings=[],
                timestamps={"created": datetime.utcnow().isoformat() + "Z"}
            )
        except Exception as e:
            return QuoteCreationResult(
                success=False,
                errors=[str(e)],
                warnings=[],
                timestamps={"failed": datetime.utcnow().isoformat() + "Z"}
            )

    def create_quote_order(self, payload: Dict[str, Any]) -> QuoteOrderCreationResult:
        # Simulate CRM API call
        try:
            if payload.get("simulate_api_error"):
                raise CRMApiError("Simulated CRM API error")
            quote_order_id = f"O-{uuid.uuid4().hex[:8]}"
            return QuoteOrderCreationResult(
                success=True,
                quoteOrderId=quote_order_id,
                status="CREATED",
                warnings=[],
                timestamps={"created": datetime.utcnow().isoformat() + "Z"}
            )
        except Exception as e:
            return QuoteOrderCreationResult(
                success=False,
                errors=[str(e)],
                warnings=[],
                timestamps={"failed": datetime.utcnow().isoformat() + "Z"}
            )

# --- Azure OpenAI Embedding Service ---

class AzureOpenAIEmbedding(ServiceBase):
    """Embed user queries for semantic search in Azure AI Search."""
    def __init__(self):
        super().__init__()
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = openai.AzureOpenAI(
                api_key=Config.get_azure_openai_api_key(),
                api_version="2024-02-01",
                azure_endpoint=Config.get_azure_openai_endpoint(),
            )
        return self._client

    @trace_agent(agent_name='CRM Quote Creation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def embed_query(self, text: str) -> List[float]:
        try:
            client = self._get_client()
            resp = client.embeddings.create(
                input=text,
                model=Config.get_azure_openai_embedding_deployment()
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

# --- Azure AI Search Retriever ---

class AzureAISearchRetriever(ServiceBase):
    """Query Azure AI Search for document context, retrieve relevant chunks."""
    def __init__(self, embedding_service: AzureOpenAIEmbedding):
        super().__init__()
        self.embedding_service = embedding_service
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = SearchClient(
                endpoint=Config.get_azure_search_endpoint(),
                index_name=Config.get_azure_search_index_name(),
                credential=AzureKeyCredential(Config.get_azure_search_api_key()),
            )
        return self._client

    @trace_agent(agent_name='CRM Quote Creation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def retrieve_context(self, query: str) -> List[str]:
        try:
            embedding = self.embedding_service.embed_query(query)
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=Config.get_rag_top_k(),
                fields="vector"
            )
            client = self._get_client()
            results = client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=Config.get_rag_top_k(),
                select=["chunk", "title"]
            )
            context_chunks = [r["chunk"] for r in results if r.get("chunk")]
            return context_chunks
        except Exception as e:
            logger.error(f"Azure AI Search retrieval failed: {e}")
            return []

# --- Response Generator ---

class ResponseGenerator(ServiceBase):
    """Format structured JSON responses for success/failure scenarios."""
    @trace_agent(agent_name='CRM Quote Creation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def generate_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Output contract
            response = {
                "status": result.get("status", "FAIL"),
                "quoteId": result.get("quoteId"),
                "quoteOrderId": result.get("quoteOrderId"),
                "timestamps": result.get("timestamps"),
                "warnings": result.get("warnings", []),
                "errors": result.get("errors", []),
                "missingInputs": result.get("missingInputs", []),
                "requestId": result.get("requestId"),
                "correlationId": result.get("correlationId"),
                "validationSummary": result.get("validationSummary", "")
            }
            # Remove keys with None values
            return {k: v for k, v in response.items() if v is not None}
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return {
                "status": "FAIL",
                "errors": ["Response formatting failed"],
                "validationSummary": "Response formatting failed"
            }

# --- Error Handler ---

class ErrorHandler(ServiceBase):
    """Manage errors, retries, fallback behaviors, and escalate to HITL if needed."""
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResponse:
        error_type = type(error).__name__
        message = str(error)
        tips = None
        if isinstance(error, InputValidationError):
            tips = "Check required fields and formats. Ensure valid email and ISO-8601 date."
        elif isinstance(error, DuplicateRequestError):
            tips = "This request appears to be a duplicate. Use a new requestId/correlationId."
        elif isinstance(error, CRMApiError):
            tips = "CRM API error. Try again later or contact support."
        elif isinstance(error, EntityResolutionError):
            tips = "Unable to resolve required CRM entities. Provide more details."
        elif isinstance(error, FallbackKnowledgeBaseError):
            tips = "Requested information not found in knowledge base."
        else:
            tips = "Check input and try again."
        self.audit_logger.log_action("error", {"error_type": error_type, "message": message, "context": context})
        return ErrorResponse(
            success=False,
            error_type=error_type,
            message=message,
            tips=tips,
            requestId=context.get("requestId"),
            correlationId=context.get("correlationId")
        )

# --- Main Agent Class ---

class CRMQuoteCreationAgent:
    """Main CRM Quote Creation Agent orchestrating all services."""
    def __init__(self):
        self.audit_logger = AuditLoggingService()
        self.idempotency_service = IdempotencyService(self.audit_logger)
        self.input_processor = InputProcessor(self.audit_logger)
        self.validation_service = ValidationService(self.audit_logger)
        self.business_rule_engine = BusinessRuleEngine(self.audit_logger)
        self.data_resolution_service = DataResolutionService(self.audit_logger)
        self.api_integration_service = APIIntegrationService(self.audit_logger)
        self.response_generator = ResponseGenerator(self.audit_logger)
        self.error_handler = ErrorHandler(self.audit_logger)
        self.embedding_service = AzureOpenAIEmbedding()
        self.retriever = AzureAISearchRetriever(self.embedding_service)
        self.llm_client = None  # Lazy init

    def _get_llm_client(self):
        if self.llm_client is None:
            self.llm_client = openai.AsyncAzureOpenAI(
                api_key=Config.get_azure_openai_api_key(),
                api_version="2024-02-01",
                azure_endpoint=Config.get_azure_openai_endpoint(),
            )
        return self.llm_client

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main agent orchestration entrypoint."""
        # --- Input Processing ---
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Parse and normalize incoming request",
            output_fn=lambda r: f"keys={list(r.keys())}"
        ) as step:
            try:
                input_data = self.input_processor.process_input(request_data)
                step.capture(input_data)
            except Exception as e:
                raise InputValidationError(str(e))

        # --- Idempotency Check ---
        async with trace_step(
            "idempotency_check", step_type="process",
            decision_summary="Check for duplicate request",
            output_fn=lambda r: f"duplicate={r}"
        ) as step:
            request_id = input_data.get("requestId") or f"REQ-{uuid.uuid4().hex[:8]}"
            correlation_id = input_data.get("correlationId") or f"CORR-{uuid.uuid4().hex[:8]}"
            is_duplicate = self.idempotency_service.check_duplicate(request_id, correlation_id)
            step.capture(is_duplicate)
            if is_duplicate:
                raise DuplicateRequestError("Duplicate request detected")

        # --- Validation ---
        async with trace_step(
            "validate_fields", step_type="process",
            decision_summary="Validate required fields and formats",
            output_fn=lambda r: f"valid={r.valid}, errors={r.errors}"
        ) as step:
            validation_result = self.validation_service.validate_fields(input_data)
            step.capture(validation_result)
            if not validation_result.valid:
                return self.response_generator.generate_response({
                    "status": "FAIL",
                    "errors": validation_result.errors,
                    "missingInputs": validation_result.missing_inputs,
                    "requestId": request_id,
                    "correlationId": correlation_id,
                    "validationSummary": validation_result.validation_summary
                })

        # --- Business Rules ---
        async with trace_step(
            "apply_business_rules", step_type="process",
            decision_summary="Apply business rules and eligibility checks",
            output_fn=lambda r: f"passed={r.passed}, errors={r.errors}"
        ) as step:
            rule_result = self.business_rule_engine.apply_rules(input_data)
            step.capture(rule_result)
            if not rule_result.passed:
                return self.response_generator.generate_response({
                    "status": "FAIL",
                    "errors": rule_result.errors,
                    "missingInputs": rule_result.missing_inputs,
                    "requestId": request_id,
                    "correlationId": correlation_id,
                    "validationSummary": rule_result.summary
                })

        # --- Data Resolution ---
        async with trace_step(
            "resolve_entities", step_type="process",
            decision_summary="Resolve CRM entities and required fields",
            output_fn=lambda r: f"missing_fields={r.missing_fields}"
        ) as step:
            resolved = self.data_resolution_service.resolve_entities(input_data)
            step.capture(resolved)
            if resolved.missing_fields:
                return self.response_generator.generate_response({
                    "status": "FAIL",
                    "errors": [f"Missing required fields: {', '.join(resolved.missing_fields)}"],
                    "missingInputs": resolved.missing_fields,
                    "requestId": request_id,
                    "correlationId": correlation_id,
                    "validationSummary": "Missing required CRM fields"
                })

        # --- RAG Retrieval ---
        async with trace_step(
            "retrieve_context", step_type="tool_call",
            decision_summary="Retrieve document context from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r)}"
        ) as step:
            user_query = f"Create CRM quote for {input_data.get('customerEmailId')}"
            context_chunks = self.retriever.retrieve_context(user_query)
            step.capture(context_chunks)

        # --- LLM Orchestration ---
        async with trace_step(
            "generate_response_llm", step_type="llm_call",
            decision_summary="Call LLM to generate structured response",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            system_prompt = (
                "You are a professional CRM Quote Creation Agent. Your role is to automate the creation of quotes and quote orders in a CRM system, "
                "following strict business rules and validations. For each request: - Validate that customerEmailId is present and in a valid email format. "
                "- Validate that receivedDateTime is present and parseable as ISO-8601. - Resolve the customer in CRM using the provided email. "
                "If not found or ineligible, return a clear error. - Resolve all required fields (Contact, Account, bill-to/ship-to, currency, price list, "
                "payment terms, tax region). If any are missing, request only the minimum additional information needed. - Ensure at least one valid line item is present. "
                "If not, prompt for product/SKU, quantity, requested dates, and shipping requirements. - Do not guess or autofill any required values. "
                "- Interact with CRM APIs to create the quote and quote order, enforcing idempotency and capturing all relevant identifiers and statuses. "
                "- Return a structured response: On success, provide quoteId, quoteOrderId, status, timestamps, and any warnings. On failure, provide a concise list of missing/invalid inputs, "
                "lookup failures, API errors, and the minimal missing inputs required to continue. - If information is not found in the knowledge base, respond with a clear fallback message."
            )
            # Compose context for LLM
            context_str = "\n".join(context_chunks) if context_chunks else ""
            user_message = (
                f"Input: {input_data}\n"
                f"CRM Entities: {resolved.model_dump()}\n"
                f"Document Context: {context_str}\n"
                "Respond in JSON with the following structure: "
                "{ \"status\": \"PASS\" | \"FAIL\", \"quoteId\": \"<string, if PASS>\", \"quoteOrderId\": \"<string, if PASS>\", "
                "\"timestamps\": \"<object, if PASS>\", \"warnings\": \"<array, optional>\", \"errors\": \"<array, if FAIL>\", "
                "\"missingInputs\": \"<array, if FAIL>\", \"requestId\": \"<string>\", \"correlationId\": \"<string>\", \"validationSummary\": \"<string>\" }"
            )
            llm_client = self._get_llm_client()
            _t0 = _time.time()
            try:
                response = await llm_client.chat.completions.create(
                    model=Config.get_azure_openai_llm_deployment(),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=Config.get_azure_openai_llm_deployment(),
                        prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                        completion_tokens=getattr(response.usage, "completion_tokens", None),
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                step.capture(content)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise

        # --- Parse LLM Output ---
        async with trace_step(
            "parse_llm_output", step_type="parse",
            decision_summary="Parse LLM JSON output",
            output_fn=lambda r: f"keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                import json
                result = json.loads(content)
                # Add requestId/correlationId if missing
                if not result.get("requestId"):
                    result["requestId"] = request_id
                if not result.get("correlationId"):
                    result["correlationId"] = correlation_id
                step.capture(result)
            except Exception as e:
                logger.error(f"Failed to parse LLM output: {e}")
                # Fallback response
                result = {
                    "status": "FAIL",
                    "errors": ["Failed to parse LLM output"],
                    "missingInputs": [],
                    "requestId": request_id,
                    "correlationId": correlation_id,
                    "validationSummary": "Failed to parse LLM output"
                }
                step.capture(result)

        # --- Final Response ---
        async with trace_step(
            "format_response", step_type="final",
            decision_summary="Format structured JSON response",
            output_fn=lambda r: f"status={r.get('status')}"
        ) as step:
            response = self.response_generator.generate_response(result)
            step.capture(response)
            return response

# --- FastAPI App ---

app = FastAPI(
    title="CRM Quote Creation Agent",
    description="Automates CRM quote and quote order creation with RAG and LLM orchestration.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = CRMQuoteCreationAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "ValidationError",
            "message": str(exc),
            "tips": "Check required fields, types, and formats. Ensure valid JSON.",
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTPException",
            "message": exc.detail,
            "tips": "Check your request and try again.",
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "tips": "Internal server error. Contact support if this persists.",
        }
    )

@app.post("/crm/quote", response_model=Dict[str, Any])
@with_content_safety(config=GUARDRAILS_CONFIG)
async def create_crm_quote(request: Request):
    try:
        # Parse JSON body
        try:
            data = await request.json()
        except Exception as e:
            logger.warning(f"Malformed JSON: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "MalformedJSON",
                    "message": "Malformed JSON in request body.",
                    "tips": "Ensure your JSON is valid (check quotes, commas, brackets)."
                }
            )
        # Input validation (size, content)
        if not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "InvalidInput",
                    "message": "Request body must be a JSON object.",
                    "tips": "Send a JSON object with required fields."
                }
            )
        if not data or len(str(data)) > 50000:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "InputTooLarge",
                    "message": "Input is empty or exceeds 50,000 characters.",
                    "tips": "Reduce input size and try again."
                }
            )
        # Clean input
        cleaned = {k: v for k, v in data.items() if isinstance(k, str)}
        # Orchestrate agent
        result = await agent.handle_request(cleaned)
        return JSONResponse(status_code=200, content={"success": True, "data": result})
    except Exception as e:
        error_resp = agent.error_handler.handle_error(e, data if 'data' in locals() else {})
        return JSONResponse(
            status_code=400 if isinstance(e, (InputValidationError, DuplicateRequestError)) else 500,
            content=error_resp.model_dump()
        )

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# --- Main Entrypoint ---



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting CRM Quote Creation Agent on http://0.0.0.0:8000")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())