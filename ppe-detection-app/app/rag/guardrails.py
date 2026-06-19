import re
import logging

logger = logging.getLogger(__name__)

# Patterns that suggest prompt injection / jailbreak attempts
SUSPICIOUS_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"system prompt",
    r"forget (everything|your instructions)",
    r"act as if",
    r"new instructions",
    r"<\s*system\s*>",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"UPDATE\s+.*SET",
    r"INSERT\s+INTO",
    r"ALTER\s+TABLE",
    r"ATTACH\s+DATABASE"
    r"PRAGMA"
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]

class GuardrailViolation(Exception):
    """Raised when input/output violates safety guardrails"""
    pass

def check_input_safety(question: str) -> str:
    """
    Validate user question before sending to agent.
    Raises GuardrailViolation if suspicious.
    """
    for pattern in COMPILED_PATTERNS:
        if pattern.search(question):
            logger.warning(f"GUARDRAIL BLOCKED: '{question}' matched pattern '{pattern.pattern}'")
            raise GuardrailViolation(
                "Your question contains content that cannot be processed. "
                "Please ask a factual question about PPE violations."
            )
        
    return question

def check_sql_safety(sql_query: str) -> str:
    """
    Validate generated SQL before execution.
    Only SELECT statements allowed.
    """
    normalized = sql_query.strip().upper()

    # Must start with SELECT
    if not normalized.startswith("SELECT"):
        logger.warning(f"GUARDRAIL BLOCKED SQL: {sql_query}")
        raise GuardrailViolation("Only SELECT queries are permitted.")
    
    # Block dangerous keywords anywhere in query
    dangerous_keywords = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", 
        "ATTACH", "DETECH", "PRAGMA", "CREATE", "REPLACE"
    ]
    for kw in dangerous_keywords:
        if kw in normalized:
            logger.warning(f"GUARDRAIL BLOCKED SQL (contains {kw}): {sql_query}")
            raise GuardrailViolation("Query contains forbidden keyword: {kw}")
    
    return sql_query