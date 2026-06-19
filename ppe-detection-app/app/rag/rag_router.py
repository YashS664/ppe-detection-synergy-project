import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from app.rag.rag_agent import agent
from app.rag.rag_models import QueryRequest, QueryResponse, ViolationsSummary, WorkerRiskProfile
from app.rag.rag_reports import generate_safety_report, get_worker_risk_profile
from app.rag.guardrails import check_input_safety, GuardrailViolation
from app.rag.rag_audit import log_query, AUDIT_DB_PATH, init_audit_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/query", response_model=QueryResponse)
async def query_violations(request: QueryRequest):
    """Natural language query on violation data"""
    try:
        check_input_safety(request.question)
    except GuardrailViolation as e:
        log_query(request.question, blocked= True, block_reason=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = agent.invoke({"input": request.question})
        output = result["output"]
        if isinstance(output, list):
            answer = " ".join([
                item["text"] for item in output
                if isinstance(item, dict) and "text" in item
            ])
        else:
            answer = str(output)

        log_query(request.question, answer=answer, blocked=False)
        
        return QueryResponse(
            question=request.question,
            answer=answer
        )
    
    except Exception as e:
        log_query(request.question, blocked=True, block_reason=str(e))
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to process query")

@router.get("/report", response_model=ViolationsSummary)
async def get_daily_report():
    """Generate structured daily safety report"""
    try:
        return generate_safety_report()
    except Exception as e:
        logger.error(f"Report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/worker/{person_id}", response_model=WorkerRiskProfile)
async def get_worker_profile(person_id: int):
    """Get risk profile for specific worker"""
    try:
        return get_worker_risk_profile(person_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Worker profile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluate")
async def run_evaluation():
    """Run RAGAS evaluation on RAG quality"""
    try:
        from app.rag.rag_evaluation import evaluate_rag_quality

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            scores = await loop.run_in_executor(pool, evaluate_rag_quality)

        if scores is not None:
            # scores_dict = dict(scores)
            print(f"DEBUG router score: {scores}")

            import math

            def extract_score(value):
                """Extract float from any format ragas returns"""
                if isinstance(value, list):
                    valid = [v for v in value if v is not None and not (isinstance(v, float) and math.isnan(v))]
                    return sum(valid) / len(valid) if valid else 0.0
                elif hasattr(value, 'iloc'):
                    return float(value.iloc[0])
                else:
                    val = float(value)
                    return 0.0 if math.isnan(val) else val

            faithfulness_score = extract_score(scores["faithfulness"])
            relevancy_score = extract_score(scores["answer_relevancy"])
            
            return {
                "faithfulness": round(faithfulness_score, 3),
                "answer_relevancy": round(relevancy_score, 3)
            }
        return {"error": "Evaluation endpoint failed"}
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/audit")
async def get_audit_log(limit: int = 20):
    """View recent query audit trail"""
    import sqlite3
    init_audit_db()
    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, question, answer, blocked, blocked_reason
        FROM audit_log ORDER BY timestamp DESC LIMIT ? 
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()

    return[{
        "timestamp": r[0], "question": r[1], "answer": r[2],
        "blocked": bool(r[3]), "block_reason": r[4]
    } for r in rows]