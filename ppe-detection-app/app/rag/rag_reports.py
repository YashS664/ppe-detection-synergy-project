import logging
import sqlite3
import os
from datetime import datetime
from app.rag.rag_agent import agent
from app.rag.rag_models import ViolationsSummary, WorkerRiskProfile
from fastapi import HTTPException

from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "database", "embeddings.db"
)

def person_exists(person_id: int) -> bool:
    """Check if person exists in DB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (person_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def generate_safety_report() -> ViolationsSummary:
    """Generate structured daily safety report"""
    try:
        logger.info("Generating safety report...")
        result = agent.invoke({
            "input": """Get me these numbers from violations table:
            1. Total violations count
            2. Hardhat violations count 
            3. Vest violations count 
            4. Total unique persons
            5. Person ID with most violations
            Return as JSON"""
        })

        output = result["output"]
        if isinstance(output, list):
            output = " ".join([
                item["text"] for item in output
                if isinstance(item, dict) and "text" in item
            ])

        structured_llm = llm.with_structured_output(ViolationsSummary)
        report = structured_llm.invoke(
            f"""Generate safety compliance report from:
            {output}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Severity: 0-2=LOW, 3-5=MEDIUM, 6-10=HIGH, 10+=CRITICAL"""
        )
        logger.info(f"Report generated: {report.severity}")
        return report
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

def get_worker_risk_profile(person_id: int) ->WorkerRiskProfile:
    """Get risk profile for specific worker"""
    try:
        if not person_exists(person_id):
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found in database")

        logger.info(f"Getting risk profile for Person {person_id}")
        result = agent.invoke({
            "input": f"""For person_id {person_id} in violations table:
            Get total, hardhat, vest violations count"""
        })

        output = result["output"]
        if isinstance(output, list):
            output = " ".join([
                item["text"] for item in output
                if isinstance(item, dict) and "text" in item
            ])

        structured_llm = llm.with_structured_output(WorkerRiskProfile)
        profile = structured_llm.invoke(
            f"""Create risk profile for Person {person_id}:
            {output}
            Risk: LOW=1-2, MEDIUM=3-5, HIGH=6+"""
        )
        return profile
    
    except Exception as e:
        logger.error(f"Risk profile failed for Person {person_id}: {e}")
        raise