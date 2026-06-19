import os
from app.rag.rag_agent import agent
import logging
import asyncio

logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    "How many violations were detected?",
    "What is the most common violation type?",
    "Which person has the most violations?",
    "How many hardhat violations are there?",
    "How many vest violations are there?"
]

# GROUND_TRUTHS = [
#     "Total count of all violations",
#     "Either hardhat or vest",
#     "Person with highest violation count",
#     "Count of hardhat violations",
#     "Count of vest violations"
# ]

def _patch_vertexai():
    """Patch missing vertexai module — only affects ragas import"""
    import sys
    from unittest.mock import MagicMock
    if 'langchain_community.chat_models.vertexai' not in sys.modules:
        mock = MagicMock()
        mock.ChatVertexAI = type('ChatVertexAI', (), {})
        sys.modules['langchain_community.chat_models.vertexai'] = mock

def evaluate_rag_quality():
    """Evaluate RAG answer quality using RAGAS"""
    try:
        _patch_vertexai()
        from ragas import evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper  
        from langchain_anthropic import ChatAnthropic
        from langchain_huggingface import HuggingFaceEmbeddings
        from datasets import Dataset

        logger.info("Starting RAGAS evaluation...")
        answers = []
        contexts = []

        for q in TEST_QUESTIONS:
            result = agent.invoke({"input": q})
            output = result["output"]
            if isinstance(output, list):
                output = " ".join([
                    item["text"] for item in output
                    if isinstance(item, dict) and "text" in item
                ])
            answers.append(str(output))
            contexts.append([str(output)])

        eval_dataset = Dataset.from_dict({
            "question": TEST_QUESTIONS,
            "answer": answers,
            "contexts": contexts
        })

        llm = ChatAnthropic(model="claude-haiku-4-5")
        ragas_llm = LangchainLLMWrapper(llm)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        scores = evaluate(
                dataset=eval_dataset,
                metrics=[Faithfulness(), ResponseRelevancy()],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False
            )
        
        loop.close()
        
        print(f"DEBUG scores: {scores}")
        return scores

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None