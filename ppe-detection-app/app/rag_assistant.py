import sqlite3
import os
import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from datetime import datetime
from app.config import DB_PATH
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("ppe_violations")

def load_db_to_vectorstore():
    """Load violation data from SQLite into ChromaDB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all violation data
    cursor.execute('''
        SELECT v.id, v.person_id, v.violation_type, v.timestamp,
                   v.confidence, p.email_count
        FROM violations v 
        JOIN persons p ON v.person_id = p.id
        ORDER BY v.timestamp DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No violations found in DB yet — run detection first.")
        return

    # Convert to text documents
    documents = []
    ids = []

    for i, row in enumerate(rows):
        vid = row[0]
        person_id = row[1]
        violation_type = row[2]
        timestamp = row[3]
        confidence = row[4]
        email_count = row[5]

        # Create natural language description
        doc = f"""
        Violation ID: {vid}
        Person ID: {person_id}
        Missing PPE: {violation_type}
        Detected at: {timestamp}
        Confidence: {confidence:.2f}
        Total alerts for this person: {email_count}
        """
        documents.append(doc.strip())
        ids.append(f"violation_{vid}")

    if documents:
        # Add to ChromaDB
        collection.add(
            documents=documents,
            ids=ids
        )
        print(f"Loaded {len(documents)} records into vector store")

def query_ppe_data(question):
    """Query PPE violation data using RAG"""

    #1. Find relevent records 
    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    #2. Build context
    context = "\n\n".join(results["documents"][0])

    #3. Ask
    response = client.messages.create(
        model = "claude-haiku-4-5",
        max_tokens=500,
        system="""You are a PPE safety compliance assistant. 
        Answer questions about PPE violations using only the provided data. 
        Be concise and factual.""", 
        messages=[{
            "role": "user", 
            "content": f"""
            Question: {question}

            Relevant violation data: {context}
            Answer based only on the data provided.
            """
        }]
    )

    return response.content[0].text

# Main 
if __name__=="__main__":
    print("Loading violation data...")
    load_db_to_vectorstore()

    print("\nPPE Safety Assistant Ready!")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Ask a question: ")
        if question.lower() == "quit":
            break

        answer = query_ppe_data(question)
        print(f"\nAnswer: {answer}\n")
