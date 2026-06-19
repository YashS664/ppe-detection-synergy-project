import sqlite3
import json
import numpy as np
import os
from datetime import datetime
from app.config import DB_PATH, REID_THRESHOLD

class PersonDatabase:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        # 1. Person Table: Persistent identity and aggregate stats
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                email_count INTEGER DEFAULT 0,
                last_email_sent TIMESTAMP
            )
        ''')
        
        # 2. Notification Details: Individual sightings and email events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_details (
                person_id INTEGER,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                email_sent TIMESTAMP,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
        ''')

        # 3. Violations Table: Stores individual PPE violation events per person.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                violation_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
        ''')
        self.conn.commit()

    def save_person(self, embedding, metadata=None):
        """
        Saves a new person embedding and creates an initial sighting record.
        embedding: numpy array or list
        """
        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            embedding_bytes = embedding.tobytes()
            
            cursor = self.conn.cursor()
            # Insert into persons table
            cursor.execute('''
                INSERT INTO persons (embedding)
                VALUES (?)
            ''', (embedding_bytes,))
            person_id = cursor.lastrowid
            
            # Create initial sighting in notification_details
            cursor.execute('''
                INSERT INTO notification_details (person_id, first_seen, last_seen)
                VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (person_id,))
            
            self.conn.commit()
            print(f"DATABASE: Successfully saved person with Global ID {person_id}")
            return person_id
        except Exception as e:
            print(f"DATABASE ERROR in save_person: {e}")
            self.conn.rollback()
            return None
    
    def save_violation(self, person_id, violation_type, confidence):
        """Save individual PPE violation event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO violations(person_id, violation_type, confidence)
                VALUES (?, ?, ?)
            ''', (person_id, violation_type, confidence))
            self.conn.commit()
            print(f"DATABASE: Saved violation - Person{person_id} missing {violation_type}")
        except Exception as e:
            print(f"DATABASE ERROR in save_violation: {e}")
            self.conn.rollback()

    def update_last_seen(self, person_id):
        """Updates the last_seen timestamp in notification_details for the most recent entry of this person."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE notification_details 
            SET last_seen = CURRENT_TIMESTAMP 
            WHERE person_id = ? AND rowid = (
                SELECT MAX(rowid) FROM notification_details WHERE person_id = ?
            )
        ''', (person_id, person_id))
        self.conn.commit()

    def update_email_alert_status(self, person_id):
        """Updates aggregate stats in persons and logs a NEW email event in notification_details."""
        cursor = self.conn.cursor()
        # 1. Update aggregate in persons
        cursor.execute('''
            UPDATE persons 
            SET last_email_sent = CURRENT_TIMESTAMP, 
                email_count = email_count + 1 
            WHERE id = ?
        ''', (person_id,))
        
        # 2. Create a NEW record in notification_details for this email event
        # We inherit the latest seen timestamps to keep context for the notification
        cursor.execute('''
            INSERT INTO notification_details (person_id, first_seen, last_seen, email_sent)
            SELECT person_id, first_seen, last_seen, CURRENT_TIMESTAMP
            FROM notification_details
            WHERE person_id = ?
            ORDER BY rowid DESC
            LIMIT 1
        ''', (person_id,))
        
        # Fallback if somehow no record exists yet
        if cursor.rowcount == 0:
            cursor.execute('''
                INSERT INTO notification_details (person_id, email_sent)
                VALUES (?, CURRENT_TIMESTAMP)
            ''', (person_id,))
            
        self.conn.commit()

    def get_email_status(self, person_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT last_email_sent, email_count FROM persons WHERE id = ?', (person_id,))
        row = cursor.fetchone()
        if row:
            return {"last_sent": row[0], "count": row[1]}
        return None

    def get_all_persons(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, embedding FROM persons')
        rows = cursor.fetchall()
        
        persons = []
        for row in rows:
            persons.append({
                "id": row[0],
                "embedding": np.frombuffer(row[1], dtype=np.float32)
            })
        return persons

    def find_match(self, embedding, threshold=REID_THRESHOLD):
        """
        Finds the closest person in the DB.
        Returns (person_id, similarity) or (None, 0)
        """
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        all_persons = self.get_all_persons()
        if not all_persons:
            return None, 0
        
        best_match_id = None
        best_similarity = -1
        
        for person in all_persons:
            # Skip if shapes don't match (e.g. model change)
            if embedding.shape != person["embedding"].shape:
                continue
                
            sim = float(np.dot(embedding, person["embedding"]))
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = person["id"]
        
        if best_similarity >= threshold:
            return best_match_id, best_similarity
        return None, best_similarity

    def get_dashboard_data(self):
        """Retrieves unified dashboard view by joining persons and notification_details."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    p.id, 
                    p.email_count, 
                    p.last_email_sent,
                    n.first_seen, 
                    n.last_seen, 
                    n.email_sent
                FROM persons p
                LEFT JOIN notification_details n ON p.id = n.person_id
                ORDER BY p.email_count DESC, n.last_seen DESC
            ''')
            rows = cursor.fetchall()
            
            data = []
            for row in rows:
                data.append({
                    "person_id": row[0],
                    "total_emails": row[1],
                    "last_email_aggregate": row[2],
                    "first_seen": row[3],
                    "last_seen": row[4],
                    "email_sent_event": row[5]
                })
            return data
        except Exception as e:
            print(f"DATABASE ERROR in get_dashboard_data: {e}")
            return []


# Global instance
db = PersonDatabase()
