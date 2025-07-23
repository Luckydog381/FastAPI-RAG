import psycopg2
from datetime import datetime

class PostgresDB:
    def __init__(self, dbname: str, user: str, password: str, host: str = "localhost", port: str = "5432"):
        """
        Initialize the connection to the PostgreSQL database and create the necessary tables.
        :dbname: Name of the database
        :user: Database user
        :password: Password for the database user
        :host: Host where the database is located (default is localhost)
        :port: Port number for the database connection (default is 5432)
        """
        self.conn = psycopg2.connect(
            database=dbname, user=user, password=password, host=host, port=port
        )
        self._init_tables()

    def _init_tables(self):
        """Initialize the tables for chat sessions and messages."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP DEFAULT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id INT REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    message TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_audit (
                    id SERIAL PRIMARY KEY,
                    chat_id INT REFERENCES chat_sessions(id) ON DELETE SET NULL,
                    question TEXT,
                    response TEXT,
                    retrieved_docs TEXT,
                    latency_ms INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feedback TEXT
                );
            """)

        self.conn.commit()

    def create_session(self):
        """Create a new chat session and return its ID."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_sessions (created_at)
                VALUES (%s)
                RETURNING id;
            """, (datetime.utcnow(),))
            session_id = cur.fetchone()[0]
        self.conn.commit()
        return session_id

    def get_active_sessions(self):
        """Retrieve all active chat sessions."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, created_at
                FROM chat_sessions
                WHERE deleted_at IS NULL
                ORDER BY created_at ASC;
            """)
            return [{"id": row[0], "created_at": row[1]} for row in cur.fetchall()]

    def add_message(self, session_id: int, message: str, sender: str):
        """Add a message to a chat session."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_messages (session_id, message, sender, created_at)
                VALUES (%s, %s, %s, %s);
            """, (session_id, message, sender, datetime.utcnow()))
        self.conn.commit()

    def get_messages(self, session_id: int):
        """Retrieve all messages for a specific chat session."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT m.id, m.message, m.sender, m.created_at
                FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.id
                WHERE m.session_id = %s AND s.deleted_at IS NULL
                ORDER BY m.created_at ASC;
            """, (session_id,))
            rows = cur.fetchall()
        return [{"id": r[0], "message": r[1], "sender": r[2], "created_at": r[3]} for r in rows]

    def add_audit(self, chat_id, question, response, retrieved_docs, latency_ms, feedback=None):
        """Add an audit record for a chat session.
        :chat_id: ID of the chat session
        :question: User's question
        :response: Gemini's response
        :retrieved_docs: Retrieved documents for context
        :latency_ms: Latency in milliseconds
        :feedback: Optional feedback from the user
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_audit (chat_id, question, response, retrieved_docs, latency_ms, timestamp, feedback)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                chat_id,
                question,
                response,
                retrieved_docs,
                latency_ms,
                datetime.utcnow(),
                feedback
            ))
        self.conn.commit()
    
    def delete_session(self, session_id: int):
        """Mark a chat session as deleted by setting the deleted_at timestamp."""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE chat_sessions
                SET deleted_at = %s
                WHERE id = %s;
            """, (datetime.utcnow(), session_id))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
