"""
Backend pour la gestion de l'historique des conversations
Utilise PostgreSQL pour la persistance des données
"""

import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration de la base de données
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/chatbot_db")

class Message(BaseModel):
    id: str
    content: str
    is_bot: bool
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class Conversation(BaseModel):
    id: str
    user_id: str
    title: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime

class ConversationCreate(BaseModel):
    user_id: str
    title: str
    initial_message: Optional[Message] = None

class MessageCreate(BaseModel):
    conversation_id: str
    content: str
    is_bot: bool
    metadata: Optional[Dict[str, Any]] = None

class ChatHistoryManager:
    def __init__(self):
        self.connection = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connexion à la base de données PostgreSQL"""
        try:
            self.connection = psycopg2.connect(DATABASE_URL)
            print("✅ Connexion à PostgreSQL réussie")
        except Exception as e:
            print(f"❌ Erreur de connexion à PostgreSQL: {e}")
            raise
    
    def create_tables(self):
        """Création des tables si elles n'existent pas"""
        with self.connection.cursor() as cursor:
            # Table des conversations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX(user_id),
                    INDEX(created_at)
                )
            """)
            
            # Table des messages
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY,
                    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    is_bot BOOLEAN NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    INDEX(conversation_id),
                    INDEX(timestamp)
                )
            """)
            
            self.connection.commit()
            print("✅ Tables créées avec succès")
    
    def create_conversation(self, user_id: str, title: str, initial_message: Optional[Message] = None) -> str:
        """Créer une nouvelle conversation"""
        conversation_id = str(uuid.uuid4())
        
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (id, user_id, title)
                VALUES (%s, %s, %s)
            """, (conversation_id, user_id, title))
            
            # Ajouter le message initial si fourni
            if initial_message:
                self.add_message(conversation_id, initial_message.content, 
                               initial_message.is_bot, initial_message.metadata)
            
            self.connection.commit()
        
        return conversation_id
    
    def add_message(self, conversation_id: str, content: str, is_bot: bool, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Ajouter un message à une conversation"""
        message_id = str(uuid.uuid4())
        
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, content, is_bot, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (message_id, conversation_id, content, is_bot, 
                  json.dumps(metadata) if metadata else None))
            
            # Mettre à jour le timestamp de la conversation
            cursor.execute("""
                UPDATE conversations 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE id = %s
            """, (conversation_id,))
            
            self.connection.commit()
        
        return message_id
    
    def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupérer les conversations d'un utilisateur"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT c.*, 
                       COUNT(m.id) as message_count,
                       MAX(m.timestamp) as last_message_time
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_id = %s
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT %s
            """, (user_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_conversation_with_messages(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Récupérer une conversation avec tous ses messages"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Récupérer la conversation
            cursor.execute("""
                SELECT * FROM conversations WHERE id = %s
            """, (conversation_id,))
            
            conversation = cursor.fetchone()
            if not conversation:
                return None
            
            # Récupérer les messages
            cursor.execute("""
                SELECT * FROM messages 
                WHERE conversation_id = %s 
                ORDER BY timestamp ASC
            """, (conversation_id,))
            
            messages = [dict(row) for row in cursor.fetchall()]
            
            result = dict(conversation)
            result['messages'] = messages
            return result
    
    def search_conversations(self, user_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Rechercher dans les conversations et messages"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT DISTINCT c.*, 
                       COUNT(m.id) as message_count,
                       MAX(m.timestamp) as last_message_time
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_id = %s 
                AND (c.title ILIKE %s OR m.content ILIKE %s)
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT %s
            """, (user_id, f"%{query}%", f"%{query}%", limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Supprimer une conversation (avec vérification utilisateur)"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                DELETE FROM conversations 
                WHERE id = %s AND user_id = %s
            """, (conversation_id, user_id))
            
            deleted = cursor.rowcount > 0
            self.connection.commit()
            return deleted
    
    def update_conversation_title(self, conversation_id: str, user_id: str, new_title: str) -> bool:
        """Mettre à jour le titre d'une conversation"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE conversations 
                SET title = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND user_id = %s
            """, (new_title, conversation_id, user_id))
            
            updated = cursor.rowcount > 0
            self.connection.commit()
            return updated

# API FastAPI
app = FastAPI(title="Chat History API")
history_manager = ChatHistoryManager()

@app.post("/conversations/")
async def create_conversation(conversation: ConversationCreate):
    """Créer une nouvelle conversation"""
    try:
        conversation_id = history_manager.create_conversation(
            conversation.user_id, 
            conversation.title, 
            conversation.initial_message
        )
        return {"conversation_id": conversation_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/messages/")
async def add_message(message: MessageCreate):
    """Ajouter un message à une conversation"""
    try:
        message_id = history_manager.add_message(
            message.conversation_id,
            message.content,
            message.is_bot,
            message.metadata
        )
        return {"message_id": message_id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str, limit: int = 50):
    """Récupérer les conversations d'un utilisateur"""
    try:
        conversations = history_manager.get_user_conversations(user_id, limit)
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Récupérer une conversation avec ses messages"""
    try:
        conversation = history_manager.get_conversation_with_messages(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
