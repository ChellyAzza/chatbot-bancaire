"""
API Backend pour connecter le frontend avec votre RAG
FastAPI + CORS pour communication avec React
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import asyncio
from typing import Optional, List, Dict
from datetime import datetime
import hashlib
import uuid

# Configuration
model_path = "./models/Llama-3.1-8B-Instruct"
adapter_path = "./llama_banking_final_fidelity"

app = FastAPI(title="Banking RAG API", version="1.0.0")

# CORS pour React
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",  # Votre frontend
        "http://localhost:5173",  # Vite par défaut
        "http://localhost:3000"   # React par défaut
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    response_time: float
    contexts_found: int
    similarity_score: float
    conversation_id: str

# Nouveaux modèles pour l'historique
class HistorySearchRequest(BaseModel):
    question: str
    threshold: float = 0.85
    max_results: int = 1

class HistorySearchResponse(BaseModel):
    found: bool
    response: Optional[str] = None
    confidence: Optional[float] = None
    original_question: Optional[str] = None
    timestamp: Optional[str] = None

class HistorySaveRequest(BaseModel):
    question: str
    response: str
    confidence: float = 0.95
    response_time: float = 0
    timestamp: str
    context: Optional[dict] = None
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    question_id: str
    feedback: str  # 'positive', 'negative', 'neutral'
    comment: Optional[str] = None
    timestamp: str

# Système RAG global
rag_system = None

class BankingRAGAPI:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.vectorizer = None
        self.question_vectors = None
        self.qa_pairs = []
        self.is_loaded = False
        
    async def load_model(self):
        """Charge le modèle de manière asynchrone"""
        if self.is_loaded:
            return

        print("🚀 Chargement du modèle RAG...")
        print("⏱️ Cela peut prendre 5-15 minutes...")
        print("🧹 Nettoyage mémoire GPU...")
        torch.cuda.empty_cache()
        
        # Tokenizer
        from pathlib import Path
        if Path(model_path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                token="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"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Modèle avec configuration GPU optimisée
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            if Path(model_path).exists():
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    max_memory={0: "7GB", "cpu": "12GB"}
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    max_memory={0: "7GB", "cpu": "12GB"},
                    token="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"
                )
            
        except Exception as e:
            print(f"⚠️ Fallback sans quantization: {e}")
            if Path(model_path).exists():
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": "cuda:0"},
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    device_map={"": "cuda:0"},
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    token="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"
                )
        
        # Charger l'adaptateur LoRA
        self.model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            torch_dtype=torch.float16
        )
        
        # Charger la base de connaissances
        await self.load_knowledge_base()
        
        self.is_loaded = True
        print("✅ Modèle RAG chargé avec succès!")
    
    async def load_knowledge_base(self):
        """Charge la base de connaissances"""
        clean_file = "cleaned_banking_qa.json"
        
        try:
            with open(clean_file, 'r', encoding='utf-8') as f:
                self.qa_pairs = json.load(f)
        except FileNotFoundError:
            from datasets import load_dataset
            dataset = load_dataset("wasifis/bank-assistant-qa")
            train_data = dataset["train"]
            
            self.qa_pairs = []
            for item in train_data:
                cleaned_item = {
                    "input": item["input"].strip(),
                    "output": item["output"].strip()
                }
                if cleaned_item["input"] and cleaned_item["output"]:
                    self.qa_pairs.append(cleaned_item)
            
            with open(clean_file, 'w', encoding='utf-8') as f:
                json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        
        # Index TF-IDF
        questions = [pair["input"] for pair in self.qa_pairs]
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.question_vectors = self.vectorizer.fit_transform(questions)
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Recherche contextes pertinents"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                relevant_contexts.append({
                    "question": self.qa_pairs[idx]["input"],
                    "answer": self.qa_pairs[idx]["output"],
                    "similarity": similarities[idx]
                })
        
        return relevant_contexts
    
    async def generate_response(self, user_question: str):
        """Génère une réponse avec votre RAG"""
        if not self.is_loaded:
            await self.load_model()
        
        start_time = time.time()
        
        # Rechercher contextes
        relevant_contexts = self.retrieve_relevant_context(user_question, top_k=3)
        
        # Construire contexte
        context_info = ""
        if relevant_contexts:
            context_info = "\n\nRelevant banking information:\n"
            for i, ctx in enumerate(relevant_contexts):
                context_info += f"Source {i+1}: {ctx['answer']}\n"
        
        # Prompt développé
        developed_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert banking consultant. Provide comprehensive, detailed, and well-structured answers to banking questions. Always give complete information with specific details.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_question}{context_info}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Based on the banking information provided, here's a comprehensive answer:

"""
        
        # Génération
        inputs = self.tokenizer(developed_prompt, return_tensors="pt", truncation=True, max_length=1500)
        
        try:
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except:
            pass
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Extraction
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Based on the banking information provided, here's a comprehensive answer:" in full_response:
            response = full_response.split("Based on the banking information provided, here's a comprehensive answer:")[-1]
        elif "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            response = full_response
        
        response = response.replace("<|eot_id|>", "").strip()
        
        # Si réponse trop courte, développer
        if len(response.split()) < 20 and relevant_contexts:
            response = self.develop_response_manually(user_question, relevant_contexts[0])
        
        response_time = time.time() - start_time
        max_similarity = max([ctx['similarity'] for ctx in relevant_contexts]) if relevant_contexts else 0.0
        
        return response, response_time, len(relevant_contexts), max_similarity
    
    def develop_response_manually(self, question, context):
        """Développe manuellement une réponse"""
        answer = context['answer']
        
        if "limit" in question.lower():
            return f"Regarding your question about limits:\n\n{answer}\n\nThese limits are designed to ensure secure banking operations while providing flexibility for your financial needs."
        elif "charge" in question.lower() or "fee" in question.lower():
            return f"Here are the detailed charges and fees:\n\n{answer}\n\nThese charges are structured to provide transparent pricing for banking services."
        else:
            return f"Based on the banking information available:\n\n{answer}\n\nFor more specific information, please contact customer service."

# Initialiser le système RAG
rag_system = BankingRAGAPI()

# Base de données en mémoire pour l'historique (pour les tests)
history_db: List[Dict] = []
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

def normalize_question(question: str) -> str:
    """Normalise une question pour la comparaison"""
    return question.lower().strip().replace('?', '').replace('.', '')

def calculate_similarity(question1: str, question2: str) -> float:
    """Calcule la similarité entre deux questions"""
    try:
        questions = [normalize_question(question1), normalize_question(question2)]
        tfidf_matrix = vectorizer.fit_transform(questions)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def find_similar_in_history(question: str, threshold: float = 0.85) -> Optional[Dict]:
    """Trouve une question similaire dans l'historique"""
    if not history_db:
        return None

    best_match = None
    best_score = 0.0

    for entry in history_db:
        similarity = calculate_similarity(question, entry['question'])
        if similarity > threshold and similarity > best_score:
            best_score = similarity
            best_match = entry.copy()
            best_match['confidence'] = similarity

    return best_match

def save_to_history(question: str, response: str, confidence: float = 0.95,
                   response_time: float = 0, session_id: str = None) -> str:
    """Sauvegarde une interaction dans l'historique"""
    entry_id = str(uuid.uuid4())
    entry = {
        'id': entry_id,
        'question': question,
        'response': response,
        'confidence': confidence,
        'response_time': response_time,
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id or 'unknown',
        'hits': 1
    }
    history_db.append(entry)

    # Limiter à 1000 entrées pour éviter la surcharge mémoire
    if len(history_db) > 1000:
        history_db.pop(0)

    return entry_id

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage"""
    print("🚀 Démarrage de l'API Banking RAG...")
    # Le modèle sera chargé lors de la première requête pour éviter le timeout

@app.get("/")
async def root():
    return {"message": "Banking RAG API is running!", "status": "active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": rag_system.is_loaded,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint principal pour le chat"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Générer réponse avec votre RAG
        response, response_time, contexts_found, similarity_score = await rag_system.generate_response(request.message)
        
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"

        # Sauvegarder automatiquement dans l'historique
        save_to_history(request.message, response, similarity_score, response_time)

        return ChatResponse(
            response=response,
            response_time=response_time,
            contexts_found=contexts_found,
            similarity_score=similarity_score,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Nouveaux endpoints pour l'historique
@app.post("/api/history/search", response_model=HistorySearchResponse)
async def search_history(request: HistorySearchRequest):
    """Recherche dans l'historique des questions similaires"""
    try:
        similar_entry = find_similar_in_history(request.question, request.threshold)

        if similar_entry:
            # Incrémenter le compteur de hits
            similar_entry['hits'] += 1

            return HistorySearchResponse(
                found=True,
                response=similar_entry['response'],
                confidence=similar_entry['confidence'],
                original_question=similar_entry['question'],
                timestamp=similar_entry['timestamp']
            )
        else:
            return HistorySearchResponse(found=False)

    except Exception as e:
        print(f"❌ Erreur dans search_history: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching history: {str(e)}")

@app.post("/api/history/save")
async def save_history(request: HistorySaveRequest):
    """Sauvegarde une nouvelle interaction dans l'historique"""
    try:
        entry_id = save_to_history(
            request.question,
            request.response,
            request.confidence,
            request.response_time,
            request.session_id
        )
        return {"success": True, "entry_id": entry_id}

    except Exception as e:
        print(f"❌ Erreur dans save_history: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving to history: {str(e)}")

@app.get("/api/history/stats")
async def get_history_stats():
    """Récupère les statistiques de l'historique"""
    try:
        if not history_db:
            return {
                "total_questions": 0,
                "cache_hit_rate": 0,
                "average_confidence": 0,
                "most_asked_topics": []
            }

        total_questions = len(history_db)
        total_hits = sum(entry.get('hits', 1) for entry in history_db)
        average_confidence = sum(entry['confidence'] for entry in history_db) / total_questions

        # Calculer le taux de cache hit (approximatif)
        cache_hit_rate = (total_hits - total_questions) / total_hits * 100 if total_hits > 0 else 0

        return {
            "total_questions": total_questions,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "average_confidence": round(average_confidence, 3),
            "most_asked_topics": ["transfers", "accounts", "loans", "fees"]
        }

    except Exception as e:
        print(f"❌ Erreur dans get_history_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting history stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Lancement de l'API Banking RAG...")
    print("📡 Frontend: http://localhost:5173")
    print("🔗 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
