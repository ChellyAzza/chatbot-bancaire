"""
Système RAG complet utilisant la base wasifis/bank-assistant-qa
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# RAG dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import chromadb
from sentence_transformers import SentenceTransformer

# Ollama integration
import ollama

class WasifisRAGSystem:
    """Système RAG utilisant les données wasifis/bank-assistant-qa"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.ollama_client = ollama.Client()
        self.documents = []
        
    def load_wasifis_data(self):
        """Charge les données wasifis/bank-assistant-qa préparées"""
        print("=== Chargement des données wasifis/bank-assistant-qa ===")
        
        try:
            # Charger les données depuis processed_data
            data_files = {
                'train': 'processed_data/train.json',
                'validation': 'processed_data/validation.json',
                'test': 'processed_data/test.json'
            }
            
            all_data = []
            
            for split_name, file_path in data_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = [json.loads(line) for line in f]
                    
                    print(f"✅ {split_name}: {len(data)} exemples")
                    all_data.extend(data)
                else:
                    print(f"⚠️ Fichier non trouvé: {file_path}")
            
            print(f"📊 Total: {len(all_data)} exemples chargés")
            
            # Convertir en documents pour RAG
            documents = []
            for i, item in enumerate(all_data):
                # Créer un document structuré
                if 'input' in item and 'output' in item:
                    content = f"Question: {item['input']}\nRéponse: {item['output']}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "wasifis/bank-assistant-qa",
                            "doc_id": i,
                            "question": item['input'],
                            "answer": item['output'],
                            "type": "qa_pair"
                        }
                    )
                    documents.append(doc)
            
            self.documents = documents
            print(f"✅ {len(documents)} documents créés pour RAG")
            return True
            
        except Exception as e:
            print(f"❌ Erreur de chargement: {e}")
            return False
    
    def setup_embeddings(self):
        """Configure le système d'embeddings"""
        print("\n=== Configuration des embeddings ===")
        
        try:
            # Utiliser un modèle d'embedding français
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            print(f"Chargement du modèle: {model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Utiliser CPU pour la compatibilité
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("✅ Modèle d'embedding chargé")
            return True
            
        except Exception as e:
            print(f"❌ Erreur d'embedding: {e}")
            return False
    
    def create_vector_store(self):
        """Crée la base de données vectorielle"""
        print("\n=== Création de la base vectorielle ===")
        
        if not self.documents or not self.embeddings:
            print("❌ Documents ou embeddings non disponibles")
            return False
        
        try:
            # Créer le répertoire pour ChromaDB
            persist_directory = "./chroma_db_wasifis"
            
            # Diviser les documents en chunks plus petits si nécessaire
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            print("Division des documents en chunks...")
            split_docs = text_splitter.split_documents(self.documents)
            print(f"✅ {len(split_docs)} chunks créés")
            
            # Créer la base vectorielle avec ChromaDB
            print("Création de la base vectorielle (cela peut prendre du temps)...")
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            
            print(f"✅ Base vectorielle créée: {persist_directory}")
            print(f"📊 {len(split_docs)} documents indexés")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur de création vectorielle: {e}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """Recherche les documents similaires à la requête"""
        if not self.vectorstore:
            print("❌ Base vectorielle non initialisée")
            return []
        
        try:
            # Recherche de similarité
            similar_docs = self.vectorstore.similarity_search(
                query=query,
                k=k
            )
            
            return similar_docs
            
        except Exception as e:
            print(f"❌ Erreur de recherche: {e}")
            return []
    
    def generate_rag_response(self, user_question: str) -> Dict:
        """Génère une réponse RAG complète"""
        print(f"\n🔍 Question: {user_question}")
        
        # 1. Rechercher des documents pertinents
        print("Recherche de documents pertinents...")
        relevant_docs = self.search_similar_documents(user_question, k=3)
        
        if not relevant_docs:
            print("❌ Aucun document pertinent trouvé")
            return {
                "answer": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
                "sources": [],
                "confidence": 0.0
            }
        
        print(f"✅ {len(relevant_docs)} documents trouvés")
        
        # 2. Construire le contexte
        context_parts = []
        sources = []
        
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"Document {i+1}:\n{doc.page_content}")
            sources.append({
                "doc_id": doc.metadata.get("doc_id", i),
                "question": doc.metadata.get("question", ""),
                "answer": doc.metadata.get("answer", ""),
                "content": doc.page_content[:200] + "..."
            })
        
        context = "\n\n".join(context_parts)
        
        # 3. Créer le prompt RAG
        rag_prompt = f"""Vous êtes un assistant bancaire expert. Utilisez les informations suivantes pour répondre à la question du client.

CONTEXTE PERTINENT:
{context}

QUESTION DU CLIENT: {user_question}

INSTRUCTIONS:
- Répondez de manière précise et professionnelle
- Utilisez uniquement les informations du contexte fourni
- Si le contexte ne contient pas assez d'informations, dites-le clairement
- Soyez concis mais complet

RÉPONSE:"""
        
        # 4. Générer la réponse avec Ollama
        try:
            print("Génération de la réponse avec llama3.1:8b...")
            response = self.ollama_client.chat(
                model='llama3.1:8b',
                messages=[
                    {
                        'role': 'user',
                        'content': rag_prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'max_tokens': 512
                }
            )
            
            answer = response['message']['content']
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": context,
                "confidence": 0.8  # Score de confiance basique
            }
            
        except Exception as e:
            print(f"❌ Erreur de génération: {e}")
            return {
                "answer": f"Erreur lors de la génération de la réponse: {e}",
                "sources": sources,
                "confidence": 0.0
            }
    
    def test_rag_system(self):
        """Teste le système RAG avec des questions"""
        print("\n=== Test du système RAG ===")
        
        test_questions = [
            "Quels sont les frais de tenue de compte?",
            "Comment ouvrir un compte épargne?",
            "Quelles sont les conditions pour un prêt immobilier?",
            "Comment activer ma carte bancaire?",
            "Que faire en cas de perte de carte?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {question}")
            print('='*60)
            
            result = self.generate_rag_response(question)
            
            print(f"🤖 Réponse RAG:")
            print(result["answer"])
            
            print(f"\n📚 Sources utilisées:")
            for j, source in enumerate(result["sources"], 1):
                print(f"  {j}. Q: {source['question'][:100]}...")
                print(f"     R: {source['answer'][:100]}...")
            
            print(f"\n📊 Confiance: {result['confidence']:.1%}")
    
    def save_rag_config(self):
        """Sauvegarde la configuration RAG"""
        config = {
            "system": "RAG with wasifis/bank-assistant-qa",
            "total_documents": len(self.documents),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_store": "ChromaDB",
            "llm_model": "llama3.1:8b (Ollama)",
            "persist_directory": "./chroma_db_wasifis",
            "chunk_size": 500,
            "chunk_overlap": 50
        }
        
        with open("rag_config_wasifis.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuration RAG sauvegardée: rag_config_wasifis.json")

def main():
    """Fonction principale"""
    print("🏦 Système RAG avec wasifis/bank-assistant-qa")
    print("=" * 50)
    
    rag_system = WasifisRAGSystem()
    
    # 1. Charger les données wasifis
    print("1. Chargement des données wasifis/bank-assistant-qa...")
    if not rag_system.load_wasifis_data():
        print("❌ Échec du chargement des données")
        return
    
    # 2. Configurer les embeddings
    print("\n2. Configuration des embeddings...")
    if not rag_system.setup_embeddings():
        print("❌ Échec de la configuration des embeddings")
        return
    
    # 3. Créer la base vectorielle
    print("\n3. Création de la base vectorielle...")
    if not rag_system.create_vector_store():
        print("❌ Échec de la création de la base vectorielle")
        return
    
    # 4. Sauvegarder la configuration
    rag_system.save_rag_config()
    
    # 5. Test du système
    print("\n4. Voulez-vous tester le système RAG? (y/n)")
    choice = input().lower().strip()
    
    if choice == 'y':
        rag_system.test_rag_system()
    
    print("\n" + "=" * 60)
    print("🎉 SYSTÈME RAG PRÊT!")
    print("✅ Base de données: wasifis/bank-assistant-qa (4,272 exemples)")
    print("✅ Embeddings: sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Base vectorielle: ChromaDB")
    print("✅ LLM: llama3.1:8b (Ollama)")
    print("\n🚀 Pour utiliser le RAG:")
    print("from rag_system_wasifis import WasifisRAGSystem")
    print("rag = WasifisRAGSystem()")
    print("response = rag.generate_rag_response('Votre question')")

if __name__ == "__main__":
    main()
