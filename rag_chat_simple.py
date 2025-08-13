"""
Chat RAG simple et fonctionnel
"""

import os
import json
import ollama
from datetime import datetime

class SimpleRAGChat:
    """Chat RAG simplifié"""
    
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.vectorstore = None
        self.load_existing_vectorstore()
    
    def load_existing_vectorstore(self):
        """Charge la base vectorielle existante"""
        try:
            if os.path.exists("./chroma_db_wasifis"):
                print("✅ Base vectorielle RAG trouvée")
                
                # Importer les composants nécessaires
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                self.vectorstore = Chroma(
                    persist_directory="./chroma_db_wasifis",
                    embedding_function=embeddings
                )
                
                print("✅ Système RAG chargé avec succès")
                return True
            else:
                print("❌ Base vectorielle non trouvée")
                print("Exécutez d'abord: python rag_system_wasifis.py")
                return False
                
        except Exception as e:
            print(f"❌ Erreur de chargement: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 3):
        """Recherche des documents pertinents"""
        if not self.vectorstore:
            return []
        
        try:
            similar_docs = self.vectorstore.similarity_search(query=query, k=k)
            return similar_docs
        except Exception as e:
            print(f"❌ Erreur de recherche: {e}")
            return []
    
    def generate_rag_response(self, user_question: str):
        """Génère une réponse RAG"""
        print(f"\n🔍 Question: {user_question}")
        
        # 1. Rechercher des documents
        relevant_docs = self.search_documents(user_question, k=3)
        
        if not relevant_docs:
            return {
                "answer": "Je n'ai pas trouvé d'informations pertinentes dans ma base de connaissances.",
                "sources": []
            }
        
        # 2. Construire le contexte
        context_parts = []
        sources = []
        
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"Document {i+1}: {doc.page_content}")
            sources.append({
                "content": doc.page_content[:150] + "...",
                "metadata": doc.metadata
            })
        
        context = "\n\n".join(context_parts)
        
        # 3. Prompt RAG
        rag_prompt = f"""Vous êtes un assistant bancaire expert. Utilisez les informations suivantes pour répondre à la question.

CONTEXTE:
{context}

QUESTION: {user_question}

Répondez de manière professionnelle en utilisant uniquement les informations du contexte. Si le contexte ne contient pas assez d'informations, dites-le clairement.

RÉPONSE:"""
        
        # 4. Générer avec Ollama
        try:
            response = self.ollama_client.chat(
                model='llama3.1:8b',
                messages=[{'role': 'user', 'content': rag_prompt}],
                options={'temperature': 0.7, 'max_tokens': 512}
            )
            
            return {
                "answer": response['message']['content'],
                "sources": sources
            }
            
        except Exception as e:
            return {
                "answer": f"Erreur de génération: {e}",
                "sources": sources
            }
    
    def interactive_chat(self):
        """Chat interactif"""
        print("\n🏦 Chat Bancaire RAG")
        print("=" * 40)
        print("💬 Posez vos questions bancaires (tapez 'quit' pour quitter)")
        print("🔍 Le système recherche dans 4,272 exemples de wasifis/bank-assistant-qa")
        print("-" * 40)
        
        conversation_history = []
        
        while True:
            user_input = input("\n👤 Vous: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'quitter', 'q']:
                print("👋 Au revoir!")
                
                # Proposer de sauvegarder
                if conversation_history:
                    save = input("💾 Sauvegarder la conversation? (y/n): ").lower()
                    if save == 'y':
                        self.save_conversation(conversation_history)
                break
            
            if not user_input:
                continue
            
            # Générer la réponse RAG
            result = self.generate_rag_response(user_input)
            
            print(f"\n🤖 Assistant RAG:")
            print(result["answer"])
            
            if result["sources"]:
                print(f"\n📚 Sources consultées ({len(result['sources'])} documents):")
                for i, source in enumerate(result["sources"], 1):
                    print(f"  {i}. {source['content']}")
            
            # Ajouter à l'historique
            conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "assistant": result["answer"],
                "sources_count": len(result["sources"])
            })
    
    def save_conversation(self, history):
        """Sauvegarde la conversation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_rag_{timestamp}.json"
        
        data = {
            "system": "Simple RAG Chat",
            "model": "llama3.1:8b + wasifis/bank-assistant-qa",
            "timestamp": datetime.now().isoformat(),
            "conversation": history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Conversation sauvegardée: {filename}")
    
    def test_rag(self):
        """Test rapide du RAG"""
        print("\n🧪 Test du système RAG")
        print("-" * 30)
        
        test_questions = [
            "Quels sont les frais de compte?",
            "Comment ouvrir un compte?",
            "Limites de transaction?"
        ]
        
        for question in test_questions:
            print(f"\n❓ {question}")
            result = self.generate_rag_response(question)
            print(f"✅ Réponse: {result['answer'][:100]}...")
            print(f"📊 Sources: {len(result['sources'])} documents")

def main():
    """Fonction principale"""
    print("🚀 Chat Bancaire RAG Simple")
    print("=" * 40)
    
    chat = SimpleRAGChat()
    
    if not chat.vectorstore:
        print("❌ Système RAG non disponible")
        print("Exécutez d'abord: python rag_system_wasifis.py")
        return
    
    while True:
        print("\nOptions:")
        print("1. 💬 Chat interactif")
        print("2. 🧪 Test rapide")
        print("3. ℹ️ Informations")
        print("4. 🚪 Quitter")
        
        choice = input("\nChoix (1-4): ").strip()
        
        if choice == "1":
            chat.interactive_chat()
        elif choice == "2":
            chat.test_rag()
        elif choice == "3":
            print(f"""
🏦 Système RAG Bancaire

Configuration:
- 🤖 LLM: llama3.1:8b (Ollama)
- 📊 Base: wasifis/bank-assistant-qa
- 🔍 Documents: 4,272 exemples
- 🧠 Embeddings: sentence-transformers/all-MiniLM-L6-v2
- 💾 Vectorstore: ChromaDB

Fonctionnalités:
✅ Recherche sémantique
✅ Génération contextuelle
✅ Citation des sources
✅ Sauvegarde conversations
            """)
        elif choice == "4":
            print("👋 Au revoir!")
            break
        else:
            print("❌ Choix invalide")

if __name__ == "__main__":
    main()
