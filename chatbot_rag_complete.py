"""
Chatbot bancaire RAG complet avec interface utilisateur
Utilise wasifis/bank-assistant-qa + llama3.1:8b
"""

import os
import json
from typing import Dict, List
import gradio as gr
from datetime import datetime

# Import du système RAG
from rag_system_wasifis import WasifisRAGSystem

class BankingChatbotRAG:
    """Chatbot bancaire avec RAG complet"""
    
    def __init__(self):
        self.rag_system = None
        self.conversation_history = []
        self.initialize_rag()
    
    def initialize_rag(self):
        """Initialise le système RAG"""
        print("🔄 Initialisation du système RAG...")
        
        try:
            self.rag_system = WasifisRAGSystem()
            
            # Charger les données si pas déjà fait
            if not os.path.exists("./chroma_db_wasifis"):
                print("Base vectorielle non trouvée, création en cours...")
                self.rag_system.load_wasifis_data()
                self.rag_system.setup_embeddings()
                self.rag_system.create_vector_store()
            else:
                print("✅ Base vectorielle existante trouvée")
                # Charger les composants nécessaires
                self.rag_system.load_wasifis_data()
                self.rag_system.setup_embeddings()
                
                # Charger la base vectorielle existante
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                self.rag_system.embeddings = embeddings
                self.rag_system.vectorstore = Chroma(
                    persist_directory="./chroma_db_wasifis",
                    embedding_function=embeddings
                )
            
            print("✅ Système RAG initialisé")
            return True
            
        except Exception as e:
            print(f"❌ Erreur d'initialisation RAG: {e}")
            return False
    
    def chat_with_rag(self, user_message: str, history: List) -> tuple:
        """Chat avec le système RAG"""
        if not self.rag_system:
            error_msg = "❌ Système RAG non initialisé"
            history.append((user_message, error_msg))
            return history, ""
        
        try:
            # Générer la réponse RAG
            rag_response = self.rag_system.generate_rag_response(user_message)
            
            # Formater la réponse avec les sources
            response = rag_response["answer"]
            
            # Ajouter les sources si disponibles
            if rag_response["sources"]:
                response += "\n\n📚 **Sources consultées:**"
                for i, source in enumerate(rag_response["sources"][:2], 1):  # Limiter à 2 sources
                    response += f"\n{i}. {source['question'][:100]}..."
            
            # Ajouter à l'historique
            history.append((user_message, response))
            
            # Sauvegarder dans l'historique interne
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_message,
                "assistant": response,
                "sources": rag_response["sources"],
                "confidence": rag_response["confidence"]
            })
            
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Erreur: {e}"
            history.append((user_message, error_msg))
            return history, ""
    
    def clear_history(self):
        """Efface l'historique"""
        self.conversation_history = []
        return []
    
    def save_conversation(self):
        """Sauvegarde la conversation"""
        if not self.conversation_history:
            return "Aucune conversation à sauvegarder"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_rag_{timestamp}.json"
        
        conversation_data = {
            "system": "Banking RAG Chatbot",
            "model": "llama3.1:8b + wasifis/bank-assistant-qa",
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(self.conversation_history),
            "conversation": self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return f"✅ Conversation sauvegardée: {filename}"
    
    def get_system_info(self):
        """Retourne les informations du système"""
        if not self.rag_system:
            return "Système RAG non initialisé"
        
        info = f"""
🏦 **Chatbot Bancaire RAG**

**Configuration:**
- 🤖 LLM: llama3.1:8b (Ollama)
- 📊 Base de données: wasifis/bank-assistant-qa
- 🔍 Documents indexés: 4,272 exemples
- 📝 Chunks vectoriels: 4,680
- 🧠 Embeddings: sentence-transformers/all-MiniLM-L6-v2
- 💾 Base vectorielle: ChromaDB

**Fonctionnalités:**
- ✅ Recherche sémantique dans la base de connaissances
- ✅ Génération de réponses contextuelles
- ✅ Citation des sources utilisées
- ✅ Historique des conversations
- ✅ Sauvegarde des échanges
        """
        return info
    
    def create_gradio_interface(self):
        """Crée l'interface Gradio"""
        
        with gr.Blocks(
            title="Chatbot Bancaire RAG",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🏦 Chatbot Bancaire avec RAG")
            gr.Markdown("Assistant bancaire intelligent utilisant **llama3.1:8b** + **wasifis/bank-assistant-qa**")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Zone de chat principale
                    chatbot = gr.Chatbot(
                        value=[],
                        label="💬 Conversation",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Votre question bancaire",
                            placeholder="Ex: Quels sont les frais de compte? Comment ouvrir un compte?",
                            scale=4,
                            container=False
                        )
                        send_btn = gr.Button("Envoyer", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Effacer", variant="secondary")
                        save_btn = gr.Button("💾 Sauvegarder", variant="secondary")
                
                with gr.Column(scale=1):
                    # Panneau d'informations
                    gr.Markdown("### ℹ️ Informations")
                    system_info = gr.Markdown(self.get_system_info())
                    
                    gr.Markdown("### 🎯 Questions suggérées")
                    suggestions = [
                        "Quels sont les frais de compte?",
                        "Comment ouvrir un compte?",
                        "Conditions pour un prêt?",
                        "Activer ma carte bancaire?",
                        "Limites de transaction?"
                    ]
                    
                    for suggestion in suggestions:
                        suggest_btn = gr.Button(
                            suggestion, 
                            variant="outline", 
                            size="sm",
                            scale=1
                        )
                        suggest_btn.click(
                            lambda s=suggestion: s,
                            outputs=[msg]
                        )
                    
                    # Zone de statut
                    status = gr.Textbox(
                        label="Statut",
                        value="✅ Système RAG prêt",
                        interactive=False,
                        container=False
                    )
            
            # Actions des boutons
            send_btn.click(
                self.chat_with_rag,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                self.chat_with_rag,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                self.clear_history,
                outputs=[chatbot]
            )
            
            save_btn.click(
                self.save_conversation,
                outputs=[status]
            )
        
        return interface

def main():
    """Fonction principale"""
    print("🚀 Lancement du Chatbot Bancaire RAG")
    print("=" * 50)
    
    # Initialiser le chatbot
    chatbot = BankingChatbotRAG()
    
    if not chatbot.rag_system:
        print("❌ Impossible d'initialiser le système RAG")
        return
    
    print("✅ Chatbot RAG initialisé avec succès")
    
    # Menu de choix
    while True:
        print("\n" + "=" * 50)
        print("MENU PRINCIPAL")
        print("1. 🌐 Interface web Gradio")
        print("2. 💬 Chat en console")
        print("3. 🧪 Test rapide")
        print("4. ℹ️ Informations système")
        print("5. 🚪 Quitter")
        print("=" * 50)
        
        choice = input("Votre choix (1-5): ").strip()
        
        if choice == "1":
            print("\n🌐 Lancement de l'interface web...")
            interface = chatbot.create_gradio_interface()
            interface.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=False,
                show_error=True
            )
        
        elif choice == "2":
            print("\n💬 Chat en console (tapez 'quit' pour quitter)")
            while True:
                user_input = input("\n👤 Vous: ").strip()
                if user_input.lower() in ['quit', 'exit', 'quitter']:
                    break
                
                if user_input:
                    response = chatbot.rag_system.generate_rag_response(user_input)
                    print(f"\n🤖 Assistant: {response['answer']}")
                    
                    if response['sources']:
                        print(f"\n📚 Sources:")
                        for i, source in enumerate(response['sources'][:2], 1):
                            print(f"  {i}. {source['question'][:80]}...")
        
        elif choice == "3":
            print("\n🧪 Test rapide du système RAG...")
            test_question = "Quels sont les frais de compte?"
            response = chatbot.rag_system.generate_rag_response(test_question)
            print(f"Q: {test_question}")
            print(f"R: {response['answer'][:200]}...")
            print(f"Sources: {len(response['sources'])} documents trouvés")
        
        elif choice == "4":
            print(chatbot.get_system_info())
        
        elif choice == "5":
            print("👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")

if __name__ == "__main__":
    main()
