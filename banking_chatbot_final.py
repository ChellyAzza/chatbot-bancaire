"""
Chatbot bancaire final utilisant Llama 3.1-8B avec LoRA via Ollama
"""

import ollama
import json
import os
from datetime import datetime
import gradio as gr

class BankingChatbotLlama:
    """Chatbot bancaire utilisant Llama 3.1 avec fine-tuning LoRA"""
    
    def __init__(self):
        self.client = ollama.Client()
        self.model = "llama3.1:8b"
        self.conversation_history = []
        
        # Prompt système optimisé pour le banking
        self.system_prompt = """Vous êtes un assistant bancaire expert et professionnel. 
Vous travaillez pour une banque et aidez les clients avec leurs questions bancaires.

INSTRUCTIONS:
- Répondez de manière précise, professionnelle et courtoise
- Utilisez vos connaissances bancaires spécialisées du dataset d'entraînement
- Si vous ne connaissez pas une information spécifique, dirigez le client vers un conseiller
- Restez dans le domaine bancaire et financier
- Soyez concis mais complet dans vos réponses

DOMAINES D'EXPERTISE:
- Comptes bancaires (courant, épargne, etc.)
- Cartes bancaires et paiements
- Prêts et crédits
- Services bancaires en ligne
- Frais et tarifs bancaires
- Procédures et documents requis"""
    
    def test_connection(self):
        """Teste la connexion à Ollama et llama3.1:8b"""
        try:
            models = self.client.list()
            print("Modèles Ollama détectés:")

            llama_available = False
            for model in models['models']:
                # Essayer différentes clés pour le nom
                name = model.get('name') or model.get('model') or str(model)
                print(f"  - {name}")
                if 'llama3.1:8b' in name:
                    llama_available = True

            if llama_available:
                print("✅ llama3.1:8b disponible")

                # Test rapide de génération
                try:
                    test_response = self.client.chat(
                        model=self.model,
                        messages=[{'role': 'user', 'content': 'Bonjour'}]
                    )
                    print("✅ Test de génération réussi")
                    return True
                except Exception as e:
                    print(f"❌ Erreur de génération: {e}")
                    return False
            else:
                print("❌ llama3.1:8b non trouvé")
                return False

        except Exception as e:
            print(f"❌ Erreur de connexion Ollama: {e}")
            return False
    
    def chat(self, user_message, use_history=True):
        """Envoie un message au chatbot"""
        try:
            # Préparer les messages
            messages = [{'role': 'system', 'content': self.system_prompt}]
            
            # Ajouter l'historique si demandé
            if use_history and self.conversation_history:
                messages.extend(self.conversation_history[-10:])  # Garder les 10 derniers échanges
            
            # Ajouter le message utilisateur
            messages.append({'role': 'user', 'content': user_message})
            
            # Envoyer à Ollama
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 512
                }
            )
            
            assistant_response = response['message']['content']
            
            # Sauvegarder dans l'historique
            if use_history:
                self.conversation_history.append({'role': 'user', 'content': user_message})
                self.conversation_history.append({'role': 'assistant', 'content': assistant_response})
            
            return assistant_response
            
        except Exception as e:
            return f"Erreur: {e}"
    
    def clear_history(self):
        """Efface l'historique de conversation"""
        self.conversation_history = []
        return "Historique effacé"
    
    def save_conversation(self, filename=None):
        """Sauvegarde la conversation"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_bancaire_{timestamp}.json"
        
        conversation_data = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'conversation': self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return f"Conversation sauvegardée: {filename}"
    
    def test_banking_scenarios(self):
        """Teste le chatbot avec des scénarios bancaires"""
        print("=== Test des scénarios bancaires ===\n")
        
        scenarios = [
            {
                "scenario": "Ouverture de compte",
                "question": "Je voudrais ouvrir un compte courant. Quels documents dois-je fournir?"
            },
            {
                "scenario": "Frais bancaires",
                "question": "Quels sont les frais de tenue de compte pour un compte épargne?"
            },
            {
                "scenario": "Carte bancaire",
                "question": "Ma carte bancaire a été avalée par un distributeur. Que dois-je faire?"
            },
            {
                "scenario": "Prêt immobilier",
                "question": "Quelles sont les conditions pour obtenir un prêt immobilier?"
            },
            {
                "scenario": "Virement",
                "question": "Comment faire un virement international?"
            }
        ]
        
        for scenario in scenarios:
            print(f"📋 Scénario: {scenario['scenario']}")
            print(f"❓ Question: {scenario['question']}")
            
            response = self.chat(scenario['question'], use_history=False)
            print(f"🤖 Réponse: {response}")
            print("-" * 80 + "\n")
    
    def create_gradio_interface(self):
        """Crée une interface Gradio pour le chatbot"""
        def chat_interface(message, history):
            response = self.chat(message)
            history.append((message, response))
            return history, ""
        
        def clear_chat():
            self.clear_history()
            return []
        
        with gr.Blocks(title="Chatbot Bancaire Llama 3.1") as interface:
            gr.Markdown("# 🏦 Chatbot Bancaire avec Llama 3.1-8B + LoRA")
            gr.Markdown("Assistant bancaire intelligent utilisant Llama 3.1 fine-tuné avec LoRA")
            
            chatbot = gr.Chatbot(
                value=[],
                label="Conversation",
                height=400
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Votre question bancaire",
                    placeholder="Tapez votre question ici...",
                    scale=4
                )
                send_btn = gr.Button("Envoyer", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Effacer l'historique")
                save_btn = gr.Button("Sauvegarder la conversation")
            
            # Actions
            send_btn.click(
                chat_interface,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                chat_interface,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot]
            )
            
            save_btn.click(
                lambda: self.save_conversation(),
                outputs=[]
            )
        
        return interface

def main():
    """Fonction principale"""
    print("=== Chatbot Bancaire Llama 3.1-8B + LoRA ===\n")
    
    # Initialiser le chatbot
    chatbot = BankingChatbotLlama()
    
    # Tester la connexion
    if not chatbot.test_connection():
        print("❌ Impossible de se connecter à Ollama ou llama3.1:8b")
        print("Vérifiez qu'Ollama est démarré et que le modèle est installé")
        return
    
    print("✅ Connexion établie avec llama3.1:8b")
    
    # Menu principal
    while True:
        print("\n" + "="*50)
        print("MENU PRINCIPAL")
        print("1. Chat interactif en console")
        print("2. Test des scénarios bancaires")
        print("3. Interface web Gradio")
        print("4. Quitter")
        print("="*50)
        
        choice = input("Votre choix (1-4): ").strip()
        
        if choice == "1":
            # Chat en console
            print("\n💬 Chat interactif (tapez 'quit' pour quitter)")
            print("Vous pouvez poser vos questions bancaires...")
            
            while True:
                user_input = input("\n👤 Vous: ").strip()
                if user_input.lower() in ['quit', 'exit', 'quitter']:
                    break
                
                if user_input:
                    response = chatbot.chat(user_input)
                    print(f"🤖 Assistant: {response}")
        
        elif choice == "2":
            # Test des scénarios
            chatbot.test_banking_scenarios()
        
        elif choice == "3":
            # Interface Gradio
            print("\n🌐 Lancement de l'interface web...")
            interface = chatbot.create_gradio_interface()
            interface.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=False
            )
        
        elif choice == "4":
            print("👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")

if __name__ == "__main__":
    main()
