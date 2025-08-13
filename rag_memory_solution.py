"""
SOLUTION 3: RAG avec Mémoire Conversationnelle
Garde le contexte des conversations précédentes
"""

class ConversationalRAG:
    def __init__(self):
        self.conversation_history = []
        self.user_context = {}
        
    def add_to_memory(self, user_question, response, contexts):
        """Ajoute à la mémoire conversationnelle"""
        
        # 🎯 POINT D'IMPLÉMENTATION: Gestion de la mémoire
        self.conversation_history.append({
            "question": user_question,
            "response": response,
            "contexts": contexts,
            "timestamp": time.time()
        })
        
        # 🔧 LIGNE CLÉS: Limiter la mémoire (garder 10 derniers échanges)
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def get_conversation_context(self):
        """Récupère le contexte conversationnel"""
        
        # 🎯 IMPLÉMENTATION: Construire le contexte
        if not self.conversation_history:
            return ""
        
        context = "\n\nContexte de la conversation précédente:\n"
        for exchange in self.conversation_history[-3:]:  # 3 derniers échanges
            context += f"Q: {exchange['question']}\nR: {exchange['response'][:100]}...\n"
        
        return context
    
    def enhanced_rag_response(self, user_question):
        """RAG avec mémoire conversationnelle"""
        
        # 🔧 LIGNE CLÉS: Inclure le contexte conversationnel
        conversation_context = self.get_conversation_context()
        
        # 🎯 IMPLÉMENTATION: Prompt enrichi
        enhanced_prompt = f"""
        Contexte conversationnel: {conversation_context}
        
        Question actuelle: {user_question}
        
        Répondez en tenant compte du contexte de la conversation.
        """
        
        return enhanced_prompt

# 🎯 COMMENT IMPLÉMENTER:
# 1. Ajoutez cette classe à rag_custom_database.py
# 2. Modifiez la ligne 167 (generate_rag_response) pour inclure:
#    conversation_context = self.get_conversation_context()
# 3. Ajoutez après chaque réponse:
#    self.add_to_memory(user_question, response, contexts)

print("💡 SOLUTION 3: RAG avec Mémoire Conversationnelle")
print("🧠 Garde le contexte des conversations")
print("🔄 Réponses cohérentes dans le temps")
print("👤 Personnalisation selon l'utilisateur")
