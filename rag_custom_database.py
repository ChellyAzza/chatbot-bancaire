"""
RAG avec votre base de données nettoyée
Utilise banking_documents/ pour des réponses précises
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

print("🔍 RAG AVEC VOTRE BASE NETTOYÉE")
print("=" * 60)

# Configuration
model_path = "./models/Llama-3.1-8B-Instruct"
adapter_path = "./llama_banking_final_fidelity"
knowledge_base_path = "./banking_documents"

# Vérifications
if not torch.cuda.is_available():
    print("❌ CUDA requis")
    exit()

print(f"✅ GPU: {torch.cuda.get_device_name()}")
torch.cuda.empty_cache()

# Configuration quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

class CustomRAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.knowledge_base = []
        self.vectorizer = None
        self.document_vectors = None
        
    def load_model(self):
        """Charge le modèle fine-tuné"""
        print("📝 Chargement tokenizer...")
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
        
        print("🦙 Chargement modèle fine-tuné...")
        if Path(model_path).exists():
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token="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"
            )
        
        self.model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            torch_dtype=torch.float16
        )
        
        print("✅ Modèle fine-tuné chargé!")
    
    def load_custom_knowledge_base(self):
        """Charge votre base de connaissances nettoyée"""
        print("📊 Chargement de votre base nettoyée...")

        self.knowledge_base = []
        documents_text = []

        # 🎯 POINT D'IMPLÉMENTATION 1: CHARGEMENT DE VOTRE BASE
        # MODIFIEZ ICI pour adapter à votre format de données

        # Charger tous les fichiers texte
        for filename in os.listdir(knowledge_base_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(knowledge_base_path, filename)
                category = filename.replace('.txt', '').replace('_', ' ').title()

                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # 🔧 PERSONNALISEZ: Comment diviser vos documents
                sections = content.split('\n\n')  # Changez selon votre format
                for i, section in enumerate(sections):
                    if section.strip():
                        self.knowledge_base.append({
                            "category": category,
                            "section": i + 1,
                            "content": section.strip(),
                            "source": filename
                        })
                        documents_text.append(section.strip())
        
        # Charger les fichiers JSON
        for filename in ['faq.json', 'index.json']:
            filepath = os.path.join(knowledge_base_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if filename == 'faq.json' and isinstance(data, list):
                    for item in data:
                        if 'question' in item and 'answer' in item:
                            self.knowledge_base.append({
                                "category": "FAQ",
                                "content": f"Q: {item['question']}\nA: {item['answer']}",
                                "source": filename
                            })
                            documents_text.append(f"{item['question']} {item['answer']}")
        
        print(f"✅ {len(self.knowledge_base)} documents chargés")
        
        # Créer l'index TF-IDF
        print("🔍 Création de l'index de recherche...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3),
            lowercase=True
        )
        
        self.document_vectors = self.vectorizer.fit_transform(documents_text)
        print("✅ Index de recherche créé!")
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Recherche les contextes les plus pertinents dans votre base"""
        
        # Vectoriser la requête
        query_vector = self.vectorizer.transform([query])
        
        # Calculer la similarité cosinus
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Obtenir les top_k résultats
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Seuil de pertinence
                relevant_contexts.append({
                    "content": self.knowledge_base[idx]["content"],
                    "category": self.knowledge_base[idx]["category"],
                    "source": self.knowledge_base[idx]["source"],
                    "similarity": similarities[idx]
                })
        
        return relevant_contexts
    
    def generate_rag_response(self, user_question, max_length=200):
        """Génère une réponse avec RAG utilisant votre base"""
        
        # 1. Rechercher les contextes pertinents
        relevant_contexts = self.retrieve_relevant_context(user_question, top_k=3)
        
        # 2. Construire le prompt avec contexte
        context_text = ""
        if relevant_contexts:
            context_text = "\n\nInformations pertinentes de la base de connaissances:\n"
            for i, ctx in enumerate(relevant_contexts):
                context_text += f"[{ctx['category']}] {ctx['content']}\n\n"
        
        # 3. Prompt RAG optimisé
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Vous êtes un assistant bancaire expert. Utilisez les informations de la base de connaissances pour répondre avec précision. Si la base contient des informations pertinentes, utilisez-les pour donner des réponses exactes et détaillées.{context_text}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # 4. Génération avec le modèle
        inputs = self.tokenizer(rag_prompt, return_tensors="pt", truncation=True, max_length=1024)
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,  # Très bas pour fidélité
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return response, relevant_contexts

# Initialisation du système RAG
print("🚀 Initialisation du système RAG personnalisé...")
rag_system = CustomRAGSystem()
rag_system.load_model()
rag_system.load_custom_knowledge_base()

print("\n" + "="*60)
print("🎯 SYSTÈME RAG PERSONNALISÉ PRÊT!")
print("="*60)

# Tests avec votre base
test_questions = [
    "Quels sont les frais de tenue de compte?",
    "Comment ouvrir un compte épargne?",
    "Quelles sont les conditions pour un prêt?",
    "Comment faire une réclamation?",
    "Quels services numériques sont disponibles?"
]

print("\n🧪 TESTS AVEC VOTRE BASE NETTOYÉE")
print("="*60)

for i, question in enumerate(test_questions):
    print(f"\n📋 Test {i+1}/{len(test_questions)}: {question}")
    
    start_time = time.time()
    response, contexts = rag_system.generate_rag_response(question)
    response_time = time.time() - start_time
    
    print(f"⏱️ Temps: {response_time:.2f}s")
    print(f"🔍 Contextes trouvés: {len(contexts)}")
    
    if contexts:
        print(f"📊 Sources: {', '.join(set(ctx['category'] for ctx in contexts))}")
        print(f"📈 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
    
    print(f"💬 Réponse: {response[:200]}...")

# Mode interactif
print("\n" + "="*60)
print("💬 MODE INTERACTIF - RAG PERSONNALISÉ")
print("="*60)

print("Posez vos questions bancaires (tapez 'quit' pour arrêter)")
print("Votre RAG utilise votre base de données nettoyée!")

while True:
    try:
        user_question = input("\n❓ Votre question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'stop', 'q']:
            break
            
        if not user_question:
            continue
        
        print("🔍 Recherche dans votre base + Génération...")
        start_time = time.time()
        response, contexts = rag_system.generate_rag_response(user_question)
        response_time = time.time() - start_time
        
        print(f"⏱️ Temps: {response_time:.2f}s")
        print(f"🔍 {len(contexts)} contextes trouvés")
        
        if contexts:
            print(f"📊 Sources: {', '.join(set(ctx['category'] for ctx in contexts))}")
            print(f"📈 Pertinence: {max(ctx['similarity'] for ctx in contexts):.2%}")
            
            # Afficher les sources utilisées
            print("📚 Sources utilisées:")
            for ctx in contexts:
                print(f"  - {ctx['category']} (similarité: {ctx['similarity']:.2%})")
        
        print(f"💬 Réponse: {response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")

print(f"\n🎉 SESSION RAG PERSONNALISÉ TERMINÉE!")
print(f"✅ Votre chatbot utilise maintenant votre base de données nettoyée!")
print(f"🎯 RAG = Modèle fine-tuné + Votre base personnalisée = Réponses ultra-précises!")
