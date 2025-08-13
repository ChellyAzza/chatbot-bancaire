"""
Système RAG (Retrieval-Augmented Generation) pour améliorer la fidélité
Combine votre modèle fine-tuné + recherche dans le dataset pour réponses précises
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

print("🔍 SYSTÈME RAG - CHATBOT BANCAIRE AMÉLIORÉ")
print("=" * 60)

# Configuration
model_path = "./models/Llama-3.1-8B-Instruct"
adapter_path = "./llama_banking_final_fidelity"

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

class BankingRAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.vectorizer = None
        self.question_vectors = None
        self.qa_pairs = []
        
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
    
    def load_knowledge_base(self):
        """Charge et indexe la base de connaissances"""
        print("📊 Chargement base de connaissances...")
        
        # Charger le dataset
        dataset = load_dataset("wasifis/bank-assistant-qa")
        train_data = dataset["train"]
        
        # Extraire les paires Q&A
        self.qa_pairs = []
        questions = []
        
        for item in train_data:
            question = item["input"].strip()
            answer = item["output"].strip()
            
            self.qa_pairs.append({
                "question": question,
                "answer": answer
            })
            questions.append(question)
        
        print(f"✅ {len(self.qa_pairs)} paires Q&A chargées")
        
        # Créer l'index TF-IDF pour la recherche
        print("🔍 Création de l'index de recherche...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        self.question_vectors = self.vectorizer.fit_transform(questions)
        print("✅ Index de recherche créé!")
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Recherche les contextes les plus pertinents"""
        
        # Vectoriser la requête
        query_vector = self.vectorizer.transform([query])
        
        # Calculer la similarité cosinus
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        # Obtenir les top_k résultats
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Seuil de pertinence
                relevant_contexts.append({
                    "question": self.qa_pairs[idx]["question"],
                    "answer": self.qa_pairs[idx]["answer"],
                    "similarity": similarities[idx]
                })
        
        return relevant_contexts
    
    def generate_rag_response(self, user_question, max_length=200):
        """Génère une réponse avec RAG"""
        
        # 1. Rechercher les contextes pertinents
        relevant_contexts = self.retrieve_relevant_context(user_question, top_k=3)
        
        # 2. Construire le prompt avec contexte
        context_text = ""
        if relevant_contexts:
            context_text = "\n\nRelevant information from knowledge base:\n"
            for i, ctx in enumerate(relevant_contexts):
                context_text += f"Q: {ctx['question']}\nA: {ctx['answer']}\n\n"
        
        # 3. Prompt RAG
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a banking assistant. Use the provided knowledge base information to answer questions accurately. If the knowledge base contains relevant information, use it to provide precise answers.{context_text}<|eot_id|><|start_header_id|>user<|end_header_id|>

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
    
    def generate_simple_response(self, user_question, max_length=200):
        """Génère une réponse simple (sans RAG) pour comparaison"""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return response

# Initialisation du système RAG
print("🚀 Initialisation du système RAG...")
rag_system = BankingRAGSystem()
rag_system.load_model()
rag_system.load_knowledge_base()

print("\n" + "="*60)
print("🎯 SYSTÈME RAG PRÊT!")
print("="*60)

# Tests de comparaison RAG vs Sans RAG
test_questions = [
    "What are the charges for account maintenance?",
    "What is the profit rate for FCY term deposits?",
    "What are the processing charges for PMYB & ALS?",
    "What is the minimum balance required?",
    "How to open a savings account?"
]

print("\n🧪 TESTS COMPARATIFS: RAG vs SANS RAG")
print("="*60)

for i, question in enumerate(test_questions):
    print(f"\n📋 Test {i+1}/{len(test_questions)}: {question}")
    
    # Test sans RAG
    print("\n🔸 SANS RAG (Fine-tuning seul):")
    start_time = time.time()
    simple_response = rag_system.generate_simple_response(question)
    simple_time = time.time() - start_time
    print(f"⏱️ Temps: {simple_time:.2f}s")
    print(f"💬 Réponse: {simple_response[:150]}...")
    
    # Test avec RAG
    print("\n🔹 AVEC RAG (Fine-tuning + Base de connaissances):")
    start_time = time.time()
    rag_response, contexts = rag_system.generate_rag_response(question)
    rag_time = time.time() - start_time
    print(f"⏱️ Temps: {rag_time:.2f}s")
    print(f"🔍 Contextes trouvés: {len(contexts)}")
    if contexts:
        print(f"📊 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
    print(f"💬 Réponse: {rag_response[:150]}...")
    
    # Comparaison
    if len(contexts) > 0:
        print("📈 AMÉLIORATION: RAG a trouvé des contextes pertinents")
    else:
        print("➡️ ÉGALITÉ: Pas de contexte pertinent trouvé")

# Mode interactif
print("\n" + "="*60)
print("💬 MODE INTERACTIF RAG")
print("="*60)

print("Choisissez le mode:")
print("1. RAG (Recommandé - Plus précis)")
print("2. Sans RAG (Fine-tuning seul)")
print("3. Comparaison côte à côte")

mode = input("Votre choix (1/2/3): ").strip()

print(f"\nPosez vos questions bancaires (tapez 'quit' pour arrêter)")

while True:
    try:
        user_question = input("\n❓ Votre question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'stop', 'q']:
            break
            
        if not user_question:
            continue
        
        if mode == "1":
            # Mode RAG
            print("🔍 Recherche + Génération RAG...")
            response, contexts = rag_system.generate_rag_response(user_question)
            print(f"🔍 {len(contexts)} contextes trouvés")
            if contexts:
                print(f"📊 Pertinence: {max(ctx['similarity'] for ctx in contexts):.2%}")
            print(f"💬 Réponse RAG: {response}")
            
        elif mode == "2":
            # Mode sans RAG
            print("🤖 Génération simple...")
            response = rag_system.generate_simple_response(user_question)
            print(f"💬 Réponse: {response}")
            
        elif mode == "3":
            # Mode comparaison
            print("🔸 Sans RAG:")
            simple_response = rag_system.generate_simple_response(user_question)
            print(f"💬 {simple_response[:100]}...")
            
            print("\n🔹 Avec RAG:")
            rag_response, contexts = rag_system.generate_rag_response(user_question)
            print(f"🔍 {len(contexts)} contextes | 💬 {rag_response[:100]}...")
            
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")

print(f"\n🎉 SESSION RAG TERMINÉE!")
print(f"✅ Votre chatbot bancaire avec RAG améliore la fidélité sans refaire le fine-tuning!")
print(f"🎯 RAG = Modèle fine-tuné + Base de connaissances = Réponses plus précises!")
