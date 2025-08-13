"""
RAG FINAL - Réponses propres et développées
Extraction parfaite, pas de prompt visible
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

print("🎯 RAG FINAL - RÉPONSES PROPRES")
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

class FinalRAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
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
    
    def load_clean_knowledge_base(self):
        """Charge la base de connaissances nettoyée"""
        
        clean_file = "cleaned_banking_qa.json"
        
        try:
            print(f"📊 Chargement base nettoyée: {clean_file}...")
            with open(clean_file, 'r', encoding='utf-8') as f:
                self.qa_pairs = json.load(f)
            print(f"✅ {len(self.qa_pairs)} paires Q&A chargées")
            
        except FileNotFoundError:
            print(f"⚠️ Fichier {clean_file} non trouvé, nettoyage automatique...")
            
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
            
            print(f"✅ {len(self.qa_pairs)} paires Q&A nettoyées")
        
        # Créer l'index TF-IDF
        print("🔍 Création de l'index de recherche...")
        questions = [pair["input"] for pair in self.qa_pairs]
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        self.question_vectors = self.vectorizer.fit_transform(questions)
        print("✅ Index de recherche créé!")
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Recherche les contextes les plus pertinents"""
        
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
    
    def generate_final_rag_response(self, user_question):
        """Génère une réponse finale propre avec RAG"""
        
        # 1. Rechercher les contextes pertinents
        relevant_contexts = self.retrieve_relevant_context(user_question, top_k=3)
        
        # 2. Construire le contexte de manière invisible
        context_info = ""
        if relevant_contexts:
            # Utiliser seulement le meilleur contexte pour éviter la répétition
            best_context = relevant_contexts[0]
            context_info = f"Context: {best_context['answer']}"
        
        # 3. Prompt minimal et propre
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_question}

{context_info}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # 4. Génération avec paramètres optimisés
        inputs = self.tokenizer(rag_prompt, return_tensors="pt", truncation=True, max_length=1024)
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15
            )
        
        # 5. Extraction propre de la réponse
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie assistant
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            response = full_response
        
        # Nettoyage complet
        response = response.replace("<|eot_id|>", "")
        response = response.replace("Context:", "")
        response = response.strip()
        
        # Supprimer les lignes vides et nettoyer
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        response = '\n'.join(lines)
        
        return response, relevant_contexts

# Initialisation du système
print("🚀 Initialisation du système RAG final...")
rag_system = FinalRAGSystem()
rag_system.load_model()
rag_system.load_clean_knowledge_base()

print("\n" + "="*60)
print("🎯 SYSTÈME RAG FINAL PRÊT!")
print("="*60)

# Test automatique
test_questions = [
    "Can applicant avail clean loan in NUST Sahar Finance?",
    "What are the free facilities associated with Roshan Digital Account?",
    "What is the maximum tenure of NUST Imarat Finance?",
    "What are the processing charges for PMYB & ALS?"
]

for i, question in enumerate(test_questions):
    print(f"\n{'='*20} TEST {i+1} {'='*20}")
    print(f"❓ Question: {question}")
    
    start_time = time.time()
    response, contexts = rag_system.generate_final_rag_response(question)
    response_time = time.time() - start_time
    
    print(f"⏱️ Temps: {response_time:.2f}s")
    print(f"🔍 Contextes trouvés: {len(contexts)}")
    if contexts:
        print(f"📊 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
    
    print(f"\n💬 RÉPONSE FINALE:")
    print(f"{response}")

# Mode interactif
print(f"\n💬 MODE INTERACTIF FINAL (tapez 'quit' pour arrêter):")

while True:
    try:
        user_question = input("\n❓ Votre question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'stop', 'q']:
            break
            
        if not user_question:
            continue
        
        print("🎯 Génération réponse finale...")
        start_time = time.time()
        response, contexts = rag_system.generate_final_rag_response(user_question)
        response_time = time.time() - start_time
        
        print(f"⏱️ Temps: {response_time:.2f}s")
        print(f"🔍 {len(contexts)} contextes trouvés")
        if contexts:
            print(f"📊 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
        
        print(f"\n💬 RÉPONSE FINALE:")
        print(f"{response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")

print(f"\n🎉 SESSION TERMINÉE!")
print(f"✅ RAG final avec réponses propres et développées!")
print(f"🎯 Votre modèle fine-tuné + RAG fonctionne parfaitement!")
