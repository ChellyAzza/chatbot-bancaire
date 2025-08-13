"""
RAG corrigé avec base de données nettoyée
Utilise seulement input/output, extraction de réponse améliorée
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

print("🔍 RAG CORRIGÉ - BASE NETTOYÉE")
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

class CleanRAGSystem:
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
        
        # Essayer de charger le fichier nettoyé
        clean_file = "cleaned_banking_qa.json"
        
        try:
            print(f"📊 Chargement base nettoyée: {clean_file}...")
            with open(clean_file, 'r', encoding='utf-8') as f:
                self.qa_pairs = json.load(f)
            print(f"✅ {len(self.qa_pairs)} paires Q&A chargées depuis fichier nettoyé")
            
        except FileNotFoundError:
            print(f"⚠️ Fichier {clean_file} non trouvé, nettoyage automatique...")
            
            # Charger et nettoyer automatiquement
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
            
            # Sauvegarder pour la prochaine fois
            with open(clean_file, 'w', encoding='utf-8') as f:
                json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {len(self.qa_pairs)} paires Q&A nettoyées et sauvegardées")
        
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
                    "question": self.qa_pairs[idx]["input"],
                    "answer": self.qa_pairs[idx]["output"],
                    "similarity": similarities[idx]
                })
        
        return relevant_contexts
    
    def generate_simple_response(self, user_question, max_length=200):
        """Génère une réponse simple (sans RAG)"""
        
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
        
        # 🔧 EXTRACTION AMÉLIORÉE DE LA RÉPONSE
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Nettoyer les artefacts
        response = response.replace("<|eot_id|>", "").strip()
        
        return response
    
    def generate_rag_response(self, user_question, max_length=300):
        """Génère une réponse avec RAG - VERSION CORRIGÉE"""
        
        # 1. Rechercher les contextes pertinents
        relevant_contexts = self.retrieve_relevant_context(user_question, top_k=3)
        
        # 2. Construire le contexte de manière simple
        context_text = ""
        if relevant_contexts:
            context_text = "\n\nRelevant examples:\n"
            for ctx in relevant_contexts:
                context_text += f"Q: {ctx['question']}\nA: {ctx['answer']}\n\n"
        
        # 3. Prompt simplifié SANS instruction système visible
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Based on the following examples, answer the question accurately:{context_text}

Question: {user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # 4. Génération
        inputs = self.tokenizer(rag_prompt, return_tensors="pt", truncation=True, max_length=1024)
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.2,  # Augmenté pour plus de créativité
                do_sample=True,
                top_p=0.9,  # Ajouté pour meilleure génération
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15  # Augmenté pour éviter répétitions
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 🔧 EXTRACTION AMÉLIORÉE DE LA RÉPONSE
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Nettoyer les artefacts
        response = response.replace("<|eot_id|>", "").strip()
        response = response.replace("Based on the following examples", "").strip()
        
        return response, relevant_contexts

# Initialisation du système
print("🚀 Initialisation du système RAG corrigé...")
rag_system = CleanRAGSystem()
rag_system.load_model()
rag_system.load_clean_knowledge_base()

print("\n" + "="*60)
print("🎯 SYSTÈME RAG CORRIGÉ PRÊT!")
print("="*60)

# Test automatique de votre question
test_question = "Can applicant avail clean loan in NUST Sahar Finance?"

print(f"\n🧪 TEST AUTOMATIQUE:")
print(f"❓ Question: {test_question}")

# Sans RAG
print(f"\n🔸 SANS RAG:")
start_time = time.time()
simple_response = rag_system.generate_simple_response(test_question)
simple_time = time.time() - start_time
print(f"⏱️ Temps: {simple_time:.2f}s")
print(f"💬 Réponse: {simple_response}")

# Avec RAG
print(f"\n🔹 AVEC RAG:")
start_time = time.time()
rag_response, contexts = rag_system.generate_rag_response(test_question)
rag_time = time.time() - start_time
print(f"⏱️ Temps: {rag_time:.2f}s")
print(f"🔍 Contextes trouvés: {len(contexts)}")
if contexts:
    print(f"📊 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
print(f"💬 Réponse: {rag_response}")

# Mode interactif
print(f"\n💬 MODE INTERACTIF (tapez 'quit' pour arrêter):")

while True:
    try:
        user_question = input("\n❓ Votre question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'stop', 'q']:
            break
            
        if not user_question:
            continue
        
        print("🔹 AVEC RAG:")
        start_time = time.time()
        response, contexts = rag_system.generate_rag_response(user_question)
        response_time = time.time() - start_time
        
        print(f"⏱️ Temps: {response_time:.2f}s")
        print(f"🔍 {len(contexts)} contextes trouvés")
        if contexts:
            print(f"📊 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
        print(f"💬 Réponse: {response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")

print(f"\n🎉 SESSION TERMINÉE!")
print(f"✅ RAG corrigé avec extraction de réponse améliorée!")
