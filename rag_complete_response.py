"""
RAG optimisé pour réponses complètes
Paramètres ajustés pour générer des réponses plus détaillées
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

print("🔍 RAG OPTIMISÉ - RÉPONSES COMPLÈTES")
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

class OptimizedRAGSystem:
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
    
    def retrieve_relevant_context(self, query, top_k=5):  # Augmenté à 5 contextes
        """Recherche les contextes les plus pertinents"""
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Seuil réduit pour plus de contextes
                relevant_contexts.append({
                    "question": self.qa_pairs[idx]["input"],
                    "answer": self.qa_pairs[idx]["output"],
                    "similarity": similarities[idx]
                })
        
        return relevant_contexts
    
    def generate_complete_rag_response(self, user_question):
        """Génère une réponse complète avec RAG optimisé"""
        
        # 1. Rechercher plus de contextes
        relevant_contexts = self.retrieve_relevant_context(user_question, top_k=5)
        
        # 2. Construire un contexte riche
        context_text = ""
        if relevant_contexts:
            context_text = "\n\nRelevant banking information:\n"
            for i, ctx in enumerate(relevant_contexts[:3]):  # Top 3 seulement
                context_text += f"Example {i+1}:\nQ: {ctx['question']}\nA: {ctx['answer']}\n\n"
        
        # 3. Prompt optimisé pour réponses complètes
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

You are a banking expert. Based on the following banking information, provide a complete and detailed answer to the question.{context_text}

Question: {user_question}

Please provide a comprehensive answer with all relevant details.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # 4. Génération avec paramètres optimisés
        inputs = self.tokenizer(rag_prompt, return_tensors="pt", truncation=True, max_length=1500)
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,  # Augmenté pour réponses plus longues
                temperature=0.3,     # Équilibré créativité/précision
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                length_penalty=1.1   # Encourage les réponses plus longues
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 5. Extraction et nettoyage améliorés
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Nettoyer tous les artefacts
        response = response.replace("<|eot_id|>", "").strip()
        response = response.replace("Based on the following", "").strip()
        response = response.replace("You are a banking expert.", "").strip()
        
        # Supprimer les lignes vides multiples
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():
                cleaned_lines.append(line.strip())
        
        response = '\n'.join(cleaned_lines)
        
        return response, relevant_contexts

# Initialisation du système
print("🚀 Initialisation du système RAG optimisé...")
rag_system = OptimizedRAGSystem()
rag_system.load_model()
rag_system.load_clean_knowledge_base()

print("\n" + "="*60)
print("🎯 SYSTÈME RAG OPTIMISÉ PRÊT!")
print("="*60)

# Test de votre question spécifique
test_question = "Can applicant avail clean loan in NUST Sahar Finance?"

print(f"\n🧪 TEST OPTIMISÉ:")
print(f"❓ Question: {test_question}")

print(f"\n🔹 RÉPONSE COMPLÈTE AVEC RAG:")
start_time = time.time()
response, contexts = rag_system.generate_complete_rag_response(test_question)
response_time = time.time() - start_time

print(f"⏱️ Temps: {response_time:.2f}s")
print(f"🔍 Contextes trouvés: {len(contexts)}")
if contexts:
    print(f"📊 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
    print(f"📚 Sources utilisées:")
    for i, ctx in enumerate(contexts[:3]):
        print(f"  {i+1}. Similarité: {ctx['similarity']:.2%}")

print(f"\n💬 RÉPONSE COMPLÈTE:")
print(f"{response}")

# Mode interactif optimisé
print(f"\n💬 MODE INTERACTIF OPTIMISÉ (tapez 'quit' pour arrêter):")

while True:
    try:
        user_question = input("\n❓ Votre question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'stop', 'q']:
            break
            
        if not user_question:
            continue
        
        print("🔹 GÉNÉRATION RÉPONSE COMPLÈTE...")
        start_time = time.time()
        response, contexts = rag_system.generate_complete_rag_response(user_question)
        response_time = time.time() - start_time
        
        print(f"⏱️ Temps: {response_time:.2f}s")
        print(f"🔍 {len(contexts)} contextes trouvés")
        if contexts:
            print(f"📊 Similarité max: {max(ctx['similarity'] for ctx in contexts):.2%}")
        
        print(f"\n💬 RÉPONSE COMPLÈTE:")
        print(f"{response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")

print(f"\n🎉 SESSION TERMINÉE!")
print(f"✅ RAG optimisé pour réponses complètes et détaillées!")
