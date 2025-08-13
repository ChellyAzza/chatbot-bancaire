"""
RAG avec réponses 100% développées et détaillées
Génère des réponses complètes, pas juste des faits bruts
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Nouvelles imports pour reconnaissance d'images et recommandations
try:
    from PIL import Image
    import pytesseract
    import cv2
    IMAGE_SUPPORT = True
    print("✅ Support reconnaissance d'images activé")
except ImportError:
    IMAGE_SUPPORT = False
    print("⚠️ Support reconnaissance d'images non disponible (pip install pillow pytesseract opencv-python)")

print("🎯 RAG - RÉPONSES 100% DÉVELOPPÉES")
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

class DevelopedRAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.vectorizer = None
        self.question_vectors = None
        self.qa_pairs = []
        
    def load_model(self):
        """Charge le modèle avec configuration optimisée"""
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
        
        try:
            # Configuration optimisée pour RTX 4060
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            if Path(model_path).exists():
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    max_memory={0: "7GB", "cpu": "12GB"}
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    max_memory={0: "7GB", "cpu": "12GB"},
                    token="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"
                )
            
            print("✅ Quantization réussie!")
            
        except Exception as e:
            print(f"⚠️ Fallback sans quantization...")
            if Path(model_path).exists():
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": "cuda:0"},
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    device_map={"": "cuda:0"},
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    token="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"
                )
        
        # Charger l'adaptateur LoRA
        self.model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            torch_dtype=torch.float16
        )
        
        print("✅ Modèle fine-tuné chargé!")
    
    def load_clean_knowledge_base(self):
        """Charge la base de connaissances"""
        clean_file = "cleaned_banking_qa.json"
        
        try:
            with open(clean_file, 'r', encoding='utf-8') as f:
                self.qa_pairs = json.load(f)
            print(f"✅ {len(self.qa_pairs)} paires Q&A chargées")
            
        except FileNotFoundError:
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
        
        # Index TF-IDF
        questions = [pair["input"] for pair in self.qa_pairs]
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.question_vectors = self.vectorizer.fit_transform(questions)
        print("✅ Index de recherche créé!")
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Recherche contextes pertinents"""
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
    
    def generate_developed_response(self, user_question):
        """Génère une réponse DÉVELOPPÉE et COMPLÈTE"""
        
        # 1. Rechercher contextes
        relevant_contexts = self.retrieve_relevant_context(user_question, top_k=3)
        
        # 2. Construire contexte riche
        context_info = ""
        if relevant_contexts:
            context_info = "\n\nRelevant banking information:\n"
            for i, ctx in enumerate(relevant_contexts):
                context_info += f"Source {i+1}: {ctx['answer']}\n"
        
        # 3. Prompt pour réponse DÉVELOPPÉE
        developed_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert banking consultant. Your task is to provide comprehensive, detailed, and well-structured answers to banking questions. Always:

1. Give complete information, not just basic facts
2. Explain the context and implications
3. Provide specific details like amounts, limits, conditions
4. Structure your response clearly
5. Be thorough and professional

Use the provided banking information to give a complete answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_question}{context_info}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Based on the banking information provided, I'll give you a comprehensive answer:

"""
        
        # 4. Génération avec paramètres pour réponses longues
        inputs = self.tokenizer(developed_prompt, return_tensors="pt", truncation=True, max_length=1500)
        
        try:
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except:
            pass
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,  # Plus long pour réponses développées
                temperature=0.4,     # Plus créatif pour développement
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.2,  # Encourage les réponses longues
                no_repeat_ngram_size=3
            )
        
        # 5. Extraction propre
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire la réponse de l'assistant
        if "Based on the banking information provided, I'll give you a comprehensive answer:" in full_response:
            response = full_response.split("Based on the banking information provided, I'll give you a comprehensive answer:")[-1]
        elif "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            response = full_response
        
        # Nettoyage
        response = response.replace("<|eot_id|>", "").strip()
        
        # Si réponse trop courte, développer avec le contexte
        if len(response.split()) < 20 and relevant_contexts:
            # Créer une réponse développée manuellement
            best_context = relevant_contexts[0]
            response = self.develop_response_manually(user_question, best_context)
        
        return response, relevant_contexts
    
    def develop_response_manually(self, question, context):
        """Développe manuellement une réponse si nécessaire"""
        
        answer = context['answer']
        
        # Patterns pour développer selon le type de question
        if "limit" in question.lower():
            developed = f"Regarding your question about limits, here are the specific details:\n\n"
            developed += f"{answer}\n\n"
            developed += "These limits are designed to ensure secure banking operations while providing flexibility for your financial needs. "
            developed += "Please note that these limits may be subject to regulatory requirements and bank policies."
            
        elif "charge" in question.lower() or "fee" in question.lower():
            developed = f"Here are the detailed charges and fees information:\n\n"
            developed += f"{answer}\n\n"
            developed += "These charges are structured to provide transparent pricing for banking services. "
            developed += "For any specific fee inquiries or waiver requests, please contact your relationship manager."
            
        elif "facility" in question.lower() or "service" in question.lower():
            developed = f"The available facilities and services include:\n\n"
            developed += f"{answer}\n\n"
            developed += "These facilities are designed to enhance your banking experience and provide convenient access to essential services. "
            developed += "Additional premium services may be available upon request."
            
        else:
            # Réponse générale développée
            developed = f"Based on the banking information available:\n\n"
            developed += f"{answer}\n\n"
            developed += "This information reflects current banking policies and may be subject to updates. "
            developed += "For the most current information or specific inquiries, please contact customer service."
        
        return developed

# Initialisation
print("🚀 Initialisation système RAG développé...")
rag_system = DevelopedRAGSystem()
rag_system.load_model()
rag_system.load_clean_knowledge_base()

print("\n" + "="*60)
print("🎯 SYSTÈME RAG DÉVELOPPÉ PRÊT!")
print("="*60)

# Tests automatiques
test_questions = [
    "Can applicant avail clean loan in NUST Sahar Finance?",
    "Are there any Credit and Debit limits in NUST Freelancer Digital Account?",
    "What are the Loan Limits of NUST Rice Finance?",
    "What are the free facilities associated with Roshan Digital Account?"
]

for i, question in enumerate(test_questions):
    print(f"\n{'='*15} TEST {i+1} {'='*15}")
    print(f"❓ Question: {question}")
    
    start_time = time.time()
    response, contexts = rag_system.generate_developed_response(question)
    response_time = time.time() - start_time
    
    print(f"⏱️ Temps: {response_time:.2f}s")
    print(f"🔍 Contextes: {len(contexts)}")
    if contexts:
        print(f"📊 Similarité: {max(ctx['similarity'] for ctx in contexts):.2%}")
    
    print(f"\n💬 RÉPONSE DÉVELOPPÉE:")
    print(f"{response}")

# Mode interactif
print(f"\n💬 MODE INTERACTIF DÉVELOPPÉ (tapez 'quit' pour arrêter):")

while True:
    try:
        user_question = input("\n❓ Votre question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'stop', 'q']:
            break
            
        if not user_question:
            continue
        
        print("🎯 Génération réponse développée...")
        start_time = time.time()
        response, contexts = rag_system.generate_developed_response(user_question)
        response_time = time.time() - start_time
        
        print(f"⏱️ Temps: {response_time:.2f}s")
        print(f"🔍 Contextes: {len(contexts)}")
        if contexts:
            print(f"📊 Similarité: {max(ctx['similarity'] for ctx in contexts):.2%}")
        
        print(f"\n💬 RÉPONSE DÉVELOPPÉE:")
        print(f"{response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")

print(f"\n🎉 SESSION TERMINÉE!")
print(f"✅ RAG avec réponses 100% développées!")
