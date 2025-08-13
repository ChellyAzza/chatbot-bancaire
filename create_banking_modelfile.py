"""
Création d'un Modelfile Ollama pour chatbot bancaire
Utilise vos données wasifis/bank-assistant-qa
"""

import json
import os

def load_sample_banking_data():
    """Charge des exemples de données bancaires"""
    try:
        with open('processed_data/train.json', 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        print(f"Données chargées: {len(data)} exemples")
        return data[:10]  # Prendre 10 exemples pour le Modelfile
    except Exception as e:
        print(f"Erreur de chargement: {e}")
        # Données de fallback
        return [
            {
                "input": "Quels sont les frais de tenue de compte?",
                "output": "Les frais de tenue de compte varient selon le type de compte. Pour un compte courant standard, les frais sont de 5 euros par mois. Pour un compte épargne, il n'y a généralement pas de frais de tenue."
            },
            {
                "input": "Comment ouvrir un compte bancaire?",
                "output": "Pour ouvrir un compte bancaire, vous devez fournir une pièce d'identité valide, un justificatif de domicile de moins de 3 mois, et vos 3 derniers bulletins de salaire. Un dépôt minimum peut être requis selon le type de compte."
            },
            {
                "input": "Quelles sont les conditions pour un prêt?",
                "output": "Les conditions pour un prêt incluent: avoir des revenus stables, un taux d'endettement inférieur à 33%, un apport personnel pour les prêts immobiliers, et un bon historique de crédit. L'âge et la situation professionnelle sont également pris en compte."
            }
        ]

def create_banking_modelfile():
    """Crée le Modelfile pour le chatbot bancaire"""
    print("Création du Modelfile bancaire...")
    
    # Charger les données
    examples = load_sample_banking_data()
    
    # Créer le contenu du Modelfile
    modelfile_content = """FROM llama3.1:8b

# Paramètres optimisés pour le banking
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Prompt système spécialisé banking
SYSTEM \"\"\"Vous êtes un assistant bancaire expert et professionnel. 
Vous travaillez pour une banque et aidez les clients avec leurs questions bancaires.

VOTRE EXPERTISE INCLUT:
- Comptes bancaires (courant, épargne, joint)
- Cartes bancaires (débit, crédit, prépayées)
- Prêts et crédits (personnel, immobilier, auto)
- Services bancaires (virements, prélèvements)
- Frais et tarifs bancaires
- Procédures administratives

STYLE DE RÉPONSE:
- Professionnel et courtois
- Précis et factuel
- Structuré avec des points clés
- Inclure les informations sur les frais quand pertinent
- Proposer des solutions adaptées au client

Répondez toujours de manière claire et complète aux questions bancaires.\"\"\"

# Template de conversation
TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"
"""
    
    # Sauvegarder le Modelfile
    with open("Modelfile.banking", 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print("Modelfile.banking créé avec succès!")
    return "Modelfile.banking"

def create_training_script():
    """Crée le script pour créer le modèle"""
    script_content = """@echo off
echo Creation du modele bancaire personnalise...
echo.

echo Verification d'Ollama...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo Erreur: Ollama n'est pas demarre
    echo Lancez 'ollama serve' d'abord
    pause
    exit /b 1
)

echo Creation du modele banking-assistant...
ollama create banking-assistant -f Modelfile.banking

if %errorlevel% equ 0 (
    echo.
    echo Modele 'banking-assistant' cree avec succes!
    echo.
    echo Test du modele:
    echo.
    ollama run banking-assistant "Quels sont les frais de compte?"
) else (
    echo Erreur lors de la creation du modele
)

echo.
pause
"""
    
    with open("create_banking_model.bat", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("Script create_banking_model.bat créé!")

def create_test_script():
    """Crée un script de test du modèle"""
    test_script = '''"""
Test du modèle bancaire personnalisé
"""

import ollama

def test_banking_model():
    print("Test du modèle banking-assistant")
    print("=" * 40)
    
    client = ollama.Client()
    
    # Vérifier que le modèle existe
    try:
        models = client.list()
        model_names = [m.get('name', '') for m in models['models']]
        
        if 'banking-assistant:latest' not in model_names:
            print("Erreur: Modèle banking-assistant non trouvé")
            print("Exécutez d'abord: create_banking_model.bat")
            return
        
        print("Modèle banking-assistant trouvé!")
        
    except Exception as e:
        print(f"Erreur de connexion Ollama: {e}")
        return
    
    # Questions de test
    test_questions = [
        "Quels sont les frais de tenue de compte?",
        "Comment ouvrir un compte épargne?",
        "Quelles sont les conditions pour un prêt immobilier?",
        "Comment activer ma carte bancaire?",
        "Que faire en cas de perte de carte?",
        "Quels sont les horaires d'ouverture?",
        "Comment faire un virement international?",
        "Quels documents pour ouvrir un compte?"
    ]
    
    print("\\nTest avec questions bancaires:")
    print("-" * 40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\n{i}. Question: {question}")
        
        try:
            response = client.chat(
                model='banking-assistant',
                messages=[
                    {'role': 'user', 'content': question}
                ]
            )
            
            answer = response['message']['content']
            print(f"   Réponse: {answer[:150]}...")
            
        except Exception as e:
            print(f"   Erreur: {e}")
    
    print("\\n" + "=" * 40)
    print("Test terminé!")

def compare_models():
    """Compare le modèle de base et le modèle personnalisé"""
    print("\\nComparaison des modèles")
    print("=" * 30)
    
    client = ollama.Client()
    test_question = "Quels sont les frais de compte?"
    
    print(f"Question: {test_question}")
    
    # Modèle de base
    print("\\nModèle de base (llama3.1:8b):")
    try:
        response = client.chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': test_question}]
        )
        print(response['message']['content'][:200] + "...")
    except:
        print("Erreur avec le modèle de base")
    
    # Modèle personnalisé
    print("\\nModèle personnalisé (banking-assistant):")
    try:
        response = client.chat(
            model='banking-assistant',
            messages=[{'role': 'user', 'content': test_question}]
        )
        print(response['message']['content'][:200] + "...")
    except:
        print("Erreur avec le modèle personnalisé")

if __name__ == "__main__":
    test_banking_model()
    compare_models()
'''
    
    with open("test_banking_model.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("Script test_banking_model.py créé!")

def main():
    """Fonction principale"""
    print("Création du Modelfile Ollama pour chatbot bancaire")
    print("=" * 50)
    
    # 1. Créer le Modelfile
    modelfile = create_banking_modelfile()
    
    # 2. Créer le script de création
    create_training_script()
    
    # 3. Créer le script de test
    create_test_script()
    
    print("\n" + "=" * 50)
    print("FICHIERS CRÉÉS:")
    print("✓ Modelfile.banking - Configuration du modèle")
    print("✓ create_banking_model.bat - Script de création")
    print("✓ test_banking_model.py - Script de test")
    
    print("\nÉTAPES SUIVANTES:")
    print("1. Exécutez: create_banking_model.bat")
    print("2. Testez avec: python test_banking_model.py")
    print("3. Utilisez: ollama run banking-assistant")
    
    print("\nCe modèle utilisera:")
    print("✓ Votre llama3.1:8b existant comme base")
    print("✓ Prompt système optimisé pour le banking")
    print("✓ Paramètres ajustés pour les réponses bancaires")
    print("✓ Aucun téléchargement depuis Hugging Face requis")

if __name__ == "__main__":
    main()
