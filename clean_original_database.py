"""
Nettoie la base de données originale wasifis/bank-assistant-qa
Garde seulement les colonnes input et output, supprime instruction
"""

from datasets import load_dataset
import json

print("🧹 NETTOYAGE BASE DE DONNÉES ORIGINALE")
print("=" * 50)

# Charger le dataset original
print("📊 Chargement wasifis/bank-assistant-qa...")
dataset = load_dataset("wasifis/bank-assistant-qa")
train_data = dataset["train"]

print(f"✅ {len(train_data)} exemples chargés")

# Examiner la structure
print("\n🔍 STRUCTURE DU DATASET:")
if len(train_data) > 0:
    sample = train_data[0]
    print("Colonnes disponibles:", list(sample.keys()))
    print("\nExemple:")
    for key, value in sample.items():
        print(f"  {key}: {str(value)[:100]}...")

# Nettoyer les données
print("\n🧹 NETTOYAGE EN COURS...")
cleaned_data = []

for i, item in enumerate(train_data):
    # Garder seulement input et output
    cleaned_item = {
        "input": item["input"].strip(),
        "output": item["output"].strip()
    }

    # Vérifier que les champs ne sont pas vides
    if cleaned_item["input"] and cleaned_item["output"]:
        cleaned_data.append(cleaned_item)

    if (i + 1) % 1000 == 0:
        print(f"  Traité: {i + 1}/{len(train_data)}")

print(f"✅ Nettoyage terminé: {len(cleaned_data)} exemples valides")

# Sauvegarder en JSON
output_file = "cleaned_banking_qa.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print(f"💾 Sauvegardé: {output_file}")

# Afficher quelques exemples nettoyés
print(f"\n📋 EXEMPLES NETTOYÉS:")
for i in range(min(3, len(cleaned_data))):
    print(f"\nExemple {i+1}:")
    print(f"  Input: {cleaned_data[i]['input'][:100]}...")
    print(f"  Output: {cleaned_data[i]['output'][:100]}...")

print(f"\n✅ BASE DE DONNÉES NETTOYÉE PRÊTE!")
print(f"📁 Fichier créé: {output_file}")
print(f"🎯 Utilisez ce fichier pour un RAG plus précis!")
