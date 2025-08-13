#!/bin/bash
# Script de fine-tuning Ollama pour chatbot bancaire

echo "🏦 Fine-tuning Ollama - Chatbot Bancaire"
echo "========================================"

# Vérifier qu'Ollama est démarré
echo "1. Vérification d'Ollama..."
ollama list > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Ollama n'est pas démarré. Lancez 'ollama serve' d'abord."
    exit 1
fi
echo "✅ Ollama est actif"

# Créer le modèle fine-tuné
echo "2. Création du modèle fine-tuné..."
ollama create banking-assistant -f Modelfile

if [ $? -eq 0 ]; then
    echo "✅ Modèle 'banking-assistant' créé avec succès!"
    echo ""
    echo "🎯 Pour utiliser le modèle:"
    echo "ollama run banking-assistant"
    echo ""
    echo "🧪 Test rapide:"
    ollama run banking-assistant "Quels sont les frais de tenue de compte?"
else
    echo "❌ Erreur lors de la création du modèle"
    exit 1
fi
