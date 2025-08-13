# 🚀 Guide de Démarrage Complet - Chatbot Bancaire Professionnel

## 📋 **Prérequis Vérifiés**
- ✅ Python 3.13.5 installé
- ✅ Node.js pour le frontend React
- ✅ Toutes les dépendances installées

## 🔧 **Installation Finale (Si pas encore fait)**

```bash
# Vérification Python
python --version  # Doit afficher Python 3.13.5

# Installation des packages
pip install openai-whisper gtts fastapi uvicorn python-multipart requests pydantic

# Vérification des installations
python -c "import whisper; print('✅ Whisper OK')"
python -c "from gtts import gTTS; print('✅ gTTS OK')"
python -c "import fastapi; print('✅ FastAPI OK')"
```

## 🎯 **Démarrage Automatique (Recommandé)**

### Option 1: Démarrage Coordonné
```bash
# Démarrage automatique de tous les services
python start_professional_chatbot.py
```

### Option 2: Démarrage Manuel (pour debug)
```bash
# Terminal 1: Service Speech (Whisper + gTTS)
python compatible_speech_service.py

# Terminal 2: Service RAG
python rag_developed_responses.py

# Terminal 3: Frontend React
cd "chat-bank-nexus-main(frontend v0)"
npm install
npm run dev
```

## 🌐 **URLs d'Accès**

Une fois tous les services démarrés :

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend Principal** | http://localhost:5173 | Interface utilisateur complète |
| **API Speech** | http://localhost:8004/docs | Documentation Whisper + gTTS |
| **API RAG** | http://localhost:8000/docs | Documentation chatbot |
| **Test Speech** | http://localhost:8004/speech/test | Test rapide du service |

## 🎤 **Fonctionnalités Intégrées**

### ✅ **1. Questions d'Action Rapide par Catégorie**
- Cliquez sur une catégorie dans les actions rapides
- Sélectionnez une question suggérée
- Questions contextuelles bancaires

### ✅ **2. Historisation des Discussions**
- Bouton "Historique" dans l'en-tête
- Sauvegarde automatique dans localStorage
- Recherche et gestion des conversations

### ✅ **3. Speech-to-Text (Whisper)**
- Cliquez sur le microphone dans le champ de saisie
- Dictez votre question en français
- Transcription optimisée pour le vocabulaire bancaire

### ✅ **4. Text-to-Speech (gTTS)**
- Contrôles vocaux sous la zone de chat
- Lecture automatique des réponses (optionnel)
- Lecture manuelle de la dernière réponse

## 🔍 **Vérification du Fonctionnement**

### 1. **Vérifiez les Statuts de Connexion**
Dans l'interface, vous devriez voir :
- 🟢 Connecté (RAG)
- 🦙 Modèle chargé
- 🚀 GPU ou 💻 CPU
- 🎤🔊 Speech (service vocal)

### 2. **Test Speech-to-Text**
1. Cliquez sur le microphone dans le champ de saisie
2. Dites : "Quels sont les frais de virement NUST ?"
3. Vérifiez que le texte apparaît dans le champ

### 3. **Test Text-to-Speech**
1. Envoyez une question au chatbot
2. Cliquez sur le bouton haut-parleur dans les contrôles vocaux
3. Écoutez la réponse synthétisée

### 4. **Test Actions Rapides**
1. Cliquez sur "Comptes NUST" dans les actions rapides
2. Sélectionnez une question suggérée
3. Vérifiez que la question est envoyée automatiquement

### 5. **Test Historique**
1. Cliquez sur l'icône "Historique" dans l'en-tête
2. Vérifiez que vos conversations sont sauvegardées
3. Testez le chargement d'une conversation précédente

## 🛠️ **Dépannage**

### Problème: Service Speech indisponible
```bash
# Vérifiez que le service est démarré
curl http://localhost:8004/speech/health

# Redémarrez le service si nécessaire
python compatible_speech_service.py
```

### Problème: RAG déconnecté
```bash
# Vérifiez le service RAG
curl http://localhost:8000/health

# Redémarrez si nécessaire
python rag_developed_responses.py
```

### Problème: Frontend ne se connecte pas
```bash
# Dans le dossier frontend
cd "chat-bank-nexus-main(frontend v0)"
npm install
npm run dev
```

## 📊 **Performances Attendues**

### Speech-to-Text (Whisper)
- **Temps de transcription :** 2-5 secondes
- **Qualité :** Excellente pour le français
- **Vocabulaire bancaire :** Optimisé

### Text-to-Speech (gTTS)
- **Temps de synthèse :** 1-3 secondes
- **Qualité :** Bonne voix française
- **Connexion requise :** Oui (gTTS utilise Google)

### Interface Utilisateur
- **Temps de réponse :** < 1 seconde
- **Historique :** Sauvegarde instantanée
- **Actions rapides :** Réponse immédiate

## 🎉 **Fonctionnalités Avancées**

### Corrections Automatiques Bancaires
Le système corrige automatiquement :
- "viremant" → "virement"
- "conte" → "compte"
- "nust" → "NUST"
- "pmyb" → "PMYB"

### Optimisations TTS
- Acronymes épelés : "NUST" → "N U S T"
- Symboles convertis : "€" → "euros"
- Pauses naturelles ajoutées

### Gestion d'Erreurs
- Fallbacks gracieux si services indisponibles
- Messages d'erreur informatifs
- Retry automatique des connexions

## 📞 **Support et Maintenance**

### Logs de Debug
- Service Speech : Logs dans le terminal
- Service RAG : Logs dans le terminal
- Frontend : Console du navigateur (F12)

### Mise à Jour
Pour mettre à jour les modèles :
```bash
# Mise à jour Whisper
pip install --upgrade openai-whisper

# Mise à jour autres packages
pip install --upgrade gtts fastapi uvicorn
```

## 🏆 **Résultat Final**

Votre chatbot bancaire dispose maintenant de :

1. ✅ **Interface utilisateur complète** avec React + TypeScript
2. ✅ **Reconnaissance vocale professionnelle** avec Whisper
3. ✅ **Synthèse vocale naturelle** avec gTTS
4. ✅ **Historisation complète** des conversations
5. ✅ **Actions rapides intelligentes** par catégorie
6. ✅ **Monitoring en temps réel** des services
7. ✅ **Optimisations bancaires** spécialisées
8. ✅ **Démarrage automatisé** de tous les services

**🎯 Qualité professionnelle, 100% gratuit, compatible Python 3.13 !**
