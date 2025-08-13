# 📁 Fichiers Essentiels du Projet BankBot AI

## ✅ Fichiers à GARDER

### Backend Python
- `backend_rag_api.py` - API principale RAG
- `backend_chat_history.py` - Gestion historique
- `banking_chatbot_final.py` - Chatbot principal
- `rag_chat_simple.py` - Version simple RAG
- `chatbot_rag_complete.py` - Version complète
- `requirements.txt` - Dépendances Python

### Base de connaissances
- `banking_knowledge_base.json` - Données bancaires
- Autres fichiers `.json` de configuration

### Frontend React
- `chat-bank-nexus-main(frontend v0)/src/` - Code source
- `chat-bank-nexus-main(frontend v0)/public/` - Assets publics
- `chat-bank-nexus-main(frontend v0)/package.json` - Dépendances
- `chat-bank-nexus-main(frontend v0)/package-lock.json` - Lock file
- `chat-bank-nexus-main(frontend v0)/tsconfig.json` - Config TypeScript
- `chat-bank-nexus-main(frontend v0)/tailwind.config.ts` - Config Tailwind
- `chat-bank-nexus-main(frontend v0)/vite.config.ts` - Config Vite
- `chat-bank-nexus-main(frontend v0)/index.html` - Page principale

### Documentation
- `README.md` - Documentation principale
- `FICHIERS_ESSENTIELS.md` - Ce fichier
- Autres fichiers `.md`

### Configuration
- `.gitignore` - Exclusions Git
- `cleanup_for_transfer.bat` - Script de nettoyage

## ❌ Fichiers à SUPPRIMER

### Dépendances (peuvent être réinstallées)
- `node_modules/` - Dépendances Node.js (npm install)
- `__pycache__/` - Cache Python
- `.pytest_cache/` - Cache tests

### Fichiers temporaires
- `*.log` - Logs
- `*.tmp`, `*.temp` - Fichiers temporaires
- `conversation_*.json` - Conversations sauvegardées
- `*.db`, `*.sqlite` - Bases de données locales

### Builds
- `dist/`, `build/` - Builds compilés
- `.cache/` - Cache de build

### IDE
- `.vscode/`, `.idea/` - Configuration IDE
- `*.swp`, `*.swo` - Fichiers temporaires éditeur

### Modèles volumineux (si présents)
- `*.bin`, `*.safetensors` - Modèles binaires
- `*.gguf`, `*.model` - Modèles Ollama
- `*.pkl`, `*.h5` - Modèles ML

## 📊 Taille estimée après nettoyage

- **Avant nettoyage** : ~32 GB
- **Après nettoyage** : ~50-100 MB
- **Réduction** : 99%+ 

## 🚀 Instructions d'installation pour le destinataire

1. **Backend** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Frontend** :
   ```bash
   cd "chat-bank-nexus-main(frontend v0)"
   npm install
   ```

3. **Ollama** (à installer séparément) :
   ```bash
   ollama pull llama3.1:8b
   ```

## 📦 Prêt pour Swiss Transfer !

Après nettoyage, le projet sera optimisé pour le partage via Swiss Transfer.
