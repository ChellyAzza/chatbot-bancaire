# 📦 Récapitulatif Complet des Dépendances

## 🎯 Vue d'Ensemble

Ce document liste **toutes les dépendances** utilisées dans le projet de chatbot bancaire, organisées par composant et usage.

## 📊 Statistiques Globales

- **Total dépendances** : 30 packages
- **Taille totale** : ~3.8GB
- **Dépendances critiques** : 20/30 (67%)
- **Temps installation** : 30-40 minutes (complet)

## 🔧 Backend Python

### API & Serveur (4 packages - ~100MB)
```txt
fastapi==0.104.1          # Framework API REST moderne
uvicorn==0.24.0           # Serveur ASGI haute performance
pydantic==2.4.2           # Validation et sérialisation données
python-multipart==0.0.6   # Support upload fichiers
```

### IA & Machine Learning (6 packages - ~3.2GB)
```txt
torch==2.1.0+cu118        # Framework deep learning avec CUDA
transformers==4.35.0      # Modèles Hugging Face (Llama, BERT, etc.)
peft==0.6.0              # Parameter Efficient Fine-Tuning (LoRA)
accelerate==0.24.0        # Optimisation multi-GPU et mixed precision
bitsandbytes==0.41.0      # Quantification 4-bit et 8-bit
datasets==2.14.1          # Gestion datasets Hugging Face
```

### Traitement Données (4 packages - ~305MB)
```txt
scikit-learn==1.3.0       # TF-IDF, cosine similarity, ML classique
numpy==1.24.3             # Calculs numériques et arrays
pandas==2.0.3             # Manipulation et analyse données
sentence-transformers==2.2.2  # Embeddings sémantiques (futur)
```

### Training & Monitoring (3 packages - ~115MB)
```txt
trl==0.7.4                # Reinforcement Learning from Human Feedback
wandb==0.15.12            # Monitoring et logging entraînement
tensorboard==2.14.1       # Visualisation métriques TensorFlow
```

## 🎨 Frontend Node.js

### Framework & Build (4 packages - ~67MB)
```json
{
  "react": "18.2.0",           // Framework UI déclaratif
  "typescript": "5.0.2",       // Typage statique JavaScript
  "vite": "4.4.5",            // Build tool ultra-rapide
  "@types/react": "18.2.15"    // Types TypeScript pour React
}
```

### Interface Utilisateur (8 packages - ~19MB)
```json
{
  "tailwindcss": "3.3.0",                    // Framework CSS utility-first
  "@radix-ui/react-avatar": "1.0.4",         // Composant avatar accessible
  "@radix-ui/react-button": "1.0.3",         // Composant bouton accessible
  "@radix-ui/react-scroll-area": "1.0.5",    // Zone défilement personnalisée
  "@radix-ui/react-toast": "1.1.4",          // Notifications toast
  "lucide-react": "0.263.1",                 // Icônes SVG modernes
  "class-variance-authority": "0.7.0",        // Variants CSS conditionnels
  "clsx": "2.0.0"                            // Utilitaire classes CSS
}
```

## 🎤 Fonctionnalités Vocales

### APIs Natives (1 API - 0MB)
```javascript
// Web Speech API - Natif navigateur
window.SpeechRecognition     // Speech-to-Text
window.speechSynthesis       // Text-to-Speech
```

## 📋 Matrices de Compatibilité

### Versions Python
| Package | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 |
|---------|-------------|-------------|--------------|--------------|
| torch | ✅ | ✅ | ✅ | ✅ |
| transformers | ✅ | ✅ | ✅ | ✅ |
| fastapi | ✅ | ✅ | ✅ | ✅ |
| peft | ✅ | ✅ | ✅ | ⚠️ |

### Versions Node.js
| Package | Node 16 | Node 18 | Node 20 | Node 21 |
|---------|---------|---------|---------|---------|
| react | ✅ | ✅ | ✅ | ✅ |
| vite | ✅ | ✅ | ✅ | ✅ |
| typescript | ✅ | ✅ | ✅ | ✅ |

## 🔍 Analyse des Dépendances

### Par Criticité
- **Critiques (20)** : Nécessaires au fonctionnement
- **Importantes (7)** : Améliorent l'expérience
- **Optionnelles (3)** : Fonctionnalités avancées

### Par Taille
- **Très lourdes (>1GB)** : torch (2.5GB)
- **Lourdes (100MB-1GB)** : transformers (450MB)
- **Moyennes (10-100MB)** : scikit-learn (85MB)
- **Légères (<10MB)** : fastapi (65MB)

### Par Fréquence d'Usage
- **Constante** : react, fastapi, torch
- **Fréquente** : transformers, numpy
- **Occasionnelle** : peft, accelerate
- **Rare** : wandb, tensorboard

## 🚀 Optimisations Possibles

### Réduction Taille
```bash
# Installation minimale (3.2GB au lieu de 3.8GB)
pip install -r requirements-backend-minimal.txt

# Frontend léger (70MB au lieu de 150MB)
npm install --production
```

### Installation Parallèle
```bash
# Backend en parallèle
pip install torch transformers &
pip install fastapi uvicorn &
wait

# Frontend optimisé
npm ci --prefer-offline
```

### Cache Docker
```dockerfile
# Cache layers pour réutilisation
FROM python:3.9-slim
COPY requirements-backend-minimal.txt .
RUN pip install -r requirements-backend-minimal.txt
COPY . .
```

## 📈 Évolution Prévue

### Ajouts Futurs
- **sentence-transformers** : Embeddings avancés
- **chromadb** : Base vectorielle
- **redis** : Cache distribué
- **prometheus** : Métriques production

### Suppressions Possibles
- **pandas** : Si pas d'analyse données
- **tensorboard** : Si monitoring externe
- **wandb** : Si pas de tracking ML

---

**📅 Dernière mise à jour** : 15 Janvier 2025  
**🔧 Version** : 1.0  
**📊 Total packages** : 30  
**💾 Taille totale** : ~3.8GB
