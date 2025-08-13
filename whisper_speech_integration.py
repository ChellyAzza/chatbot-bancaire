"""
Intégration de Whisper pour la reconnaissance vocale
Alternative professionnelle à Web Speech API
"""

import os
import io
import wave
import tempfile
from typing import Optional, Dict, Any
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Option 1: Whisper local (pour développement)
try:
    import whisper
    WHISPER_LOCAL_AVAILABLE = True
    print("✅ Whisper local disponible")
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False
    print("⚠️ Whisper local non installé")

# Option 2: OpenAI API (pour production)
try:
    import openai
    OPENAI_API_AVAILABLE = True
    print("✅ OpenAI API disponible")
except ImportError:
    OPENAI_API_AVAILABLE = False
    print("⚠️ OpenAI API non installé")

class WhisperSpeechRecognition:
    def __init__(self, use_local: bool = True, model_size: str = "base"):
        """
        Initialise le système de reconnaissance vocale Whisper
        
        Args:
            use_local: Utiliser Whisper local (True) ou OpenAI API (False)
            model_size: Taille du modèle local ("tiny", "base", "small", "medium", "large")
        """
        self.use_local = use_local
        self.model_size = model_size
        self.model = None
        
        if use_local and WHISPER_LOCAL_AVAILABLE:
            self._load_local_model()
        elif not use_local and OPENAI_API_AVAILABLE:
            self._setup_openai_api()
        else:
            raise Exception("Aucun backend Whisper disponible")
    
    def _load_local_model(self):
        """Charge le modèle Whisper local"""
        try:
            print(f"🔄 Chargement du modèle Whisper {self.model_size}...")
            self.model = whisper.load_model(self.model_size)
            print(f"✅ Modèle Whisper {self.model_size} chargé")
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def _setup_openai_api(self):
        """Configure l'API OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY non définie")
        
        openai.api_key = api_key
        print("✅ OpenAI API configurée")
    
    def transcribe_audio(self, audio_file_path: str, language: str = "fr") -> Dict[str, Any]:
        """
        Transcrit un fichier audio
        
        Args:
            audio_file_path: Chemin vers le fichier audio
            language: Code langue (fr, en, etc.)
            
        Returns:
            Dict avec le texte transcrit et métadonnées
        """
        try:
            if self.use_local:
                return self._transcribe_local(audio_file_path, language)
            else:
                return self._transcribe_api(audio_file_path, language)
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def _transcribe_local(self, audio_file_path: str, language: str) -> Dict[str, Any]:
        """Transcription avec modèle local"""
        result = self.model.transcribe(
            audio_file_path,
            language=language,
            task="transcribe"  # ou "translate" pour traduire en anglais
        )
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language", language),
            "segments": result.get("segments", []),
            "success": True,
            "method": "local",
            "model": self.model_size
        }
    
    def _transcribe_api(self, audio_file_path: str, language: str) -> Dict[str, Any]:
        """Transcription avec API OpenAI"""
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language=language
            )
        
        return {
            "text": transcript["text"].strip(),
            "language": language,
            "success": True,
            "method": "api",
            "model": "whisper-1"
        }
    
    def translate_to_english(self, audio_file_path: str) -> Dict[str, Any]:
        """Traduit l'audio directement en anglais"""
        try:
            if self.use_local:
                result = self.model.transcribe(
                    audio_file_path,
                    task="translate"  # Traduit directement en anglais
                )
                return {
                    "text": result["text"].strip(),
                    "original_language": result.get("language", "unknown"),
                    "success": True,
                    "method": "local_translate"
                }
            else:
                with open(audio_file_path, "rb") as audio_file:
                    transcript = openai.Audio.translate(
                        model="whisper-1",
                        file=audio_file
                    )
                return {
                    "text": transcript["text"].strip(),
                    "success": True,
                    "method": "api_translate"
                }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False
            }

# API FastAPI pour intégration avec le frontend
app = FastAPI(title="Whisper Speech Recognition API")

# Configuration CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite/React ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du système Whisper
try:
    # Essayer d'abord le modèle local
    speech_recognizer = WhisperSpeechRecognition(use_local=True, model_size="base")
except:
    try:
        # Fallback vers l'API OpenAI
        speech_recognizer = WhisperSpeechRecognition(use_local=False)
    except:
        speech_recognizer = None
        print("❌ Aucun backend Whisper disponible")

@app.post("/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...),
    language: str = "fr"
):
    """
    Endpoint pour transcrire un fichier audio
    """
    if not speech_recognizer:
        raise HTTPException(status_code=500, detail="Service de reconnaissance vocale indisponible")
    
    # Vérifier le type de fichier
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Fichier audio requis")
    
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcrire
        result = speech_recognizer.transcribe_audio(temp_file_path, language)
        
        # Nettoyer le fichier temporaire
        os.unlink(temp_file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(e)}")

@app.post("/translate")
async def translate_audio_endpoint(audio: UploadFile = File(...)):
    """
    Endpoint pour traduire un fichier audio en anglais
    """
    if not speech_recognizer:
        raise HTTPException(status_code=500, detail="Service de reconnaissance vocale indisponible")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        result = speech_recognizer.translate_to_english(temp_file_path)
        os.unlink(temp_file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traduction: {str(e)}")

@app.get("/health")
async def health_check():
    """Vérification de l'état du service"""
    return {
        "status": "healthy",
        "whisper_local": WHISPER_LOCAL_AVAILABLE,
        "openai_api": OPENAI_API_AVAILABLE,
        "active_backend": "local" if speech_recognizer and speech_recognizer.use_local else "api"
    }

@app.get("/models")
async def available_models():
    """Liste des modèles disponibles"""
    models = {
        "local_models": [
            {"name": "tiny", "size": "~39 MB", "speed": "Très rapide", "quality": "Basique"},
            {"name": "base", "size": "~74 MB", "speed": "Rapide", "quality": "Bonne"},
            {"name": "small", "size": "~244 MB", "speed": "Moyen", "quality": "Très bonne"},
            {"name": "medium", "size": "~769 MB", "speed": "Lent", "quality": "Excellente"},
            {"name": "large", "size": "~1550 MB", "speed": "Très lent", "quality": "Parfaite"}
        ],
        "api_models": [
            {"name": "whisper-1", "description": "Modèle OpenAI optimisé"}
        ]
    }
    return models

if __name__ == "__main__":
    print("🚀 Démarrage du serveur Whisper...")
    print("📝 Documentation: http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002)
