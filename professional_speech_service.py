"""
Service Speech professionnel pour le chatbot bancaire
Intègre Whisper (STT) et Coqui TTS de façon optimisée
"""

import os
import io
import tempfile
import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import whisper
from TTS.api import TTS
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    voice_speed: Optional[float] = 1.0
    voice_pitch: Optional[float] = 1.0

class SpeechService:
    def __init__(self):
        self.whisper_model = None
        self.tts_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_files = []
        
        logger.info(f"🎯 Initialisation Speech Service sur {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Charge les modèles Whisper et TTS de façon optimisée"""
        try:
            # Charger Whisper (modèle base pour équilibre vitesse/qualité)
            logger.info("📥 Chargement Whisper...")
            self.whisper_model = whisper.load_model("base", device=self.device)
            logger.info("✅ Whisper chargé avec succès")
            
            # Charger Coqui TTS français
            logger.info("📥 Chargement Coqui TTS...")
            self.tts_model = TTS(model_name="tts_models/fr/css10/vits", progress_bar=False)
            if torch.cuda.is_available():
                self.tts_model = self.tts_model.to(self.device)
            logger.info("✅ Coqui TTS chargé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles: {e}")
            raise
    
    async def transcribe_audio(self, audio_file: UploadFile) -> Dict[str, Any]:
        """
        Transcrit un fichier audio avec Whisper
        Optimisé pour les questions bancaires
        """
        try:
            # Validation du fichier
            if not audio_file.content_type.startswith("audio/"):
                raise ValueError("Fichier audio requis")
            
            # Sauvegarder temporairement
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                self.temp_files.append(temp_path)
            
            # Transcription avec Whisper
            logger.info("🎤 Transcription en cours...")
            result = self.whisper_model.transcribe(
                temp_path,
                language="fr",  # Français pour le contexte bancaire
                task="transcribe",
                fp16=torch.cuda.is_available(),  # Optimisation GPU
                verbose=False
            )
            
            # Nettoyage et validation
            transcript = result["text"].strip()
            confidence = self._calculate_confidence(result)
            
            # Post-traitement pour le contexte bancaire
            transcript = self._enhance_banking_transcript(transcript)
            
            logger.info(f"✅ Transcription réussie: {transcript[:50]}...")
            
            return {
                "success": True,
                "transcript": transcript,
                "confidence": confidence,
                "language": result.get("language", "fr"),
                "processing_time": len(content) / 1000,  # Estimation
                "model": "whisper-base"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur transcription: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcript": ""
            }
        finally:
            # Nettoyage du fichier temporaire
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
                    if temp_path in self.temp_files:
                        self.temp_files.remove(temp_path)
            except:
                pass
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calcule un score de confiance basé sur les segments"""
        if "segments" not in whisper_result:
            return 0.8  # Score par défaut
        
        segments = whisper_result["segments"]
        if not segments:
            return 0.5
        
        # Moyenne des probabilités des segments
        total_prob = sum(segment.get("avg_logprob", -1.0) for segment in segments)
        avg_prob = total_prob / len(segments)
        
        # Convertir en score 0-1
        confidence = max(0.0, min(1.0, (avg_prob + 1.0)))
        return round(confidence, 2)
    
    def _enhance_banking_transcript(self, transcript: str) -> str:
        """
        Améliore la transcription pour le contexte bancaire
        Corrige les termes bancaires mal transcrits
        """
        banking_corrections = {
            # Termes bancaires courants
            "virement": ["viremant", "vireman", "virement"],
            "compte": ["conte", "compt", "compte"],
            "solde": ["sold", "solde"],
            "crédit": ["crédi", "credit"],
            "débit": ["débi", "debit"],
            "épargne": ["épargn", "epargne"],
            "prêt": ["pret", "prêt"],
            "carte bancaire": ["cart bancaire", "carte bancair"],
            "découvert": ["découver", "decouvert"],
            "intérêts": ["interet", "interets"],
            "NUST": ["nust", "noust", "nus"],
            "PMYB": ["pmyb", "p m y b"],
            "ALS": ["als", "a l s"]
        }
        
        transcript_lower = transcript.lower()
        for correct, variants in banking_corrections.items():
            for variant in variants:
                if variant in transcript_lower:
                    transcript = transcript.replace(variant, correct)
                    transcript = transcript.replace(variant.capitalize(), correct.capitalize())
        
        return transcript.strip()
    
    async def synthesize_speech(self, text: str, voice_speed: float = 1.0) -> str:
        """
        Synthétise la parole avec Coqui TTS
        Optimisé pour les réponses bancaires
        """
        try:
            if not text.strip():
                raise ValueError("Texte vide")
            
            # Préparation du texte pour la synthèse
            text = self._prepare_text_for_tts(text)
            
            # Génération du fichier audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name
                self.temp_files.append(output_path)
            
            logger.info("🔊 Synthèse vocale en cours...")
            
            # Synthèse avec Coqui TTS
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speed=voice_speed
            )
            
            logger.info(f"✅ Synthèse réussie: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur synthèse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur synthèse: {str(e)}")
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """
        Prépare le texte pour une meilleure synthèse vocale
        Optimise la prononciation des termes bancaires
        """
        # Remplacements pour améliorer la prononciation
        tts_replacements = {
            "€": "euros",
            "%": "pourcent",
            "NUST": "N.U.S.T.",
            "PMYB": "P.M.Y.B.",
            "ALS": "A.L.S.",
            "SMS": "S.M.S.",
            "PIN": "P.I.N.",
            "ATM": "A.T.M.",
            "IBAN": "I.B.A.N.",
            "BIC": "B.I.C.",
            "RIB": "R.I.B."
        }
        
        for original, replacement in tts_replacements.items():
            text = text.replace(original, replacement)
        
        # Ajouter des pauses pour une meilleure élocution
        text = text.replace(". ", ". ... ")
        text = text.replace("? ", "? ... ")
        text = text.replace("! ", "! ... ")
        
        return text
    
    def cleanup_temp_files(self):
        """Nettoie les fichiers temporaires"""
        for file_path in self.temp_files[:]:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                self.temp_files.remove(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Impossible de supprimer {file_path}: {e}")

# Initialisation du service
speech_service = SpeechService()

# API FastAPI
app = FastAPI(
    title="Professional Speech Service",
    description="Service Speech professionnel pour chatbot bancaire",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/speech/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(..., description="Fichier audio à transcrire")
):
    """
    Transcrit un fichier audio en texte
    Optimisé pour les questions bancaires en français
    """
    result = await speech_service.transcribe_audio(audio)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/speech/synthesize")
async def synthesize_endpoint(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """
    Synthétise du texte en parole
    Optimisé pour les réponses bancaires
    """
    try:
        audio_path = await speech_service.synthesize_speech(
            text=request.text,
            voice_speed=request.voice_speed
        )
        
        # Programmer le nettoyage du fichier après envoi
        background_tasks.add_task(
            lambda: os.unlink(audio_path) if os.path.exists(audio_path) else None
        )
        
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename="response.wav",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speech/health")
async def health_check():
    """Vérification de l'état du service"""
    return {
        "status": "healthy",
        "whisper_loaded": speech_service.whisper_model is not None,
        "tts_loaded": speech_service.tts_model is not None,
        "device": speech_service.device,
        "cuda_available": torch.cuda.is_available(),
        "temp_files_count": len(speech_service.temp_files)
    }

@app.get("/speech/models")
async def models_info():
    """Informations sur les modèles chargés"""
    return {
        "whisper": {
            "model": "base",
            "language": "fr",
            "device": speech_service.device
        },
        "tts": {
            "model": "tts_models/fr/css10/vits",
            "language": "fr",
            "device": speech_service.device
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage lors de l'arrêt du service"""
    logger.info("🧹 Nettoyage des fichiers temporaires...")
    speech_service.cleanup_temp_files()

if __name__ == "__main__":
    logger.info("🚀 Démarrage du service Speech professionnel...")
    logger.info("📖 Documentation: http://localhost:8004/docs")
    logger.info("🎤 Endpoint STT: http://localhost:8004/speech/transcribe")
    logger.info("🔊 Endpoint TTS: http://localhost:8004/speech/synthesize")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info",
        access_log=True
    )
