"""
Service Speech simplifié et compatible
Utilise Whisper + gTTS (plus compatible que Coqui TTS)
"""

import os
import tempfile
import logging
from typing import Optional, Dict, Any
import whisper
from gtts import gTTS
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    voice_speed: Optional[float] = 1.0
    language: Optional[str] = "fr"

class SimpleSpeechService:
    def __init__(self):
        self.whisper_model = None
        self.temp_files = []
        
        logger.info("🎯 Initialisation Service Speech Simplifié")
        self._load_whisper()
    
    def _load_whisper(self):
        """Charge le modèle Whisper"""
        try:
            logger.info("📥 Chargement Whisper base...")
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur chargement Whisper: {e}")
            raise
    
    async def transcribe_audio(self, audio_file: UploadFile) -> Dict[str, Any]:
        """Transcrit un fichier audio avec Whisper"""
        try:
            # Validation
            if not audio_file.content_type.startswith("audio/"):
                raise ValueError("Fichier audio requis")
            
            # Sauvegarder temporairement
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                self.temp_files.append(temp_path)
            
            logger.info("🎤 Transcription en cours...")
            
            # Transcription avec Whisper
            result = self.whisper_model.transcribe(
                temp_path,
                language="fr",
                task="transcribe",
                verbose=False
            )
            
            # Nettoyage et amélioration
            transcript = result["text"].strip()
            transcript = self._enhance_banking_transcript(transcript)
            
            # Calcul de confiance simplifié
            confidence = self._calculate_simple_confidence(result)
            
            logger.info(f"✅ Transcription: {transcript[:50]}...")
            
            return {
                "success": True,
                "transcript": transcript,
                "confidence": confidence,
                "language": result.get("language", "fr"),
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
            # Nettoyage
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
                    if temp_path in self.temp_files:
                        self.temp_files.remove(temp_path)
            except:
                pass
    
    def _calculate_simple_confidence(self, whisper_result: Dict) -> float:
        """Calcule un score de confiance simple"""
        # Score basé sur la longueur et la présence de mots
        text = whisper_result.get("text", "").strip()
        if not text:
            return 0.0
        
        # Plus le texte est long et cohérent, plus la confiance est élevée
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        elif word_count < 3:
            return 0.6
        elif word_count < 10:
            return 0.8
        else:
            return 0.9
    
    def _enhance_banking_transcript(self, transcript: str) -> str:
        """Améliore la transcription pour le contexte bancaire"""
        banking_corrections = {
            # Termes bancaires courants
            "virement": ["viremant", "vireman"],
            "compte": ["conte", "compt"],
            "solde": ["sold"],
            "crédit": ["crédi", "credit"],
            "débit": ["débi", "debit"],
            "épargne": ["épargn", "epargne"],
            "prêt": ["pret"],
            "carte bancaire": ["cart bancaire", "carte bancair"],
            "découvert": ["découver", "decouvert"],
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
    
    async def synthesize_speech(self, text: str, language: str = "fr", slow: bool = False) -> str:
        """Synthétise la parole avec gTTS"""
        try:
            if not text.strip():
                raise ValueError("Texte vide")
            
            # Préparation du texte
            text = self._prepare_text_for_tts(text)
            
            logger.info("🔊 Synthèse vocale gTTS...")
            
            # Synthèse avec gTTS
            tts = gTTS(text=text, lang=language, slow=slow)
            
            # Sauvegarder dans un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                output_path = temp_file.name
                self.temp_files.append(output_path)
            
            tts.save(output_path)
            
            logger.info(f"✅ Synthèse réussie: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur synthèse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur synthèse: {str(e)}")
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """Prépare le texte pour gTTS"""
        # Remplacements pour améliorer la prononciation
        replacements = {
            "€": "euros",
            "%": "pourcent",
            "NUST": "N U S T",
            "PMYB": "P M Y B",
            "ALS": "A L S",
            "SMS": "S M S",
            "PIN": "P I N",
            "IBAN": "I B A N",
            "BIC": "B I C",
            "RIB": "R I B"
        }
        
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        
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
speech_service = SimpleSpeechService()

# API FastAPI
app = FastAPI(
    title="Simple Speech Service",
    description="Service Speech simplifié avec Whisper + gTTS",
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
    """Transcrit un fichier audio en texte"""
    result = await speech_service.transcribe_audio(audio)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/speech/synthesize")
async def synthesize_endpoint(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """Synthétise du texte en parole"""
    try:
        # Déterminer si utiliser le mode lent
        slow_mode = request.voice_speed < 0.8 if request.voice_speed else False
        
        audio_path = await speech_service.synthesize_speech(
            text=request.text,
            language=request.language,
            slow=slow_mode
        )
        
        # Programmer le nettoyage
        background_tasks.add_task(
            lambda: os.unlink(audio_path) if os.path.exists(audio_path) else None
        )
        
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="response.mp3",
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
        "tts_engine": "gTTS",
        "temp_files_count": len(speech_service.temp_files)
    }

@app.get("/speech/models")
async def models_info():
    """Informations sur les modèles"""
    return {
        "whisper": {
            "model": "base",
            "language": "fr"
        },
        "tts": {
            "engine": "gTTS",
            "language": "fr",
            "quality": "good"
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage lors de l'arrêt"""
    logger.info("🧹 Nettoyage des fichiers temporaires...")
    speech_service.cleanup_temp_files()

if __name__ == "__main__":
    logger.info("🚀 Démarrage Service Speech Simplifié...")
    logger.info("📖 Documentation: http://localhost:8004/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
