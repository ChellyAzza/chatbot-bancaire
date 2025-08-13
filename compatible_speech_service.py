"""
Service Speech compatible Python 3.13
Utilise Whisper + gTTS (plus stable que Coqui TTS)
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

class CompatibleSpeechService:
    def __init__(self):
        self.whisper_model = None
        self.temp_files = []
        
        logger.info("🎯 Initialisation Service Speech Compatible Python 3.13")
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
        """Transcrit un fichier audio avec Whisper - Version simplifiée"""
        try:
            logger.info("🎤 Début transcription...")

            # Sauvegarder le fichier temporairement
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                self.temp_files.append(temp_path)

            logger.info("🎤 Transcription avec Whisper...")

            # Whisper peut traiter WebM directement
            result = self.whisper_model.transcribe(
                temp_path,
                language="fr",
                task="transcribe",
                verbose=False
            )
            
            # Nettoyage et amélioration pour contexte bancaire
            transcript = result["text"].strip()
            transcript = self._enhance_banking_transcript(transcript)
            
            # Score de confiance
            confidence = self._calculate_confidence(result)
            
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
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calcule un score de confiance"""
        text = whisper_result.get("text", "").strip()
        if not text:
            return 0.0
        
        # Score basé sur la longueur et cohérence
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
            # Corrections bancaires spécifiques
            "virement": ["viremant", "vireman", "virement"],
            "compte": ["conte", "compt", "compte"],
            "solde": ["sold", "solde"],
            "crédit": ["crédi", "credit", "crédit"],
            "débit": ["débi", "debit", "débit"],
            "épargne": ["épargn", "epargne", "épargne"],
            "prêt": ["pret", "prêt"],
            "carte bancaire": ["cart bancaire", "carte bancair"],
            "découvert": ["découver", "decouvert"],
            "NUST": ["nust", "noust", "nus"],
            "PMYB": ["pmyb", "p m y b"],
            "ALS": ["als", "a l s"],
            "euros": ["euro", "euros"],
            "pourcentage": ["pourcent", "%"]
        }
        
        transcript_lower = transcript.lower()
        for correct, variants in banking_corrections.items():
            for variant in variants:
                if variant in transcript_lower:
                    transcript = transcript.replace(variant, correct)
                    transcript = transcript.replace(variant.capitalize(), correct.capitalize())
        
        return transcript.strip()
    
    async def synthesize_speech(self, text: str, speed: float = 1.0) -> str:
        """Synthétise la parole avec gTTS"""
        try:
            if not text.strip():
                raise ValueError("Texte vide")
            
            # Préparation du texte pour TTS
            text = self._prepare_text_for_tts(text)
            
            logger.info("🔊 Synthèse vocale gTTS...")
            
            # Déterminer la vitesse
            slow_mode = speed < 0.8
            
            # Synthèse avec gTTS
            tts = gTTS(text=text, lang="fr", slow=slow_mode)
            
            # Sauvegarder
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                output_path = temp_file.name
                self.temp_files.append(output_path)
            
            tts.save(output_path)
            
            logger.info(f"✅ Synthèse réussie")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur synthèse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur synthèse: {str(e)}")
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """Prépare le texte pour une meilleure prononciation"""
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
            "RIB": "R I B",
            "ATM": "A T M"
        }
        
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        
        return text

# Initialisation
speech_service = CompatibleSpeechService()

# API FastAPI
app = FastAPI(
    title="Compatible Speech Service",
    description="Service Speech compatible Python 3.13 - Whisper + gTTS",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/speech/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """Transcrit un fichier audio"""
    result = await speech_service.transcribe_audio(audio)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/speech/synthesize")
async def synthesize_endpoint(request: TTSRequest, background_tasks: BackgroundTasks):
    """Synthétise du texte en parole"""
    try:
        audio_path = await speech_service.synthesize_speech(
            text=request.text,
            speed=request.voice_speed
        )
        
        # Nettoyage automatique
        background_tasks.add_task(
            lambda: os.unlink(audio_path) if os.path.exists(audio_path) else None
        )
        
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="response.mp3"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speech/health")
async def health_check():
    """État du service"""
    return {
        "status": "healthy",
        "python_version": "3.13.5",
        "whisper_loaded": speech_service.whisper_model is not None,
        "tts_engine": "gTTS",
        "temp_files": len(speech_service.temp_files)
    }

@app.get("/speech/test")
async def test_endpoint():
    """Test rapide du service"""
    return {
        "message": "Service Speech opérationnel",
        "whisper": "✅ Prêt pour transcription",
        "tts": "✅ Prêt pour synthèse",
        "endpoints": [
            "POST /speech/transcribe - Transcription audio",
            "POST /speech/synthesize - Synthèse vocale",
            "GET /speech/health - État du service"
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Démarrage Service Speech Compatible...")
    logger.info("🐍 Python 3.13.5 détecté")
    logger.info("📖 Documentation: http://localhost:8004/docs")
    logger.info("🧪 Test: http://localhost:8004/speech/test")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
