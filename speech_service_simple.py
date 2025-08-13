"""
Service Speech SIMPLE - Sans conversion FFmpeg
Utilise directement Whisper sur WebM
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

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    voice_speed: Optional[float] = 1.0

class SimpleSpeechService:
    def __init__(self):
        self.whisper_model = None
        self.temp_files = []
        logger.info("🎯 Service Speech SIMPLE - Sans FFmpeg")
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
        """Transcrit directement sans conversion"""
        temp_path = None
        try:
            logger.info("🎤 Début transcription SIMPLE...")
            
            # Sauvegarder le fichier
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            logger.info(f"📁 Fichier: {temp_path}")
            
            # Vérifier taille du fichier
            file_size = os.path.getsize(temp_path)
            logger.info(f"📊 Taille: {file_size} bytes")
            
            if file_size < 1000:  # Fichier trop petit
                return {
                    "success": False,
                    "error": "Fichier audio trop court",
                    "transcript": ""
                }
            
            # Transcription DIRECTE avec Whisper
            logger.info("🎤 Transcription directe...")
            
            try:
                result = self.whisper_model.transcribe(
                    temp_path,
                    language="fr",
                    task="transcribe",
                    verbose=False
                )
                
                transcript = result["text"].strip()
                logger.info(f"✅ Résultat: '{transcript}'")
                
                if not transcript:
                    return {
                        "success": False,
                        "error": "Aucun texte détecté",
                        "transcript": ""
                    }
                
                return {
                    "success": True,
                    "transcript": transcript,
                    "confidence": 0.9,
                    "language": "fr",
                    "model": "whisper-base-direct"
                }
                
            except Exception as whisper_error:
                logger.error(f"❌ Erreur Whisper: {whisper_error}")
                return {
                    "success": False,
                    "error": f"Erreur transcription: {str(whisper_error)}",
                    "transcript": ""
                }
            
        except Exception as e:
            logger.error(f"❌ ERREUR GÉNÉRALE: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcript": ""
            }
        finally:
            # Nettoyage
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info("🗑️ Fichier supprimé")
                except:
                    logger.warning("⚠️ Impossible de supprimer le fichier")
    
    async def synthesize_speech(self, text: str, speed: float = 1.0) -> str:
        """Synthèse vocale avec gTTS"""
        try:
            logger.info(f"🔊 Synthèse: {text[:30]}...")
            
            tts = gTTS(text=text, lang="fr", slow=(speed < 0.8))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                output_path = temp_file.name
            
            tts.save(output_path)
            logger.info("✅ Synthèse réussie")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur synthèse: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialisation
speech_service = SimpleSpeechService()
app = FastAPI(title="Speech Service SIMPLE", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Speech Service SIMPLE", "status": "running"}

@app.get("/speech/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_loaded": speech_service.whisper_model is not None,
        "tts_engine": "gTTS",
        "version": "SIMPLE-DIRECT",
        "ffmpeg_required": False
    }

@app.post("/speech/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """Endpoint de transcription SIMPLE"""
    logger.info(f"📥 Requête: {audio.content_type}")
    
    result = await speech_service.transcribe_audio(audio)
    
    if not result["success"]:
        logger.error(f"❌ Échec: {result['error']}")
        raise HTTPException(status_code=400, detail=result["error"])
    
    logger.info("✅ Succès transcription")
    return result

@app.post("/speech/synthesize")
async def synthesize_endpoint(request: TTSRequest, background_tasks: BackgroundTasks):
    """Endpoint de synthèse vocale"""
    try:
        audio_path = await speech_service.synthesize_speech(request.text, request.voice_speed)
        
        background_tasks.add_task(
            lambda: os.unlink(audio_path) if os.path.exists(audio_path) else None
        )
        
        return FileResponse(
            audio_path, 
            media_type="audio/mpeg", 
            filename="response.mp3"
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur synthèse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Démarrage Service Speech SIMPLE")
    logger.info("📖 Documentation: http://localhost:8004/docs")
    uvicorn.run(app, host="0.0.0.0", port=8004)
