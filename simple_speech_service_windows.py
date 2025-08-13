"""
Service Speech simplifié pour Windows
Sans conversion audio - accepte seulement WAV
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    voice_speed: Optional[float] = 1.0

class SimpleSpeechService:
    def __init__(self):
        self.whisper_model = None
        self.temp_files = []
        logger.info("🎯 Service Speech Windows - Format WAV uniquement")
        self._load_whisper()
    
    def _load_whisper(self):
        try:
            logger.info("📥 Chargement Whisper...")
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper chargé")
        except Exception as e:
            logger.error(f"❌ Erreur Whisper: {e}")
            raise
    
    async def transcribe_audio(self, audio_file: UploadFile) -> Dict[str, Any]:
        try:
            # Accepter seulement WAV pour éviter les problèmes de conversion
            if "wav" not in audio_file.content_type.lower():
                return {
                    "success": False,
                    "error": "Format non supporté. Utilisez WAV uniquement.",
                    "transcript": ""
                }
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                self.temp_files.append(temp_path)
            
            logger.info("🎤 Transcription...")
            result = self.whisper_model.transcribe(temp_path, language="fr")
            
            transcript = result["text"].strip()
            
            return {
                "success": True,
                "transcript": transcript,
                "confidence": 0.9,
                "language": "fr",
                "model": "whisper-base"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcript": ""
            }
        finally:
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
                    if temp_path in self.temp_files:
                        self.temp_files.remove(temp_path)
            except:
                pass
    
    async def synthesize_speech(self, text: str, speed: float = 1.0) -> str:
        try:
            tts = gTTS(text=text, lang="fr", slow=(speed < 0.8))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                output_path = temp_file.name
                self.temp_files.append(output_path)
            
            tts.save(output_path)
            return output_path
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Service et API
speech_service = SimpleSpeechService()
app = FastAPI(title="Simple Speech Service Windows")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/speech/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    result = await speech_service.transcribe_audio(audio)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/speech/synthesize")
async def synthesize_endpoint(request: TTSRequest, background_tasks: BackgroundTasks):
    try:
        audio_path = await speech_service.synthesize_speech(request.text, request.voice_speed)
        background_tasks.add_task(lambda: os.unlink(audio_path) if os.path.exists(audio_path) else None)
        return FileResponse(audio_path, media_type="audio/mpeg", filename="response.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speech/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_loaded": speech_service.whisper_model is not None,
        "tts_engine": "gTTS",
        "format_supported": "WAV only"
    }

if __name__ == "__main__":
    logger.info("🚀 Service Speech Windows (WAV uniquement)")
    uvicorn.run(app, host="0.0.0.0", port=8004)
