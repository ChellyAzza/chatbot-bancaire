"""
Intégration Text-to-Speech pour le chatbot bancaire
Utilise Coqui TTS (gratuit) comme option principale
"""

import os
import io
import tempfile
from typing import Optional, Dict, Any, List
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Option 1: Coqui TTS (GRATUIT - Recommandé)
try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
    print("✅ Coqui TTS disponible")
except ImportError:
    COQUI_TTS_AVAILABLE = False
    print("⚠️ Coqui TTS non installé (pip install TTS)")

# Option 2: gTTS (GRATUIT)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    print("✅ gTTS disponible")
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️ gTTS non installé (pip install gtts)")

# Option 3: OpenAI TTS (PAYANT)
try:
    import openai
    OPENAI_TTS_AVAILABLE = True
    print("✅ OpenAI TTS disponible")
except ImportError:
    OPENAI_TTS_AVAILABLE = False
    print("⚠️ OpenAI TTS non installé")

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    language: Optional[str] = "fr"

class TextToSpeechService:
    def __init__(self, preferred_engine: str = "coqui"):
        """
        Initialise le service Text-to-Speech
        
        Args:
            preferred_engine: "coqui", "gtts", ou "openai"
        """
        self.preferred_engine = preferred_engine
        self.coqui_model = None
        self.available_engines = []
        
        self._check_available_engines()
        self._initialize_engines()
    
    def _check_available_engines(self):
        """Vérifie les moteurs TTS disponibles"""
        if COQUI_TTS_AVAILABLE:
            self.available_engines.append("coqui")
        if GTTS_AVAILABLE:
            self.available_engines.append("gtts")
        if OPENAI_TTS_AVAILABLE:
            self.available_engines.append("openai")
        
        print(f"🔊 Moteurs TTS disponibles: {self.available_engines}")
    
    def _initialize_engines(self):
        """Initialise les moteurs TTS"""
        if "coqui" in self.available_engines:
            try:
                # Charger un modèle français de qualité
                self.coqui_model = TTS(model_name="tts_models/fr/css10/vits")
                print("✅ Modèle Coqui TTS français chargé")
            except Exception as e:
                print(f"⚠️ Erreur chargement Coqui TTS: {e}")
                if "coqui" in self.available_engines:
                    self.available_engines.remove("coqui")
        
        if "openai" in self.available_engines:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                print("✅ OpenAI TTS configuré")
            else:
                print("⚠️ OPENAI_API_KEY manquante")
                if "openai" in self.available_engines:
                    self.available_engines.remove("openai")
    
    def synthesize_speech(self, text: str, voice: str = "default", 
                         speed: float = 1.0, language: str = "fr") -> Dict[str, Any]:
        """
        Synthétise la parole à partir du texte
        
        Args:
            text: Texte à synthétiser
            voice: Voix à utiliser
            speed: Vitesse de parole
            language: Langue
            
        Returns:
            Dict avec le fichier audio et métadonnées
        """
        if not self.available_engines:
            return {
                "success": False,
                "error": "Aucun moteur TTS disponible"
            }
        
        # Essayer le moteur préféré en premier
        if self.preferred_engine in self.available_engines:
            result = self._synthesize_with_engine(
                text, self.preferred_engine, voice, speed, language
            )
            if result["success"]:
                return result
        
        # Fallback vers d'autres moteurs
        for engine in self.available_engines:
            if engine != self.preferred_engine:
                result = self._synthesize_with_engine(
                    text, engine, voice, speed, language
                )
                if result["success"]:
                    return result
        
        return {
            "success": False,
            "error": "Échec de synthèse avec tous les moteurs"
        }
    
    def _synthesize_with_engine(self, text: str, engine: str, voice: str, 
                               speed: float, language: str) -> Dict[str, Any]:
        """Synthétise avec un moteur spécifique"""
        try:
            if engine == "coqui":
                return self._synthesize_coqui(text, voice, speed)
            elif engine == "gtts":
                return self._synthesize_gtts(text, language, speed)
            elif engine == "openai":
                return self._synthesize_openai(text, voice, speed)
            else:
                return {"success": False, "error": f"Moteur {engine} non supporté"}
        except Exception as e:
            return {"success": False, "error": f"Erreur {engine}: {str(e)}"}
    
    def _synthesize_coqui(self, text: str, voice: str, speed: float) -> Dict[str, Any]:
        """Synthèse avec Coqui TTS"""
        if not self.coqui_model:
            return {"success": False, "error": "Modèle Coqui non chargé"}
        
        # Créer fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            output_path = temp_file.name
        
        # Synthétiser
        self.coqui_model.tts_to_file(text=text, file_path=output_path)
        
        return {
            "success": True,
            "audio_file": output_path,
            "engine": "coqui",
            "format": "wav",
            "quality": "high"
        }
    
    def _synthesize_gtts(self, text: str, language: str, speed: float) -> Dict[str, Any]:
        """Synthèse avec gTTS"""
        # Ajuster la vitesse (gTTS ne supporte pas directement)
        if speed != 1.0:
            # On peut répéter des mots pour simuler la lenteur
            if speed < 1.0:
                text = text.replace(" ", "  ")  # Espaces supplémentaires
        
        tts = gTTS(text=text, lang=language, slow=(speed < 0.8))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            output_path = temp_file.name
        
        tts.save(output_path)
        
        return {
            "success": True,
            "audio_file": output_path,
            "engine": "gtts",
            "format": "mp3",
            "quality": "medium"
        }
    
    def _synthesize_openai(self, text: str, voice: str, speed: float) -> Dict[str, Any]:
        """Synthèse avec OpenAI TTS"""
        # Voix disponibles OpenAI
        openai_voices = {
            "default": "nova",
            "female": "nova",
            "male": "onyx",
            "alloy": "alloy",
            "echo": "echo",
            "fable": "fable",
            "nova": "nova",
            "onyx": "onyx",
            "shimmer": "shimmer"
        }
        
        selected_voice = openai_voices.get(voice, "nova")
        
        response = openai.audio.speech.create(
            model="tts-1",  # ou "tts-1-hd" pour haute qualité
            voice=selected_voice,
            input=text,
            speed=speed
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            output_path = temp_file.name
            response.stream_to_file(output_path)
        
        return {
            "success": True,
            "audio_file": output_path,
            "engine": "openai",
            "format": "mp3",
            "quality": "high",
            "voice": selected_voice
        }
    
    def get_available_voices(self) -> Dict[str, List[str]]:
        """Retourne les voix disponibles par moteur"""
        voices = {}
        
        if "coqui" in self.available_engines:
            voices["coqui"] = ["default", "female", "male"]
        
        if "gtts" in self.available_engines:
            voices["gtts"] = ["default"]
        
        if "openai" in self.available_engines:
            voices["openai"] = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
        
        return voices

# API FastAPI
app = FastAPI(title="Text-to-Speech API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le service TTS
tts_service = TextToSpeechService(preferred_engine="coqui")

@app.post("/synthesize")
async def synthesize_text(request: TTSRequest):
    """
    Synthétise du texte en parole
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide")
    
    result = tts_service.synthesize_speech(
        text=request.text,
        voice=request.voice,
        speed=request.speed,
        language=request.language
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Récupère un fichier audio généré
    """
    if os.path.exists(filename):
        return FileResponse(filename, media_type="audio/wav")
    else:
        raise HTTPException(status_code=404, detail="Fichier audio non trouvé")

@app.get("/voices")
async def get_available_voices():
    """
    Liste des voix disponibles
    """
    return {
        "voices": tts_service.get_available_voices(),
        "engines": tts_service.available_engines,
        "preferred_engine": tts_service.preferred_engine
    }

@app.get("/health")
async def health_check():
    """
    Vérification de l'état du service
    """
    return {
        "status": "healthy",
        "available_engines": tts_service.available_engines,
        "coqui_available": COQUI_TTS_AVAILABLE,
        "gtts_available": GTTS_AVAILABLE,
        "openai_available": OPENAI_TTS_AVAILABLE
    }

if __name__ == "__main__":
    print("🔊 Démarrage du serveur TTS...")
    print("📝 Documentation: http://localhost:8003/docs")
    uvicorn.run(app, host="0.0.0.0", port=8003)
