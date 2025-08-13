import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { ragApiClient } from "@/config/api";
import { Mic, Volume2 } from "lucide-react";

export const ConnectionStatus = () => {
  const [status, setStatus] = useState<{
    connected: boolean;
    modelLoaded: boolean;
    gpuAvailable: boolean;
  }>({
    connected: false,
    modelLoaded: false,
    gpuAvailable: false,
  });

  const [speechStatus, setSpeechStatus] = useState<{
    connected: boolean;
    whisperLoaded: boolean;
    ttsEngine: string;
  }>({
    connected: false,
    whisperLoaded: false,
    ttsEngine: '',
  });

  const checkConnection = async () => {
    // Vérifier RAG
    try {
      const health = await ragApiClient.checkHealth();
      setStatus({
        connected: true,
        modelLoaded: health.model_loaded,
        gpuAvailable: health.gpu_available,
      });
    } catch (error) {
      setStatus({
        connected: false,
        modelLoaded: false,
        gpuAvailable: false,
      });
    }

    // Vérifier Speech Service
    try {
      const speechResponse = await fetch('http://localhost:8004/speech/health');
      if (speechResponse.ok) {
        const speechHealth = await speechResponse.json();
        setSpeechStatus({
          connected: true,
          whisperLoaded: speechHealth.whisper_loaded,
          ttsEngine: speechHealth.tts_engine,
        });
      } else {
        setSpeechStatus({
          connected: false,
          whisperLoaded: false,
          ttsEngine: '',
        });
      }
    } catch (error) {
      setSpeechStatus({
        connected: false,
        whisperLoaded: false,
        ttsEngine: '',
      });
    }
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 10000); // Vérifier toutes les 10s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex gap-2 items-center">
      <Badge variant={status.connected ? "default" : "destructive"}>
        {status.connected ? "🟢 Connecté" : "🔴 Déconnecté"}
      </Badge>
      
      {status.connected && (
        <>
          <Badge variant={status.modelLoaded ? "default" : "secondary"}>
            {status.modelLoaded ? "🦙 Modèle chargé" : "⏳ Chargement..."}
          </Badge>
          
          <Badge variant={status.gpuAvailable ? "default" : "secondary"}>
            {status.gpuAvailable ? "🚀 GPU" : "💻 CPU"}
          </Badge>
        </>
      )}

      {/* Statut Speech Service */}
      <Badge variant={speechStatus.connected ? "default" : "secondary"}>
        {speechStatus.connected ? (
          <div className="flex items-center gap-1">
            <Mic className="h-3 w-3" />
            <Volume2 className="h-3 w-3" />
            Speech
          </div>
        ) : (
          "🔴 Speech"
        )}
      </Badge>

      {speechStatus.connected && speechStatus.whisperLoaded && (
        <Badge variant="outline" className="text-xs">
          {speechStatus.ttsEngine}
        </Badge>
      )}
    </div>
  );
};
