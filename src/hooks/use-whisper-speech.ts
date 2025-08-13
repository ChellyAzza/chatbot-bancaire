import { useState, useRef, useCallback } from 'react';

interface WhisperResponse {
  text: string;
  language?: string;
  success: boolean;
  error?: string;
  method?: string;
  model?: string;
}

export const useWhisperSpeech = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSupported, setIsSupported] = useState(true);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Démarrer l'enregistrement
  const startRecording = useCallback(async () => {
    try {
      setError(null);
      
      // Demander permission microphone
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000, // Optimal pour Whisper
          channelCount: 1,   // Mono
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      // Créer MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus' // Format supporté
      });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: 'audio/webm;codecs=opus' 
        });
        
        // Envoyer à Whisper pour transcription
        await transcribeWithWhisper(audioBlob);
        
        // Nettoyer le stream
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
      
    } catch (err) {
      setError('Impossible d\'accéder au microphone');
      setIsSupported(false);
      console.error('Recording error:', err);
    }
  }, []);

  // Arrêter l'enregistrement
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  // Transcrire avec Whisper
  const transcribeWithWhisper = useCallback(async (audioBlob: Blob) => {
    setIsTranscribing(true);
    setError(null);
    
    try {
      // Préparer FormData pour l'API
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('language', 'fr'); // Français
      
      // Appel à l'API Whisper
      const response = await fetch('http://localhost:8002/transcribe', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result: WhisperResponse = await response.json();
      
      if (result.success && result.text) {
        setTranscript(result.text);
      } else {
        setError(result.error || 'Erreur de transcription');
      }
      
    } catch (err) {
      setError('Erreur de connexion au service Whisper');
      console.error('Transcription error:', err);
    } finally {
      setIsTranscribing(false);
    }
  }, []);

  // Transcrire un fichier audio uploadé
  const transcribeFile = useCallback(async (file: File) => {
    setIsTranscribing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('audio', file);
      formData.append('language', 'fr');
      
      const response = await fetch('http://localhost:8002/transcribe', {
        method: 'POST',
        body: formData,
      });
      
      const result: WhisperResponse = await response.json();
      
      if (result.success && result.text) {
        setTranscript(result.text);
        return result.text;
      } else {
        setError(result.error || 'Erreur de transcription');
        return null;
      }
      
    } catch (err) {
      setError('Erreur de transcription du fichier');
      console.error('File transcription error:', err);
      return null;
    } finally {
      setIsTranscribing(false);
    }
  }, []);

  // Vérifier la disponibilité du service
  const checkWhisperService = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8002/health');
      const health = await response.json();
      setIsSupported(health.status === 'healthy');
      return health;
    } catch (err) {
      setIsSupported(false);
      setError('Service Whisper indisponible');
      return null;
    }
  }, []);

  // Nettoyer le transcript
  const clearTranscript = useCallback(() => {
    setTranscript('');
  }, []);

  // Nettoyer les erreurs
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // États
    isRecording,
    isTranscribing,
    transcript,
    error,
    isSupported,
    
    // Actions
    startRecording,
    stopRecording,
    transcribeFile,
    checkWhisperService,
    clearTranscript,
    clearError,
    
    // Utilitaires
    isProcessing: isRecording || isTranscribing
  };
};
