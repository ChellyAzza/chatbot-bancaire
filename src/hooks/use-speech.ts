import { useState, useEffect, useRef, useCallback } from 'react';

interface TranscriptionResult {
  success: boolean;
  transcript: string;
  confidence: number;
  error?: string;
  language?: string;
  model?: string;
}

interface SpeechServiceHealth {
  status: string;
  python_version: string;
  whisper_loaded: boolean;
  tts_engine: string;
  temp_files: number;
}

const SPEECH_API_URL = 'http://localhost:8004';

export const useSpeech = () => {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [isSupported, setIsSupported] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serviceHealth, setServiceHealth] = useState<SpeechServiceHealth | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  // Vérifier le support et l'état du service
  useEffect(() => {
    checkServiceHealth();
  }, []);

  const checkServiceHealth = useCallback(async () => {
    try {
      const response = await fetch(`${SPEECH_API_URL}/speech/health`);
      if (response.ok) {
        const health: SpeechServiceHealth = await response.json();
        setServiceHealth(health);
        setIsSupported(health.whisper_loaded && health.tts_engine === 'gTTS');
      } else {
        setIsSupported(false);
        setError('Service Speech indisponible');
      }
    } catch (err) {
      setIsSupported(false);
      setError('Impossible de contacter le service Speech');
    }
  }, []);

  // Démarrer l'enregistrement audio
  const startListening = useCallback(async () => {
    if (!isSupported) {
      setError('Service Speech non disponible');
      return;
    }

    if (isListening) return;

    try {
      setError(null);
      setTranscript('');
      setConfidence(0);

      // Demander permission microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000, // Optimal pour Whisper
          channelCount: 1,   // Mono
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // Créer MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
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

        // Transcrire avec Whisper
        await transcribeWithWhisper(audioBlob);

        // Nettoyer le stream
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsListening(true);

    } catch (err) {
      setError('Impossible d\'accéder au microphone');
      console.error('Recording error:', err);
    }
  }, [isSupported, isListening]);

  // Arrêter l'enregistrement
  const stopListening = useCallback(() => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
    }
  }, [isListening]);

  // Convertir WebM en WAV côté client
  const convertAndTranscribe = useCallback(async (audioBlob: Blob) => {
    setIsTranscribing(true);
    setError(null);

    try {
      // Créer un contexte audio pour la conversion
      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // Convertir en WAV
      const wavBlob = await audioBufferToWav(audioBuffer);

      // Transcrire le WAV
      await transcribeWithWhisper(wavBlob);

    } catch (err) {
      setError('Erreur de conversion audio');
      console.error('Conversion error:', err);
      setIsTranscribing(false);
    }
  }, []);

  // Convertir AudioBuffer en WAV
  const audioBufferToWav = useCallback(async (audioBuffer: AudioBuffer): Promise<Blob> => {
    const numberOfChannels = 1; // Mono
    const sampleRate = 16000; // Optimal pour Whisper
    const length = audioBuffer.length;
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);

    // En-tête WAV
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);

    // Données audio
    const channelData = audioBuffer.getChannelData(0);
    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }, []);

  // Transcrire avec Whisper
  const transcribeWithWhisper = useCallback(async (audioBlob: Blob) => {
    setIsTranscribing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');

      const response = await fetch(`${SPEECH_API_URL}/speech/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: TranscriptionResult = await response.json();

      if (result.success && result.transcript) {
        setTranscript(result.transcript);
        setConfidence(result.confidence);
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

  // Synthèse vocale avec Coqui TTS
  const speak = useCallback(async (text: string, options?: {
    speed?: number;
    pitch?: number;
  }) => {
    if (!isSupported) {
      setError('Service Speech non disponible');
      return;
    }

    if (isSpeaking) {
      stopSpeaking();
    }

    try {
      setError(null);
      setIsSpeaking(true);

      const response = await fetch(`${SPEECH_API_URL}/speech/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          voice_speed: options?.speed || 1.0
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Récupérer le fichier audio
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      // Jouer l'audio
      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;

      audio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
        currentAudioRef.current = null;
      };

      audio.onerror = () => {
        setError('Erreur de lecture audio');
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
        currentAudioRef.current = null;
      };

      await audio.play();

    } catch (err) {
      setError('Erreur de synthèse vocale');
      setIsSpeaking(false);
      console.error('TTS error:', err);
    }
  }, [isSupported, isSpeaking]);

  // Arrêter la synthèse vocale
  const stopSpeaking = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      setIsSpeaking(false);
      currentAudioRef.current = null;
    }
  }, []);

  // Nettoyer les ressources
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && isListening) {
        mediaRecorderRef.current.stop();
      }
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
    };
  }, [isListening]);

  return {
    // États
    isListening,
    isSpeaking,
    isTranscribing,
    transcript,
    confidence,
    isSupported,
    error,
    serviceHealth,

    // Actions
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    checkServiceHealth,

    // Utilitaires
    clearTranscript: () => setTranscript(''),
    clearError: () => setError(null),
    isProcessing: isListening || isTranscribing || isSpeaking
  };
};
