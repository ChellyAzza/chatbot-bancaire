import { useState, useCallback, useRef, useEffect } from 'react';

// Types pour la Web Speech API
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onstart: (() => void) | null;
  onend: (() => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

export const useWebSpeech = () => {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [speechAvailable, setSpeechAvailable] = useState(false);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  // Vérifier la disponibilité de la Web Speech API
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const speechSynthesis = window.speechSynthesis;

    if (SpeechRecognition && speechSynthesis) {
      setSpeechAvailable(true);
      synthRef.current = speechSynthesis;
      setError(null);
    } else {
      setSpeechAvailable(false);
      setError('Web Speech API non supportée par ce navigateur');
    }
  }, []);

  // Démarrer l'écoute
  const startListening = useCallback(() => {
    if (!speechAvailable) {
      setError('Speech API non disponible');
      return;
    }

    try {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();

      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'fr-FR';

      recognition.onstart = () => {
        setIsListening(true);
        setError(null);
        setTranscript('');
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        const result = event.results[0];
        if (result.isFinal) {
          const transcriptText = result[0].transcript;
          setTranscript(transcriptText);
          setIsListening(false);
        }
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        setError(`Erreur reconnaissance vocale: ${event.error}`);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      recognition.start();

    } catch (err) {
      setError('Erreur lors du démarrage de la reconnaissance vocale');
      setIsListening(false);
    }
  }, [speechAvailable]);

  // Arrêter l'écoute
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, []);

  // Synthèse vocale
  const speak = useCallback((text: string) => {
    if (!synthRef.current || !speechAvailable) {
      setError('Synthèse vocale non disponible');
      return;
    }

    try {
      // Arrêter toute synthèse en cours
      synthRef.current.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'fr-FR';
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;

      utterance.onstart = () => {
        setIsSpeaking(true);
        setError(null);
      };

      utterance.onend = () => {
        setIsSpeaking(false);
      };

      utterance.onerror = (event) => {
        setError(`Erreur synthèse vocale: ${event.error}`);
        setIsSpeaking(false);
      };

      synthRef.current.speak(utterance);

    } catch (err) {
      setError('Erreur lors de la synthèse vocale');
      setIsSpeaking(false);
    }
  }, [speechAvailable]);

  // Arrêter la synthèse
  const stopSpeaking = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.cancel();
      setIsSpeaking(false);
    }
  }, []);

  // Nettoyage
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, []);

  return {
    // États
    isListening,
    isSpeaking,
    transcript,
    error,
    speechAvailable,

    // Actions
    startListening,
    stopListening,
    speak,
    stopSpeaking,

    // Utilitaires
    clearTranscript: () => setTranscript(''),
    clearError: () => setError(null)
  };
};
