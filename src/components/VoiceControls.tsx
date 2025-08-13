import { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  AlertCircle
} from "lucide-react";
import { useWebSpeech } from "@/hooks/use-web-speech";
import { useToast } from "@/hooks/use-toast";

interface VoiceControlsProps {
  onTranscriptReady: (transcript: string) => void;
  lastBotMessage?: string;
  className?: string;
}

export const VoiceControls = ({
  onTranscriptReady, 
  lastBotMessage, 
  className = "" 
}: VoiceControlsProps) => {
  const [autoSpeak, setAutoSpeak] = useState(false);
  const { toast } = useToast();
  
  const {
    isListening,
    isSpeaking,
    transcript,
    speechAvailable,
    error,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    clearTranscript,
    clearError
  } = useWebSpeech();

  // Envoyer le transcript quand il est prêt
  useEffect(() => {
    if (transcript && !isListening) {
      onTranscriptReady(transcript);
      clearTranscript();
    }
  }, [transcript, isListening, onTranscriptReady, clearTranscript]);

  // Lecture automatique des réponses du bot
  useEffect(() => {
    if (autoSpeak && lastBotMessage && !isSpeaking) {
      speak(lastBotMessage);
    }
  }, [lastBotMessage, autoSpeak, speak, isSpeaking]);

  // Gestion des erreurs
  useEffect(() => {
    if (error) {
      toast({
        title: "Erreur vocale",
        description: error,
        variant: "destructive"
      });
      clearError();
    }
  }, [error, toast, clearError]);

  const handleMicClick = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  const handleSpeakClick = () => {
    if (isSpeaking) {
      stopSpeaking();
    } else if (lastBotMessage) {
      speak(lastBotMessage);
    }
  };

  if (!speechAvailable) {
    return (
      <Card className={`p-3 bg-muted/50 ${className}`}>
        <div className="flex items-center gap-2 text-muted-foreground">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">
            Web Speech API non supportée par ce navigateur
          </span>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`p-3 bg-gradient-surface border-border/50 ${className}`}>
      <div className="space-y-3">
        {/* Contrôles principaux */}
        <div className="flex items-center gap-2">
          {/* Bouton Microphone */}
          <Button
            variant={isListening ? "destructive" : "outline"}
            size="sm"
            onClick={handleMicClick}
            className="relative"
          >
            {isListening ? (
              <>
                <MicOff className="h-4 w-4 mr-2" />
                Arrêter
              </>
            ) : (
              <>
                <Mic className="h-4 w-4 mr-2" />
                Écouter
              </>
            )}
          </Button>

          {/* Bouton Lecture */}
          <Button
            variant={isSpeaking ? "destructive" : "outline"}
            size="sm"
            onClick={handleSpeakClick}
            disabled={!lastBotMessage}
          >
            {isSpeaking ? (
              <>
                <VolumeX className="h-4 w-4 mr-2" />
                Arrêter
              </>
            ) : (
              <>
                <Volume2 className="h-4 w-4 mr-2" />
                Lire
              </>
            )}
          </Button>

          {/* Auto-lecture */}
          <Button
            variant={autoSpeak ? "default" : "ghost"}
            size="sm"
            onClick={() => setAutoSpeak(!autoSpeak)}
            className="text-xs"
          >
            Auto-lecture
          </Button>
        </div>

        {/* Statuts */}
        <div className="flex flex-wrap gap-1">
          {isListening && (
            <Badge variant="default" className="animate-pulse">
              <Mic className="h-3 w-3 mr-1" />
              Écoute...
            </Badge>
          )}

          {isSpeaking && (
            <Badge variant="secondary">
              <Volume2 className="h-3 w-3 mr-1" />
              Lecture...
            </Badge>
          )}

          {transcript && isListening && (
            <Badge variant="outline" className="max-w-32 truncate">
              {transcript}
            </Badge>
          )}
        </div>

        {/* Message d'aide */}
        {!isListening && !isSpeaking && (
          <p className="text-xs text-muted-foreground">
            Cliquez sur "Écouter" pour parler au chatbot ou "Lire" pour entendre la dernière réponse
          </p>
        )}
      </div>
    </Card>
  );
};
