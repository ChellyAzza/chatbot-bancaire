import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Mic, PlusCircle, MicOff, Sparkles } from "lucide-react";
import { useWebSpeech } from "@/hooks/use-web-speech";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
}

export const ChatInput = ({ onSendMessage, isLoading = false }: ChatInputProps) => {
  const [message, setMessage] = useState("");

  const {
    isListening,
    transcript,
    speechAvailable,
    startListening,
    stopListening,
    clearTranscript
  } = useWebSpeech();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message.trim());
      setMessage("");
    }
  };

  const handleMicClick = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  // Mettre à jour le message avec le transcript
  useEffect(() => {
    if (transcript && !isListening) {
      setMessage(transcript);
      clearTranscript();
    }
  }, [transcript, isListening, clearTranscript]);

  return (
    <div className="p-6 bg-gradient-to-r from-background/95 via-background to-background/95 border-t border-border/30 backdrop-blur-xl">
      <form onSubmit={handleSubmit} className="flex gap-4 items-end max-w-4xl mx-auto">
        {/* Bouton d'ajout de fichier */}
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-12 w-12 rounded-xl hover:bg-primary/10 transition-all duration-300 hover:scale-105 shadow-lg border border-border/50"
        >
          <PlusCircle className="h-5 w-5" />
        </Button>

        {/* Zone de saisie ultra-moderne */}
        <div className="flex-1 relative">
          <div className="relative">
            <Input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Posez votre question bancaire..."
              disabled={isLoading}
              className={cn(
                "h-12 pr-16 pl-4 bg-card/80 border-2 border-border/50 rounded-xl",
                "focus:border-primary/50 focus:ring-4 focus:ring-primary/10",
                "transition-all duration-300 shadow-lg backdrop-blur-sm",
                "placeholder:text-muted-foreground/60 text-base",
                isListening && "border-destructive/50 ring-4 ring-destructive/10"
              )}
            />
            
            {/* Indicateur d'écoute */}
            {isListening && (
              <div className="absolute inset-0 rounded-xl border-2 border-destructive/30 animate-pulse pointer-events-none" />
            )}

            {/* Bouton microphone intégré */}
            {speechAvailable && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={handleMicClick}
                className={cn(
                  "absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 rounded-lg",
                  "hover:bg-secondary/80 transition-all duration-200",
                  isListening 
                    ? "text-destructive bg-destructive/10 animate-pulse" 
                    : "hover:scale-110"
                )}
                disabled={isLoading}
              >
                {isListening ? (
                  <MicOff className="h-4 w-4" />
                ) : (
                  <Mic className="h-4 w-4" />
                )}
              </Button>
            )}
          </div>

          {/* Indicateur de transcription */}
          {isListening && (
            <div className="absolute -top-10 left-0 right-0 text-center">
              <div className="inline-flex items-center gap-2 text-xs text-destructive font-medium bg-background/90 px-3 py-1.5 rounded-lg backdrop-blur-sm border border-destructive/20 shadow-lg">
                <div className="w-2 h-2 bg-destructive rounded-full animate-pulse" />
                Écoute en cours...
              </div>
            </div>
          )}

          {/* Suggestions d'aide */}
          {!message && !isListening && (
            <div className="absolute -top-8 left-0 text-xs text-muted-foreground/80">
              💡 Essayez: "Quels sont les frais de virement?" ou cliquez sur le micro
            </div>
          )}
        </div>

        {/* Bouton d'envoi ultra-moderne */}
        <Button
          type="submit"
          disabled={!message.trim() || isLoading}
          className={cn(
            "h-12 px-8 rounded-xl font-medium transition-all duration-300",
            "bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70",
            "shadow-xl hover:shadow-2xl hover:scale-105",
            "disabled:opacity-50 disabled:scale-100 disabled:shadow-lg",
            "border border-primary/20"
          )}
        >
          {isLoading ? (
            <div className="flex items-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary-foreground border-t-transparent" />
              <span className="text-sm">Envoi...</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Send className="h-4 w-4" />
              <span className="text-sm font-medium">Envoyer</span>
            </div>
          )}
        </Button>
      </form>

      {/* Barre de statut moderne */}
      <div className="flex items-center justify-center mt-4 gap-4 text-xs text-muted-foreground/80">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          RAG Connecté
        </div>
        {speechAvailable && (
          <div className="flex items-center gap-1">
            <Sparkles className="h-3 w-3" />
            Speech API Active
          </div>
        )}
        <div className="text-xs">
          BankBot AI • Assistant bancaire intelligent
        </div>
      </div>
    </div>
  );
};
