import { useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bot, User, Volume2, VolumeX, Copy, Check, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWebSpeech } from "@/hooks/use-web-speech";
import { useToast } from "@/hooks/use-toast";

interface ChatMessageProps {
  message: string;
  isBot: boolean;
  timestamp: string;
  confidence?: number;
  responseTime?: number;
}

export const ChatMessage = ({
  message,
  isBot,
  timestamp,
  confidence,
  responseTime
}: ChatMessageProps) => {
  const [copied, setCopied] = useState(false);
  const { speak, stopSpeaking, isSpeaking, speechAvailable } = useWebSpeech();
  const { toast } = useToast();

  const handleSpeak = () => {
    if (isSpeaking) {
      stopSpeaking();
    } else {
      speak(message);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message);
      setCopied(true);
      toast({
        title: "Copié !",
        description: "Message copié dans le presse-papiers",
      });
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      toast({
        title: "Erreur",
        description: "Impossible de copier le message",
        variant: "destructive",
      });
    }
  };

  return (
    <div className={cn(
      "group flex gap-4 p-4 animate-in slide-in-from-bottom-2 duration-500",
      isBot ? "justify-start" : "justify-end"
    )}>
      {isBot && (
        <Avatar className="h-12 w-12 ring-2 ring-primary/20 shadow-lg">
          <AvatarFallback className="bg-gradient-to-br from-blue-500 to-purple-600">
            <Bot className="h-6 w-6 text-white" />
          </AvatarFallback>
        </Avatar>
      )}

      <div className="flex flex-col gap-2 max-w-2xl">
        <Card className={cn(
          "p-4 backdrop-blur-sm border-border/50 transition-all duration-300 hover:shadow-xl",
          isBot
            ? "bg-gradient-to-br from-card/90 to-card/70 hover:from-card to-card/80 shadow-lg border-l-4 border-l-primary/30"
            : "bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-xl"
        )}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message}</p>

          {/* Métadonnées pour les messages du bot */}
          {isBot && (confidence || responseTime) && (
            <div className="flex gap-2 mt-3 pt-2 border-t border-border/30">
              {confidence && (
                <Badge variant="secondary" className="text-xs">
                  <Sparkles className="h-3 w-3 mr-1" />
                  {Math.round(confidence * 100)}% confiance
                </Badge>
              )}
              {responseTime && (
                <Badge variant="outline" className="text-xs">
                  {responseTime.toFixed(1)}s
                </Badge>
              )}
            </div>
          )}
        </Card>

        {/* Contrôles intégrés pour les messages du bot */}
        {isBot && (
          <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            {/* Bouton Lecture */}
            {speechAvailable && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSpeak}
                className={cn(
                  "h-8 px-3 text-xs text-muted-foreground hover:text-foreground hover:bg-primary/10 transition-colors",
                  isSpeaking && "bg-primary/20 text-primary hover:text-primary"
                )}
              >
                {isSpeaking ? (
                  <>
                    <VolumeX className="h-3 w-3 mr-1" />
                    Arrêter
                  </>
                ) : (
                  <>
                    <Volume2 className="h-3 w-3 mr-1" />
                    Lire
                  </>
                )}
              </Button>
            )}

            {/* Bouton Copier */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="h-8 px-3 text-xs text-muted-foreground hover:text-foreground hover:bg-secondary/10 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 mr-1 text-green-500" />
                  Copié
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3 mr-1" />
                  Copier
                </>
              )}
            </Button>

            {/* Timestamp */}
            <span className="text-xs text-muted-foreground ml-auto">
              {timestamp}
            </span>
          </div>
        )}

        {/* Timestamp pour les messages utilisateur */}
        {!isBot && (
          <span className="text-xs text-muted-foreground self-end">
            {timestamp}
          </span>
        )}
      </div>

      {!isBot && (
        <Avatar className="h-12 w-12 ring-2 ring-accent/30 shadow-lg">
          <AvatarFallback className="bg-gradient-to-br from-green-500 to-blue-500">
            <User className="h-6 w-6 text-white" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
};