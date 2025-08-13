import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Bot, Layout, Sidebar, History, Plus, Bug } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

interface ChatHeaderProps {
  showQuickActionsTop?: boolean;
  onToggleQuickActions?: () => void;
  onOpenHistory?: () => void;
  onNewConversation?: () => void;
  onToggleDiagnostic?: () => void;
}

export const ChatHeader = ({
  showQuickActionsTop = true,
  onToggleQuickActions,
  onOpenHistory,
  onNewConversation,
  onToggleDiagnostic
}: ChatHeaderProps) => {
  return (
    <div className="border-b border-border/30 bg-gradient-to-r from-background via-background/95 to-background backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Avatar className="h-11 w-11 ring-2 ring-primary/20 shadow-lg">
                <AvatarFallback className="bg-gradient-to-br from-blue-500 to-purple-600">
                  <Bot className="h-5 w-5 text-white" />
                </AvatarFallback>
              </Avatar>
              {/* Indicateur de statut moderne */}
              <div className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 bg-green-500 rounded-full border-2 border-background">
                <div className="w-full h-full bg-green-400 rounded-full animate-pulse" />
              </div>
            </div>

            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text">
                  BankBot AI
                </h1>
                <div className="px-2 py-0.5 bg-primary/10 text-primary text-xs font-medium rounded-full border border-primary/20">
                  Pro
                </div>
              </div>
              <p className="text-sm text-muted-foreground/80">
                Assistant bancaire intelligent • IA avancée
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Bouton Nouvelle conversation */}
            {onNewConversation && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onNewConversation}
                className="h-9 w-9 rounded-xl hover:bg-primary/10 hover:scale-105 transition-all duration-200 shadow-sm"
                title="Nouvelle conversation"
              >
                <Plus className="h-4 w-4" />
              </Button>
            )}

            {/* Bouton Historique */}
            {onOpenHistory && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onOpenHistory}
                className="h-9 w-9 rounded-xl hover:bg-secondary/10 hover:scale-105 transition-all duration-200 shadow-sm"
                title="Historique des conversations"
              >
                <History className="h-4 w-4" />
              </Button>
            )}

            {/* Bouton Diagnostic (mode développement) */}
            {onToggleDiagnostic && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onToggleDiagnostic}
                className="h-9 w-9 rounded-xl hover:bg-orange-500/10 hover:scale-105 transition-all duration-200 shadow-sm"
                title="Diagnostic de l'historique"
              >
                <Bug className="h-4 w-4" />
              </Button>
            )}

            <div className="w-px h-5 bg-gradient-to-b from-transparent via-border to-transparent mx-2" />
            <ThemeToggle />
          </div>
        </div>
      </div>
    </div>
  );
};