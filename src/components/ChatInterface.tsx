import { useState, useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { ChatHeader } from "./ChatHeader";
import { QuickActions } from "./QuickActions";
import { QuickActionsSidebar } from "./QuickActionsSidebar";
import { ChatHistory } from "./ChatHistory";
import { HistoryDiagnostic } from "./HistoryDiagnostic";
import { SimpleHistoryTest } from "./SimpleHistoryTest";

import { useToast } from "@/hooks/use-toast";
import { useChatHistory, Message } from "@/hooks/use-chat-history";
import { useHybridSearch } from "@/hooks/use-hybrid-search";

const getInitialMessage = (): Message => ({
  id: "1",
  content: "Bonjour ! Je suis votre assistant bancaire intelligent. Comment puis-je vous aider aujourd'hui ?",
  isBot: true,
  timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
});

export const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([getInitialMessage()]);
  const [isLoading, setIsLoading] = useState(false);
  const [showQuickActionsTop, setShowQuickActionsTop] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [showDiagnostic, setShowDiagnostic] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Hook pour l'historique des conversations
  const {
    conversations,
    currentConversationId,
    createNewConversation,
    addMessageToCurrentConversation,
    loadConversation,
    deleteConversation,
    renameConversation,
    getCurrentConversation,
    clearAllHistory,
    setCurrentConversationId
  } = useChatHistory();

  // Hook pour la recherche hybride
  const {
    searchResponse,
    saveNewResponse,
    getCompleteStats,
    isSearching
  } = useHybridSearch();

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Raccourci clavier pour debug (Ctrl+D)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        debugCurrentState();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Initialiser automatiquement la première conversation
  useEffect(() => {
    console.log('🔄 Effet d\'initialisation:', { isInitialized, currentConversationId, conversationsCount: conversations.length });

    if (!isInitialized && !currentConversationId) {
      console.log('🚀 Initialisation automatique de la première conversation');
      const initialMessage = getInitialMessage();
      console.log('📝 Message initial créé:', initialMessage);
      const newConvId = createNewConversation(initialMessage);
      console.log('✅ Première conversation initialisée:', newConvId);
      setIsInitialized(true);

      // Vérifier immédiatement
      setTimeout(() => {
        const current = getCurrentConversation();
        console.log('🔍 Conversation après initialisation:', current);
      }, 100);
    }
  }, [isInitialized, currentConversationId, createNewConversation, conversations.length]);

  const handleSendMessage = async (content: string) => {
    console.log('🚀 === DÉBUT handleSendMessage ===');
    console.log('📝 Contenu du message:', content);
    console.log('🔍 État avant envoi:', {
      currentConversationId,
      conversationsCount: conversations.length,
      isInitialized,
      messagesCount: messages.length
    });

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      isBot: false,
      timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
    };

    console.log('👤 Message utilisateur créé:', userMessage);

    setMessages(prev => {
      console.log('📱 Mise à jour état local messages:', prev.length, '->', prev.length + 1);
      return [...prev, userMessage];
    });
    setIsLoading(true);

    // Sauvegarder le message utilisateur dans l'historique
    console.log('💾 === DÉBUT SAUVEGARDE HISTORIQUE ===');
    console.log('🔍 État historique avant sauvegarde:', {
      currentConversationId,
      conversations: conversations.map(c => ({
        id: c.id,
        title: c.title,
        messagesCount: c.messages.length
      }))
    });

    const conversationId = await addMessageToCurrentConversation(userMessage);
    console.log('📝 Résultat addMessageToCurrentConversation:', conversationId);

    // Debug: vérifier l'état de l'historique après ajout
    setTimeout(() => {
      const currentConv = getCurrentConversation();
      console.log('🔍 Conversation actuelle après ajout utilisateur:', currentConv);
      console.log('💾 localStorage après ajout:', localStorage.getItem('chat_conversations'));
    }, 50);

    try {
      // 🔍 RECHERCHE HYBRIDE AVANT RAG PIPELINE
      console.log('🚀 Démarrage de la recherche hybride...');
      const hybridResult = await searchResponse(content);

      if (hybridResult.found && hybridResult.response) {
        // ✅ RÉPONSE TROUVÉE DANS L'HISTORIQUE
        const botResponse: Message = {
          id: (Date.now() + 1).toString(),
          content: hybridResult.response,
          isBot: true,
          timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
        };

        setMessages(prev => [...prev, botResponse]);
        console.log('💾 Sauvegarde de la réponse bot (cache):', botResponse);

        // Attendre que le message utilisateur soit sauvegardé, puis ajouter le bot
        await addMessageToCurrentConversation(botResponse);

        // Debug: vérifier l'état après ajout bot
        const currentConvAfterBot = getCurrentConversation();
        console.log('🔍 Conversation après ajout bot (cache):', currentConvAfterBot);

        // Toast spécifique selon la source
        const sourceLabels = {
          'local_exact': '⚡ Cache local (exact)',
          'local_similar': '🔍 Cache local (similaire)',
          'backend_similar': '🧠 IA historique'
        };

        toast({
          title: sourceLabels[hybridResult.source as keyof typeof sourceLabels] || "Réponse trouvée",
          description: `Temps: ${hybridResult.searchTime}ms | Confiance: ${Math.round((hybridResult.confidence || 0) * 100)}%`,
        });

        setIsLoading(false);
        return;
      }

      // 🤖 RAG PIPELINE COMPLET (si aucune correspondance trouvée)
      console.log('🤖 Aucune correspondance - utilisation du RAG Pipeline...');

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          conversation_id: `conv_${Date.now()}`
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        isBot: true,
        timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, botResponse]);
      console.log('💾 Sauvegarde de la réponse bot (RAG):', botResponse);

      // Attendre que le message utilisateur soit sauvegardé, puis ajouter le bot
      await addMessageToCurrentConversation(botResponse);

      // Debug: vérifier l'état après ajout bot RAG
      const currentConvAfterRAG = getCurrentConversation();
      console.log('🔍 Conversation après ajout bot (RAG):', currentConvAfterRAG);

      // 💾 SAUVEGARDER LA NOUVELLE RÉPONSE DANS L'HISTORIQUE
      await saveNewResponse(content, data.response, data.similarity_score || 0.95, data.response_time || 0);

      // Afficher les métriques de performance
      toast({
        title: "🤖 Nouvelle réponse générée",
        description: `Temps: ${data.response_time.toFixed(2)}s | Contextes: ${data.contexts_found} | Similarité: ${(data.similarity_score * 100).toFixed(1)}%`,
      });

    } catch (error) {
      console.error('Erreur API:', error);

      // Fallback vers réponse locale
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: "Désolé, je rencontre un problème technique. Veuillez réessayer dans quelques instants.",
        isBot: true,
        timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, botResponse]);

      // Sauvegarder la réponse d'erreur dans l'historique
      await addMessageToCurrentConversation(botResponse);

      toast({
        title: "Erreur de connexion",
        description: "Impossible de contacter le serveur RAG. Mode hors ligne activé.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getBotResponse = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    
    if (lowerMessage.includes('solde')) {
      return "Votre solde actuel est de 2 847,50 €. Votre compte présente une évolution positive de +127,30 € ce mois-ci.";
    } else if (lowerMessage.includes('virement')) {
      return "Pour effectuer un virement, vous pouvez utiliser notre application mobile ou vous rendre dans l'espace client web. Avez-vous besoin d'aide pour un virement spécifique ?";
    } else if (lowerMessage.includes('épargne')) {
      return "Vous avez accès à plusieurs produits d'épargne : Livret A (3%), LDD (3%), et PEL (2,20%). Souhaitez-vous plus d'informations sur l'un d'entre eux ?";
    } else if (lowerMessage.includes('investissement')) {
      return "Nos conseillers recommandent une diversification de portefeuille. Nous proposons des assurances-vie, PEA, et investissements durables. Quel est votre profil de risque ?";
    } else if (lowerMessage.includes('sécurité')) {
      return "Votre sécurité est notre priorité. Utilisez toujours l'authentification forte, ne partagez jamais vos codes, et vérifiez régulièrement vos comptes.";
    } else {
      return "Je comprends votre demande. Pour vous fournir une réponse plus précise, pourriez-vous me donner plus de détails ? Je suis là pour vous aider avec tous vos besoins bancaires.";
    }
  };

  const handleQuickAction = (action: string) => {
    handleSendMessage(action);
    toast({
      title: "Question envoyée",
      description: "Votre question a été transmise à l'assistant.",
    });
  };

  // Fonctions pour gérer l'historique
  const handleNewConversation = () => {
    const initialMessage = getInitialMessage();
    setMessages([initialMessage]);
    createNewConversation(initialMessage);
    setHistoryOpen(false);
    toast({
      title: "Nouvelle conversation",
      description: "Une nouvelle conversation a été créée.",
    });
  };

  const handleLoadConversation = (conversationId: string) => {
    const conversationMessages = loadConversation(conversationId);
    console.log('🔄 Chargement conversation:', conversationId);
    console.log('📝 Messages trouvés:', conversationMessages.length);
    console.log('💬 Détail messages:', conversationMessages);

    if (conversationMessages.length > 0) {
      setMessages(conversationMessages);
      toast({
        title: "Conversation chargée",
        description: `${conversationMessages.length} messages restaurés.`,
      });
    } else {
      setMessages([getInitialMessage()]);
      toast({
        title: "Conversation vide",
        description: "Aucun message trouvé dans cette conversation.",
        variant: "destructive"
      });
    }
    setHistoryOpen(false);
  };

  const handleDeleteConversation = (conversationId: string) => {
    deleteConversation(conversationId);
    toast({
      title: "Conversation supprimée",
      description: "La conversation a été supprimée de l'historique.",
    });
  };

  const handleClearHistory = () => {
    clearAllHistory();
    setHistoryOpen(false);
    toast({
      title: "Historique vidé",
      description: "Tout l'historique des conversations a été supprimé.",
    });
  };

  // Debug rapide de l'état
  const debugCurrentState = () => {
    console.log('🔍 === DEBUG ÉTAT ACTUEL ===');
    console.log('currentConversationId:', currentConversationId);
    console.log('conversations:', conversations);
    console.log('messages (état local):', messages);
    console.log('isInitialized:', isInitialized);
    console.log('localStorage:', localStorage.getItem('chat_conversations'));

    const current = getCurrentConversation();
    console.log('getCurrentConversation():', current);
  };

  // Obtenir le dernier message du bot pour la synthèse vocale
  const getLastBotMessage = (): string | undefined => {
    const botMessages = messages.filter(msg => msg.isBot);
    return botMessages.length > 0 ? botMessages[botMessages.length - 1].content : undefined;
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      <ChatHeader
        showQuickActionsTop={showQuickActionsTop}
        onToggleQuickActions={() => {
          setShowQuickActionsTop(!showQuickActionsTop);
          if (!showQuickActionsTop) {
            setSidebarOpen(true);
          }
        }}
        onOpenHistory={() => setHistoryOpen(true)}
        onNewConversation={handleNewConversation}
        onToggleDiagnostic={() => setShowDiagnostic(!showDiagnostic)}
      />

      {/* Section QuickActions toujours visible */}
      {showQuickActionsTop && (
        <div className="border-b border-border/50 bg-background/95 backdrop-blur-sm">
          <div className="max-w-4xl mx-auto p-4">
            <QuickActions onActionClick={handleQuickAction} />
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col min-h-0">
        <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
          <div className="max-w-4xl mx-auto space-y-1">
            
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message.content}
                isBot={message.isBot}
                timestamp={message.timestamp}
                confidence={message.isBot ? 0.95 : undefined}
                responseTime={message.isBot ? Math.random() * 2 + 0.5 : undefined}
              />
            ))}
            
            {isLoading && (
              <div className="flex gap-4 p-4">
                <div className="h-10 w-10 rounded-full bg-gradient-primary flex items-center justify-center">
                  <div className="h-2 w-2 bg-primary-foreground rounded-full animate-pulse"></div>
                </div>
                <div className="bg-card/80 rounded-lg p-4 max-w-md">
                  <div className="flex gap-1">
                    <div className="h-2 w-2 bg-primary rounded-full animate-bounce"></div>
                    <div className="h-2 w-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="h-2 w-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>



        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading || isSearching} />
      </div>

      {/* Sidebar pour les actions rapides (alternative) */}
      {!showQuickActionsTop && (
        <QuickActionsSidebar
          onActionClick={handleQuickAction}
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />
      )}

      {/* Historique des conversations */}
      <ChatHistory
        conversations={conversations}
        currentConversationId={currentConversationId}
        onLoadConversation={handleLoadConversation}
        onDeleteConversation={handleDeleteConversation}
        onRenameConversation={renameConversation}
        onNewConversation={handleNewConversation}
        onClearHistory={handleClearHistory}
        isOpen={historyOpen}
        onClose={() => setHistoryOpen(false)}
      />

      {/* Diagnostic de l'historique */}
      {showDiagnostic && (
        <div className="fixed top-20 right-4 z-50 space-y-4">
          <HistoryDiagnostic />
          <SimpleHistoryTest />
        </div>
      )}
    </div>
  );
};