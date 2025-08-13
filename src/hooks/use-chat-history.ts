import { useState, useEffect } from 'react';

export interface Message {
  id: string;
  content: string;
  isBot: boolean;
  timestamp: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

const STORAGE_KEY = 'chat_conversations';
const MAX_CONVERSATIONS = 50; // Limite pour éviter de surcharger le localStorage

export const useChatHistory = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);

  // Charger les conversations depuis localStorage au démarrage
  useEffect(() => {
    const savedConversations = localStorage.getItem(STORAGE_KEY);
    if (savedConversations) {
      try {
        const parsed = JSON.parse(savedConversations);
        setConversations(parsed);
      } catch (error) {
        console.error('Erreur lors du chargement de l\'historique:', error);
      }
    }
  }, []);

  // Sauvegarder les conversations dans localStorage
  const saveConversations = (convs: Conversation[]) => {
    try {
      console.log('💾 Sauvegarde dans localStorage:', {
        conversationsCount: convs.length,
        totalMessages: convs.reduce((sum, conv) => sum + conv.messages.length, 0),
        conversations: convs.map(conv => ({
          id: conv.id,
          title: conv.title,
          messagesCount: conv.messages.length,
          userMessages: conv.messages.filter(m => !m.isBot).length,
          botMessages: conv.messages.filter(m => m.isBot).length
        }))
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
      console.log('✅ Sauvegarde localStorage réussie');
    } catch (error) {
      console.error('❌ Erreur lors de la sauvegarde:', error);
    }
  };

  // Créer une nouvelle conversation
  const createNewConversation = (initialMessage?: Message): string => {
    console.log('🆕 createNewConversation appelé avec:', initialMessage ? {
      id: initialMessage.id,
      content: initialMessage.content.substring(0, 50),
      isBot: initialMessage.isBot
    } : 'aucun message initial');

    const now = new Date().toISOString();
    const newConversation: Conversation = {
      id: `conv_${Date.now()}`,
      title: initialMessage ?
        generateConversationTitle(initialMessage.content) :
        'Nouvelle conversation',
      messages: initialMessage ? [initialMessage] : [],
      createdAt: now,
      updatedAt: now
    };

    console.log('📋 Nouvelle conversation créée:', {
      id: newConversation.id,
      title: newConversation.title,
      messagesCount: newConversation.messages.length,
      messages: newConversation.messages
    });

    const updatedConversations = [newConversation, ...conversations].slice(0, MAX_CONVERSATIONS);
    setConversations(updatedConversations);
    setCurrentConversationId(newConversation.id);
    saveConversations(updatedConversations);

    console.log('✅ Conversation créée et définie comme courante:', newConversation.id);
    return newConversation.id;
  };

  // Nouvelle fonction d'ajout de message plus robuste
  const addMessageToCurrentConversation = (message: Message): Promise<string> => {
    console.log('🔄 addMessageToCurrentConversation appelé:', {
      message: { id: message.id, content: message.content.substring(0, 50), isBot: message.isBot },
      currentConversationId,
      conversationsCount: conversations.length
    });

    return new Promise<string>((resolve) => {
      if (!currentConversationId) {
        // Créer une nouvelle conversation si aucune n'est active
        console.log('🆕 Création d\'une nouvelle conversation car currentConversationId est null');
        const newId = createNewConversation(message);
        console.log('✅ Nouvelle conversation créée avec ID:', newId);
        resolve(newId);
        return;
      }

      console.log('📝 Ajout du message à la conversation existante:', currentConversationId);

      // Utiliser une fonction de mise à jour pour éviter les conflits d'état
      setConversations(prevConversations => {
        console.log('🔍 État précédent des conversations:', prevConversations.length);

        const updatedConversations = prevConversations.map(conv => {
          if (conv.id === currentConversationId) {
            // Vérifier si le message existe déjà pour éviter les doublons
            const messageExists = conv.messages.some(m => m.id === message.id);
            if (messageExists) {
              console.log('⚠️ Message déjà existant, ignoré:', message.id);
              return conv;
            }

            const updatedConv = {
              ...conv,
              messages: [...conv.messages, message],
              updatedAt: new Date().toISOString(),
              // Mettre à jour le titre si c'est le premier message utilisateur
              title: conv.messages.length === 1 && !message.isBot ?
                generateConversationTitle(message.content) :
                conv.title
            };

            console.log('🔄 Conversation mise à jour:', {
              id: updatedConv.id,
              title: updatedConv.title,
              messagesCount: updatedConv.messages.length,
              userMessages: updatedConv.messages.filter(m => !m.isBot).length,
              botMessages: updatedConv.messages.filter(m => m.isBot).length,
              lastMessage: updatedConv.messages[updatedConv.messages.length - 1]
            });

            return updatedConv;
          }
          return conv;
        });

        // Sauvegarder immédiatement
        saveConversations(updatedConversations);
        console.log('💾 Conversations sauvegardées, total:', updatedConversations.length);

        return updatedConversations;
      });

      resolve(currentConversationId);
    });
  };

  // Version synchrone pour les cas simples
  const addMessageToCurrentConversationSync = (message: Message): string => {
    console.log('🔄 addMessageToCurrentConversationSync appelé:', {
      message: { id: message.id, content: message.content.substring(0, 50), isBot: message.isBot },
      currentConversationId,
      conversationsCount: conversations.length
    });

    if (!currentConversationId) {
      // Créer une nouvelle conversation si aucune n'est active
      console.log('🆕 Création d\'une nouvelle conversation car currentConversationId est null');
      const newId = createNewConversation(message);
      console.log('✅ Nouvelle conversation créée avec ID:', newId);
      return newId;
    }

    console.log('📝 Ajout du message à la conversation existante:', currentConversationId);

    // Utiliser une fonction de mise à jour pour éviter les conflits d'état
    setConversations(prevConversations => {
      console.log('🔍 État précédent des conversations:', prevConversations.length);

      const updatedConversations = prevConversations.map(conv => {
        if (conv.id === currentConversationId) {
          // Vérifier si le message existe déjà pour éviter les doublons
          const messageExists = conv.messages.some(m => m.id === message.id);
          if (messageExists) {
            console.log('⚠️ Message déjà existant, ignoré:', message.id);
            return conv;
          }

          const updatedConv = {
            ...conv,
            messages: [...conv.messages, message],
            updatedAt: new Date().toISOString(),
            // Mettre à jour le titre si c'est le premier message utilisateur
            title: conv.messages.length === 1 && !message.isBot ?
              generateConversationTitle(message.content) :
              conv.title
          };

          console.log('🔄 Conversation mise à jour (sync):', {
            id: updatedConv.id,
            title: updatedConv.title,
            messagesCount: updatedConv.messages.length,
            userMessages: updatedConv.messages.filter(m => !m.isBot).length,
            botMessages: updatedConv.messages.filter(m => m.isBot).length,
            lastMessage: updatedConv.messages[updatedConv.messages.length - 1]
          });

          return updatedConv;
        }
        return conv;
      });

      // Sauvegarder immédiatement
      saveConversations(updatedConversations);
      console.log('💾 Conversations sauvegardées (sync), total:', updatedConversations.length);

      return updatedConversations;
    });

    return currentConversationId;
  };

  // Charger une conversation existante
  const loadConversation = (conversationId: string): Message[] => {
    const conversation = conversations.find(conv => conv.id === conversationId);
    if (conversation) {
      setCurrentConversationId(conversationId);
      return conversation.messages;
    }
    return [];
  };

  // Supprimer une conversation
  const deleteConversation = (conversationId: string) => {
    const updatedConversations = conversations.filter(conv => conv.id !== conversationId);
    setConversations(updatedConversations);
    saveConversations(updatedConversations);
    
    // Si la conversation supprimée était la conversation courante, la réinitialiser
    if (currentConversationId === conversationId) {
      setCurrentConversationId(null);
    }
  };

  // Renommer une conversation
  const renameConversation = (conversationId: string, newTitle: string) => {
    const updatedConversations = conversations.map(conv => 
      conv.id === conversationId ? 
        { ...conv, title: newTitle, updatedAt: new Date().toISOString() } : 
        conv
    );
    setConversations(updatedConversations);
    saveConversations(updatedConversations);
  };

  // Générer un titre automatique basé sur le premier message
  const generateConversationTitle = (firstMessage: string): string => {
    const words = firstMessage.split(' ').slice(0, 6);
    let title = words.join(' ');
    if (firstMessage.split(' ').length > 6) {
      title += '...';
    }
    return title || 'Nouvelle conversation';
  };

  // Obtenir la conversation courante
  const getCurrentConversation = (): Conversation | null => {
    if (!currentConversationId) return null;
    return conversations.find(conv => conv.id === currentConversationId) || null;
  };

  // Vider tout l'historique
  const clearAllHistory = () => {
    setConversations([]);
    setCurrentConversationId(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  return {
    conversations,
    currentConversationId,
    createNewConversation,
    addMessageToCurrentConversation,
    addMessageToCurrentConversationSync,
    loadConversation,
    deleteConversation,
    renameConversation,
    getCurrentConversation,
    clearAllHistory,
    setCurrentConversationId
  };
};
