import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useChatHistory, Message } from "@/hooks/use-chat-history";

export const SimpleHistoryTest = () => {
  const [testMessage, setTestMessage] = useState('');
  const {
    conversations,
    currentConversationId,
    addMessageToCurrentConversationSync,
    createNewConversation,
    getCurrentConversation
  } = useChatHistory();

  const addUserMessage = () => {
    if (!testMessage.trim()) return;

    const userMessage: Message = {
      id: `test_user_${Date.now()}`,
      content: testMessage,
      isBot: false,
      timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
    };

    console.log('🧪 TEST: Ajout message utilisateur:', userMessage);
    console.log('🧪 TEST: currentConversationId avant:', currentConversationId);
    console.log('🧪 TEST: conversations avant:', conversations);

    const result = addMessageToCurrentConversationSync(userMessage);
    
    console.log('🧪 TEST: Résultat addMessage:', result);
    console.log('🧪 TEST: currentConversationId après:', currentConversationId);
    
    // Vérifier immédiatement
    setTimeout(() => {
      const current = getCurrentConversation();
      console.log('🧪 TEST: Conversation actuelle après:', current);
      
      const stored = localStorage.getItem('chat_conversations');
      console.log('🧪 TEST: localStorage après:', stored ? JSON.parse(stored) : null);
    }, 100);

    setTestMessage('');
  };

  const addBotMessage = () => {
    const botMessage: Message = {
      id: `test_bot_${Date.now()}`,
      content: "Réponse automatique du bot",
      isBot: true,
      timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
    };

    console.log('🤖 TEST: Ajout message bot:', botMessage);
    addMessageToCurrentConversationSync(botMessage);
  };

  const createTestConversation = () => {
    const initialMessage: Message = {
      id: `test_init_${Date.now()}`,
      content: "Message initial de test",
      isBot: true,
      timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
    };

    console.log('🆕 TEST: Création nouvelle conversation');
    const newId = createNewConversation(initialMessage);
    console.log('🆕 TEST: Nouvelle conversation créée:', newId);
  };

  const analyzeState = () => {
    console.log('🔍 ANALYSE ÉTAT:');
    console.log('- conversations:', conversations);
    console.log('- currentConversationId:', currentConversationId);
    console.log('- localStorage:', localStorage.getItem('chat_conversations'));
    
    const totalMessages = conversations.reduce((sum, conv) => sum + conv.messages.length, 0);
    const userMessages = conversations.reduce((sum, conv) => 
      sum + conv.messages.filter(m => !m.isBot).length, 0);
    const botMessages = conversations.reduce((sum, conv) => 
      sum + conv.messages.filter(m => m.isBot).length, 0);

    alert(`ÉTAT ACTUEL:
Conversations: ${conversations.length}
Messages total: ${totalMessages}
Messages utilisateur: ${userMessages}
Messages bot: ${botMessages}
Conversation courante: ${currentConversationId || 'Aucune'}`);
  };

  return (
    <Card className="p-4 m-4 max-w-md">
      <h3 className="text-lg font-semibold mb-4">🧪 Test Simple Historique</h3>
      
      <div className="space-y-3">
        <div className="flex gap-2">
          <Input
            value={testMessage}
            onChange={(e) => setTestMessage(e.target.value)}
            placeholder="Message de test..."
            onKeyPress={(e) => e.key === 'Enter' && addUserMessage()}
          />
          <Button onClick={addUserMessage} size="sm">
            👤 User
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button onClick={addBotMessage} variant="outline" size="sm">
            🤖 Bot
          </Button>
          <Button onClick={createTestConversation} variant="outline" size="sm">
            🆕 Conv
          </Button>
        </div>

        <Button onClick={analyzeState} variant="secondary" className="w-full">
          🔍 Analyser État
        </Button>

        <div className="text-xs text-muted-foreground space-y-1">
          <div>Conversations: {conversations.length}</div>
          <div>Conv. courante: {currentConversationId || 'Aucune'}</div>
          <div>Messages total: {conversations.reduce((sum, conv) => sum + conv.messages.length, 0)}</div>
          <div>Messages user: {conversations.reduce((sum, conv) => sum + conv.messages.filter(m => !m.isBot).length, 0)}</div>
        </div>
      </div>
    </Card>
  );
};
