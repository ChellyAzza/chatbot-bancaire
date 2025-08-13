import React from 'react';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useChatHistory } from "@/hooks/use-chat-history";

export const HistoryDiagnostic = () => {
  const { conversations, getCurrentConversation, addMessageToCurrentConversationSync } = useChatHistory();

  const runDiagnostic = () => {
    console.log('🔍 === DIAGNOSTIC DE L\'HISTORIQUE ===');
    
    // 1. Vérifier le localStorage
    const stored = localStorage.getItem('chat_conversations');
    console.log('💾 Données localStorage:', stored ? JSON.parse(stored) : 'Aucune donnée');
    
    // 2. Vérifier l'état React
    console.log('⚛️ État React conversations:', conversations);
    
    // 3. Conversation actuelle
    const current = getCurrentConversation();
    console.log('📝 Conversation actuelle:', current);
    
    // 4. Analyser chaque conversation
    conversations.forEach((conv, index) => {
      console.log(`\n📋 Conversation ${index + 1}:`, {
        id: conv.id,
        title: conv.title,
        totalMessages: conv.messages.length,
        userMessages: conv.messages.filter(m => !m.isBot).length,
        botMessages: conv.messages.filter(m => m.isBot).length,
        createdAt: conv.createdAt,
        updatedAt: conv.updatedAt
      });
      
      // Détail des messages
      conv.messages.forEach((msg, msgIndex) => {
        console.log(`  Message ${msgIndex + 1}:`, {
          id: msg.id,
          isBot: msg.isBot,
          content: msg.content.substring(0, 50) + '...',
          timestamp: msg.timestamp
        });
      });
    });
    
    // 5. Résumé
    const totalMessages = conversations.reduce((sum, conv) => sum + conv.messages.length, 0);
    const totalUserMessages = conversations.reduce((sum, conv) => 
      sum + conv.messages.filter(m => !m.isBot).length, 0);
    const totalBotMessages = conversations.reduce((sum, conv) => 
      sum + conv.messages.filter(m => m.isBot).length, 0);
    
    console.log('\n📊 RÉSUMÉ:');
    console.log(`Total conversations: ${conversations.length}`);
    console.log(`Total messages: ${totalMessages}`);
    console.log(`Messages utilisateur: ${totalUserMessages}`);
    console.log(`Messages bot: ${totalBotMessages}`);
    
    // Afficher aussi dans une alerte
    alert(`DIAGNOSTIC HISTORIQUE:
    
Conversations: ${conversations.length}
Messages total: ${totalMessages}
Messages utilisateur: ${totalUserMessages}
Messages bot: ${totalBotMessages}

Voir la console pour plus de détails`);
  };

  const clearHistory = () => {
    if (confirm('Êtes-vous sûr de vouloir vider tout l\'historique ?')) {
      localStorage.removeItem('chat_conversations');
      window.location.reload();
    }
  };

  const testAddUserMessage = () => {
    const testMessage = {
      id: `test_${Date.now()}`,
      content: "Message de test utilisateur",
      isBot: false,
      timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
    };

    console.log('🧪 Test: Ajout forcé d\'un message utilisateur:', testMessage);
    addMessageToCurrentConversationSync(testMessage);

    // Vérifier immédiatement
    setTimeout(() => {
      runDiagnostic();
    }, 100);
  };

  const testDirectLocalStorage = () => {
    console.log('🧪 Test: Modification directe du localStorage');

    // Récupérer les données actuelles
    const stored = localStorage.getItem('chat_conversations');
    const conversations = stored ? JSON.parse(stored) : [];

    console.log('📦 Données actuelles:', conversations);

    // Ajouter un message utilisateur à la première conversation
    if (conversations.length > 0) {
      const testMessage = {
        id: `direct_test_${Date.now()}`,
        content: "Message test direct localStorage",
        isBot: false,
        timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
      };

      conversations[0].messages.push(testMessage);
      conversations[0].updatedAt = new Date().toISOString();

      localStorage.setItem('chat_conversations', JSON.stringify(conversations));
      console.log('✅ Message ajouté directement au localStorage');

      // Recharger la page pour voir l'effet
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    } else {
      console.log('❌ Aucune conversation trouvée');
    }
  };

  return (
    <Card className="p-4 m-4">
      <h3 className="text-lg font-semibold mb-4">🔍 Diagnostic de l'historique</h3>
      <div className="space-y-2">
        <Button onClick={runDiagnostic} variant="outline" className="w-full">
          Analyser l'historique
        </Button>
        <Button onClick={testAddUserMessage} variant="secondary" className="w-full">
          🧪 Test: Ajouter message utilisateur
        </Button>
        <Button onClick={testDirectLocalStorage} variant="outline" className="w-full">
          🔧 Test: Modification directe localStorage
        </Button>
        <Button onClick={clearHistory} variant="destructive" className="w-full">
          Vider l'historique
        </Button>
        <div className="text-sm text-muted-foreground">
          <p>Conversations: {conversations.length}</p>
          <p>Messages total: {conversations.reduce((sum, conv) => sum + conv.messages.length, 0)}</p>
        </div>
      </div>
    </Card>
  );
};
