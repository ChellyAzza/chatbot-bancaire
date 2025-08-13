import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import {
  History,
  MessageSquare,
  Trash2,
  Edit3,
  Plus,
  Search,
  Calendar,
  X,
  Check,
  Bug
} from "lucide-react";
import { Conversation } from "@/hooks/use-chat-history";

interface ChatHistoryProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onLoadConversation: (conversationId: string) => void;
  onDeleteConversation: (conversationId: string) => void;
  onRenameConversation: (conversationId: string, newTitle: string) => void;
  onNewConversation: () => void;
  onClearHistory: () => void;
  isOpen: boolean;
  onClose: () => void;
}

export const ChatHistory = ({
  conversations,
  currentConversationId,
  onLoadConversation,
  onDeleteConversation,
  onRenameConversation,
  onNewConversation,
  onClearHistory,
  isOpen,
  onClose
}: ChatHistoryProps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  const filteredConversations = conversations.filter(conv =>
    conv.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    conv.messages.some(msg => 
      msg.content.toLowerCase().includes(searchTerm.toLowerCase())
    )
  );

  const handleStartEdit = (conversation: Conversation) => {
    setEditingId(conversation.id);
    setEditingTitle(conversation.title);
  };

  const handleSaveEdit = () => {
    if (editingId && editingTitle.trim()) {
      onRenameConversation(editingId, editingTitle.trim());
    }
    setEditingId(null);
    setEditingTitle('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditingTitle('');
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) {
      return "Aujourd'hui";
    } else if (diffDays === 2) {
      return "Hier";
    } else if (diffDays <= 7) {
      return `Il y a ${diffDays - 1} jours`;
    } else {
      return date.toLocaleDateString('fr-FR', { 
        day: 'numeric', 
        month: 'short' 
      });
    }
  };

  const getMessagePreview = (conversation: Conversation) => {
    if (conversation.messages.length === 0) {
      return 'Aucun message';
    }

    // Prendre le dernier message (peu importe si c'est bot ou utilisateur)
    const lastMessage = conversation.messages[conversation.messages.length - 1];
    const prefix = lastMessage.isBot ? '🤖 ' : '👤 ';

    const content = lastMessage.content.length > 45
      ? lastMessage.content.substring(0, 45) + '...'
      : lastMessage.content;

    return prefix + content;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex">
      <div className="w-80 bg-background border-r border-border h-full flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <History className="h-5 w-5 text-primary" />
              <h2 className="font-semibold">Historique</h2>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0 hover:bg-primary/10 text-muted-foreground hover:text-foreground transition-colors"
              title="Fermer l'historique"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Actions */}
          <div className="space-y-2">
            <Button
              onClick={onNewConversation}
              className="w-full justify-start gap-2 hover:bg-primary/5 transition-colors"
              variant="outline"
            >
              <Plus className="h-4 w-4" />
              Nouvelle conversation
            </Button>

            {/* Recherche */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Rechercher..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>
        </div>

        {/* Liste des conversations */}
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {filteredConversations.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">
                  {searchTerm ? 'Aucune conversation trouvée' : 'Aucune conversation'}
                </p>
              </div>
            ) : (
              filteredConversations.map((conversation) => (
                <Card
                  key={conversation.id}
                  className={`group p-3 cursor-pointer transition-all duration-200 hover:bg-primary/5 hover:shadow-md ${
                    currentConversationId === conversation.id
                      ? 'bg-primary/10 border-primary/30 shadow-lg'
                      : 'bg-card/50 hover:border-primary/20'
                  }`}
                  onClick={() => onLoadConversation(conversation.id)}
                >
                  <div className="space-y-2">
                    {/* Titre */}
                    {editingId === conversation.id ? (
                      <div className="flex items-center gap-1">
                        <Input
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          className="h-6 text-sm"
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveEdit();
                            if (e.key === 'Escape') handleCancelEdit();
                          }}
                          autoFocus
                        />
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSaveEdit();
                          }}
                          className="h-6 w-6 p-0"
                        >
                          <Check className="h-3 w-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelEdit();
                          }}
                          className="h-6 w-6 p-0"
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    ) : (
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-sm truncate flex-1">
                          {conversation.title}
                        </h4>
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all duration-200">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              console.log('🔍 Debug Conversation:', conversation.id);
                              console.log('📝 Messages:', conversation.messages);
                              console.log('👤 Messages utilisateur:', conversation.messages.filter(m => !m.isBot));
                              console.log('🤖 Messages bot:', conversation.messages.filter(m => m.isBot));

                              // Debug détaillé de chaque message
                              conversation.messages.forEach((msg, index) => {
                                console.log(`Message ${index + 1}:`, {
                                  id: msg.id,
                                  content: msg.content.substring(0, 50) + '...',
                                  isBot: msg.isBot,
                                  timestamp: msg.timestamp
                                });
                              });

                              // Vérifier le localStorage
                              const stored = localStorage.getItem('chat_conversations');
                              console.log('💾 LocalStorage data:', stored ? JSON.parse(stored) : 'Aucune donnée');

                              alert(`Conversation: ${conversation.title}\nMessages: ${conversation.messages.length}\nUtilisateur: ${conversation.messages.filter(m => !m.isBot).length}\nBot: ${conversation.messages.filter(m => m.isBot).length}\n\nVoir la console pour plus de détails`);
                            }}
                            className="h-6 w-6 p-0 hover:bg-blue-500/10 text-muted-foreground hover:text-blue-500 transition-colors"
                            title="Debug conversation"
                          >
                            <Bug className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartEdit(conversation);
                            }}
                            className="h-6 w-6 p-0 hover:bg-primary/10 text-muted-foreground hover:text-foreground transition-colors"
                            title="Renommer la conversation"
                          >
                            <Edit3 className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              if (confirm('Êtes-vous sûr de vouloir supprimer cette conversation ?')) {
                                onDeleteConversation(conversation.id);
                              }
                            }}
                            className="h-6 w-6 p-0 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                            title="Supprimer la conversation"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    )}

                    {/* Aperçu et date */}
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground truncate">
                        {getMessagePreview(conversation)}
                      </p>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Calendar className="h-3 w-3" />
                        {formatDate(conversation.updatedAt)}
                        <span className="ml-auto flex items-center gap-2">
                          <span>
                            {conversation.messages.filter(m => !m.isBot).length}👤 / {conversation.messages.filter(m => m.isBot).length}🤖
                          </span>
                          <span className="text-muted-foreground/60">
                            ({conversation.messages.length} total)
                          </span>
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        {conversations.length > 0 && (
          <div className="p-4 border-t border-border">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (confirm('Êtes-vous sûr de vouloir supprimer tout l\'historique ? Cette action est irréversible.')) {
                  onClearHistory();
                }
              }}
              className="w-full text-destructive hover:text-destructive hover:bg-destructive/5 transition-colors"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Vider l'historique
            </Button>
          </div>
        )}
      </div>

      {/* Zone cliquable pour fermer */}
      <div className="flex-1" onClick={onClose} />
    </div>
  );
};
