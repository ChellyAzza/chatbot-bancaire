import { API_CONFIG } from '@/config/api';

interface HistorySearchResult {
  found: boolean;
  response?: string;
  confidence?: number;
  originalQuestion?: string;
  timestamp?: string;
  source: 'exact' | 'similar' | 'none';
}

interface HistoryEntry {
  id: string;
  question: string;
  response: string;
  timestamp: string;
  confidence: number;
  user_feedback?: 'positive' | 'negative' | 'neutral';
}

export class HistorySearchService {
  private static instance: HistorySearchService;
  private cache: Map<string, HistorySearchResult> = new Map();

  static getInstance(): HistorySearchService {
    if (!HistorySearchService.instance) {
      HistorySearchService.instance = new HistorySearchService();
    }
    return HistorySearchService.instance;
  }

  // Rechercher dans l'historique backend avec similarité sémantique
  async searchSimilarResponse(question: string, threshold: number = 0.85): Promise<HistorySearchResult> {
    try {
      // Vérifier le cache en mémoire d'abord
      const cacheKey = `${question}_${threshold}`;
      if (this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey)!;
      }

      const response = await fetch(`${API_CONFIG.BASE_URL}/api/history/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          threshold,
          max_results: 1
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const result: HistorySearchResult = {
        found: data.found || false,
        response: data.response,
        confidence: data.confidence,
        originalQuestion: data.original_question,
        timestamp: data.timestamp,
        source: data.found ? 'similar' : 'none'
      };

      // Mettre en cache le résultat
      this.cache.set(cacheKey, result);
      
      return result;
    } catch (error) {
      console.error('Erreur lors de la recherche dans l\'historique:', error);
      return {
        found: false,
        source: 'none'
      };
    }
  }

  // Sauvegarder une nouvelle interaction dans l'historique backend
  async saveToHistory(
    question: string, 
    response: string, 
    confidence: number = 0.95,
    responseTime: number = 0,
    context?: any
  ): Promise<boolean> {
    try {
      const historyResponse = await fetch(`${API_CONFIG.BASE_URL}/api/history/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          response,
          confidence,
          response_time: responseTime,
          timestamp: new Date().toISOString(),
          context,
          session_id: this.getSessionId()
        }),
      });

      return historyResponse.ok;
    } catch (error) {
      console.error('Erreur lors de la sauvegarde dans l\'historique:', error);
      return false;
    }
  }

  // Obtenir les questions fréquentes
  async getFrequentQuestions(limit: number = 10): Promise<HistoryEntry[]> {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/api/history/frequent?limit=${limit}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Erreur lors de la récupération des questions fréquentes:', error);
      return [];
    }
  }

  // Envoyer un feedback sur une réponse
  async submitFeedback(
    questionId: string, 
    feedback: 'positive' | 'negative' | 'neutral',
    comment?: string
  ): Promise<boolean> {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/api/history/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question_id: questionId,
          feedback,
          comment,
          timestamp: new Date().toISOString()
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Erreur lors de l\'envoi du feedback:', error);
      return false;
    }
  }

  // Obtenir les statistiques de l'historique
  async getHistoryStats(): Promise<{
    total_questions: number;
    cache_hit_rate: number;
    average_confidence: number;
    most_asked_topics: string[];
  }> {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/api/history/stats`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Erreur lors de la récupération des statistiques:', error);
      return {
        total_questions: 0,
        cache_hit_rate: 0,
        average_confidence: 0,
        most_asked_topics: []
      };
    }
  }

  // Recherche hybride : local + backend
  async hybridSearch(question: string): Promise<{
    result: HistorySearchResult;
    searchTime: number;
    source: 'local_exact' | 'local_similar' | 'backend_similar' | 'none';
  }> {
    const startTime = Date.now();
    
    // 1. Recherche exacte locale (instantanée)
    // Cette partie sera appelée depuis le composant qui utilise useQuestionCache
    
    // 2. Recherche similaire backend (si pas trouvé localement)
    const backendResult = await this.searchSimilarResponse(question, 0.85);
    
    const searchTime = Date.now() - startTime;
    
    return {
      result: backendResult,
      searchTime,
      source: backendResult.found ? 'backend_similar' : 'none'
    };
  }

  // Générer ou récupérer un ID de session
  private getSessionId(): string {
    let sessionId = sessionStorage.getItem('bankbot_session_id');
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('bankbot_session_id', sessionId);
    }
    return sessionId;
  }

  // Vider le cache en mémoire
  clearCache(): void {
    this.cache.clear();
  }
}

// Export de l'instance singleton
export const historySearchService = HistorySearchService.getInstance();
