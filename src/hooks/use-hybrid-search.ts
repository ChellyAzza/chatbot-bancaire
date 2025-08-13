import { useState, useCallback } from 'react';
import { useQuestionCache } from './use-question-cache';
import { historySearchService } from '@/services/history-search';
import { useToast } from './use-toast';

interface HybridSearchResult {
  found: boolean;
  response?: string;
  confidence?: number;
  source: 'local_exact' | 'local_similar' | 'backend_similar' | 'rag_pipeline';
  searchTime: number;
  originalQuestion?: string;
}

interface SearchStats {
  localExactHits: number;
  localSimilarHits: number;
  backendSimilarHits: number;
  ragPipelineUses: number;
  totalSearches: number;
  averageSearchTime: number;
}

export const useHybridSearch = () => {
  const { toast } = useToast();
  const {
    findExactMatch,
    findSimilarMatch,
    cacheResponse,
    getCacheStats
  } = useQuestionCache();

  const [searchStats, setSearchStats] = useState<SearchStats>({
    localExactHits: 0,
    localSimilarHits: 0,
    backendSimilarHits: 0,
    ragPipelineUses: 0,
    totalSearches: 0,
    averageSearchTime: 0
  });

  const [isSearching, setIsSearching] = useState(false);

  // Recherche hybride complète
  const searchResponse = useCallback(async (question: string): Promise<HybridSearchResult> => {
    const startTime = Date.now();
    setIsSearching(true);

    try {
      // 1. NIVEAU 1: Recherche exacte locale (0-5ms)
      console.log('🔍 Recherche exacte locale...');
      const exactMatch = findExactMatch(question);
      
      if (exactMatch) {
        const searchTime = Date.now() - startTime;
        
        setSearchStats(prev => ({
          ...prev,
          localExactHits: prev.localExactHits + 1,
          totalSearches: prev.totalSearches + 1,
          averageSearchTime: ((prev.averageSearchTime * prev.totalSearches) + searchTime) / (prev.totalSearches + 1)
        }));

        toast({
          title: "⚡ Réponse instantanée",
          description: `Trouvée dans le cache local (${searchTime}ms)`,
        });

        return {
          found: true,
          response: exactMatch.response,
          confidence: exactMatch.confidence,
          source: 'local_exact',
          searchTime,
          originalQuestion: exactMatch.question
        };
      }

      // 2. NIVEAU 2: Recherche similaire locale (5-20ms)
      console.log('🔍 Recherche similaire locale...');
      const similarMatch = findSimilarMatch(question);
      
      if (similarMatch) {
        const searchTime = Date.now() - startTime;
        
        setSearchStats(prev => ({
          ...prev,
          localSimilarHits: prev.localSimilarHits + 1,
          totalSearches: prev.totalSearches + 1,
          averageSearchTime: ((prev.averageSearchTime * prev.totalSearches) + searchTime) / (prev.totalSearches + 1)
        }));

        return {
          found: true,
          response: similarMatch.response,
          confidence: similarMatch.confidence * 0.9, // Réduire légèrement la confiance
          source: 'local_similar',
          searchTime,
          originalQuestion: similarMatch.question
        };
      }

      // 3. NIVEAU 3: Recherche backend avec IA (100-500ms)
      console.log('🔍 Recherche backend avec IA...');
      const backendResult = await historySearchService.searchSimilarResponse(question, 0.85);
      
      if (backendResult.found && backendResult.response) {
        const searchTime = Date.now() - startTime;
        
        // Sauvegarder dans le cache local pour les prochaines fois
        cacheResponse(question, backendResult.response, backendResult.confidence || 0.85, searchTime);
        
        setSearchStats(prev => ({
          ...prev,
          backendSimilarHits: prev.backendSimilarHits + 1,
          totalSearches: prev.totalSearches + 1,
          averageSearchTime: ((prev.averageSearchTime * prev.totalSearches) + searchTime) / (prev.totalSearches + 1)
        }));

        toast({
          title: "🧠 Réponse intelligente",
          description: `Trouvée par IA dans l'historique (${searchTime}ms)`,
        });

        return {
          found: true,
          response: backendResult.response,
          confidence: backendResult.confidence,
          source: 'backend_similar',
          searchTime,
          originalQuestion: backendResult.originalQuestion
        };
      }

      // 4. NIVEAU 4: Aucune correspondance trouvée - utiliser RAG Pipeline
      const searchTime = Date.now() - startTime;
      
      setSearchStats(prev => ({
        ...prev,
        ragPipelineUses: prev.ragPipelineUses + 1,
        totalSearches: prev.totalSearches + 1,
        averageSearchTime: ((prev.averageSearchTime * prev.totalSearches) + searchTime) / (prev.totalSearches + 1)
      }));

      console.log('🔍 Aucune correspondance trouvée - utilisation du RAG Pipeline');
      
      return {
        found: false,
        source: 'rag_pipeline',
        searchTime
      };

    } catch (error) {
      console.error('Erreur lors de la recherche hybride:', error);
      
      const searchTime = Date.now() - startTime;
      return {
        found: false,
        source: 'rag_pipeline',
        searchTime
      };
    } finally {
      setIsSearching(false);
    }
  }, [findExactMatch, findSimilarMatch, cacheResponse, toast]);

  // Sauvegarder une nouvelle réponse après RAG Pipeline
  const saveNewResponse = useCallback(async (
    question: string,
    response: string,
    confidence: number = 0.95,
    responseTime: number = 0
  ) => {
    // Sauvegarder localement
    cacheResponse(question, response, confidence, responseTime);
    
    // Sauvegarder sur le backend
    await historySearchService.saveToHistory(question, response, confidence, responseTime);
    
    console.log('💾 Nouvelle réponse sauvegardée dans l\'historique');
  }, [cacheResponse]);

  // Obtenir les statistiques complètes
  const getCompleteStats = useCallback(async () => {
    const localStats = getCacheStats();
    const backendStats = await historySearchService.getHistoryStats();
    
    return {
      local: localStats,
      backend: backendStats,
      search: searchStats,
      efficiency: {
        cacheHitRate: ((searchStats.localExactHits + searchStats.localSimilarHits) / searchStats.totalSearches) * 100 || 0,
        averageSearchTime: searchStats.averageSearchTime,
        ragUsageRate: (searchStats.ragPipelineUses / searchStats.totalSearches) * 100 || 0
      }
    };
  }, [getCacheStats, searchStats]);

  // Obtenir les questions fréquentes pour suggestions
  const getFrequentQuestions = useCallback(async (limit: number = 5) => {
    return await historySearchService.getFrequentQuestions(limit);
  }, []);

  return {
    searchResponse,
    saveNewResponse,
    getCompleteStats,
    getFrequentQuestions,
    isSearching,
    searchStats
  };
};
