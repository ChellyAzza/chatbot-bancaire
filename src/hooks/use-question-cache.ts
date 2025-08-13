import { useState, useCallback } from 'react';
import { useToast } from './use-toast';

interface CachedResponse {
  question: string;
  response: string;
  timestamp: number;
  hits: number;
  confidence: number;
  responseTime: number;
}

interface QuestionCache {
  [questionHash: string]: CachedResponse;
}

export const useQuestionCache = () => {
  const { toast } = useToast();
  const [cacheStats, setCacheStats] = useState({ hits: 0, misses: 0 });

  // Normaliser une question pour la comparaison
  const normalizeQuestion = useCallback((question: string): string => {
    return question
      .toLowerCase()
      .trim()
      .replace(/[^\w\s]/g, '') // Supprimer la ponctuation
      .replace(/\s+/g, ' ') // Normaliser les espaces
      .replace(/\b(le|la|les|un|une|des|du|de|d')\b/g, '') // Supprimer articles
      .trim();
  }, []);

  // Générer un hash simple pour la question
  const hashQuestion = useCallback((question: string): string => {
    const normalized = normalizeQuestion(question);
    let hash = 0;
    for (let i = 0; i < normalized.length; i++) {
      const char = normalized.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }, [normalizeQuestion]);

  // Charger le cache depuis localStorage
  const loadCache = useCallback((): QuestionCache => {
    try {
      const cached = localStorage.getItem('bankbot_question_cache');
      return cached ? JSON.parse(cached) : {};
    } catch (error) {
      console.warn('Erreur lors du chargement du cache:', error);
      return {};
    }
  }, []);

  // Sauvegarder le cache dans localStorage
  const saveCache = useCallback((cache: QuestionCache) => {
    try {
      // Limiter à 50 entrées pour éviter de surcharger localStorage
      const entries = Object.entries(cache);
      if (entries.length > 50) {
        // Garder les 50 plus récentes et les plus utilisées
        const sorted = entries.sort((a, b) => {
          const scoreA = a[1].hits * 0.3 + (Date.now() - a[1].timestamp) * -0.7;
          const scoreB = b[1].hits * 0.3 + (Date.now() - b[1].timestamp) * -0.7;
          return scoreB - scoreA;
        });
        const limitedCache = Object.fromEntries(sorted.slice(0, 50));
        localStorage.setItem('bankbot_question_cache', JSON.stringify(limitedCache));
      } else {
        localStorage.setItem('bankbot_question_cache', JSON.stringify(cache));
      }
    } catch (error) {
      console.warn('Erreur lors de la sauvegarde du cache:', error);
    }
  }, []);

  // Rechercher une réponse exacte dans le cache local
  const findExactMatch = useCallback((question: string): CachedResponse | null => {
    const cache = loadCache();
    const questionHash = hashQuestion(question);
    
    if (cache[questionHash]) {
      // Incrémenter le compteur de hits
      cache[questionHash].hits += 1;
      saveCache(cache);
      
      setCacheStats(prev => ({ ...prev, hits: prev.hits + 1 }));
      
      return cache[questionHash];
    }
    
    setCacheStats(prev => ({ ...prev, misses: prev.misses + 1 }));
    return null;
  }, [hashQuestion, loadCache, saveCache]);

  // Rechercher des questions similaires (logique simple basée sur mots-clés)
  const findSimilarMatch = useCallback((question: string): CachedResponse | null => {
    const cache = loadCache();
    const normalized = normalizeQuestion(question);
    const questionWords = normalized.split(' ').filter(word => word.length > 2);
    
    let bestMatch: CachedResponse | null = null;
    let bestScore = 0;
    
    Object.values(cache).forEach(cached => {
      const cachedNormalized = normalizeQuestion(cached.question);
      const cachedWords = cachedNormalized.split(' ').filter(word => word.length > 2);
      
      // Calculer la similarité basée sur les mots communs
      const commonWords = questionWords.filter(word => cachedWords.includes(word));
      const similarity = commonWords.length / Math.max(questionWords.length, cachedWords.length);
      
      // Seuil de similarité de 0.7 (70% de mots en commun)
      if (similarity > 0.7 && similarity > bestScore) {
        bestScore = similarity;
        bestMatch = cached;
      }
    });
    
    if (bestMatch) {
      toast({
        title: "Réponse similaire trouvée",
        description: `Basée sur une question similaire (${Math.round(bestScore * 100)}% de correspondance)`,
      });
    }
    
    return bestMatch;
  }, [normalizeQuestion, loadCache, toast]);

  // Sauvegarder une nouvelle réponse dans le cache
  const cacheResponse = useCallback((
    question: string, 
    response: string, 
    confidence: number = 0.95,
    responseTime: number = 0
  ) => {
    const cache = loadCache();
    const questionHash = hashQuestion(question);
    
    cache[questionHash] = {
      question,
      response,
      timestamp: Date.now(),
      hits: 1,
      confidence,
      responseTime
    };
    
    saveCache(cache);
  }, [hashQuestion, loadCache, saveCache]);

  // Vider le cache
  const clearCache = useCallback(() => {
    localStorage.removeItem('bankbot_question_cache');
    setCacheStats({ hits: 0, misses: 0 });
    toast({
      title: "Cache vidé",
      description: "Le cache des questions a été supprimé.",
    });
  }, [toast]);

  // Obtenir les statistiques du cache
  const getCacheStats = useCallback(() => {
    const cache = loadCache();
    const entries = Object.values(cache);
    
    return {
      totalEntries: entries.length,
      totalHits: entries.reduce((sum, entry) => sum + entry.hits, 0),
      averageConfidence: entries.length > 0 
        ? entries.reduce((sum, entry) => sum + entry.confidence, 0) / entries.length 
        : 0,
      cacheHitRate: cacheStats.hits + cacheStats.misses > 0 
        ? (cacheStats.hits / (cacheStats.hits + cacheStats.misses)) * 100 
        : 0,
      ...cacheStats
    };
  }, [loadCache, cacheStats]);

  return {
    findExactMatch,
    findSimilarMatch,
    cacheResponse,
    clearCache,
    getCacheStats,
    cacheStats
  };
};
