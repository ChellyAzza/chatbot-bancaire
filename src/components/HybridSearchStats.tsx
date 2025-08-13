import { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart3, 
  Zap, 
  Brain, 
  Database, 
  Clock, 
  TrendingUp,
  RefreshCw,
  X
} from "lucide-react";
import { useHybridSearch } from "@/hooks/use-hybrid-search";

interface HybridSearchStatsProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HybridSearchStats = ({ isOpen, onClose }: HybridSearchStatsProps) => {
  const { getCompleteStats, searchStats } = useHybridSearch();
  const [stats, setStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const loadStats = async () => {
    setIsLoading(true);
    try {
      const completeStats = await getCompleteStats();
      setStats(completeStats);
    } catch (error) {
      console.error('Erreur lors du chargement des statistiques:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadStats();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-y-auto bg-background border-border">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <BarChart3 className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-bold">Statistiques de Recherche Hybride</h2>
                <p className="text-sm text-muted-foreground">
                  Performance et efficacité du système intelligent
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={loadStats}
                disabled={isLoading}
                className="h-9 w-9"
              >
                <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-9 w-9"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : stats ? (
            <div className="space-y-6">
              {/* Métriques principales */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 border-green-200 dark:border-green-800">
                  <div className="flex items-center gap-3">
                    <Zap className="h-8 w-8 text-green-600" />
                    <div>
                      <p className="text-2xl font-bold text-green-700 dark:text-green-400">
                        {searchStats.localExactHits}
                      </p>
                      <p className="text-sm text-green-600 dark:text-green-500">Cache Exact</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-950 dark:to-cyan-950 border-blue-200 dark:border-blue-800">
                  <div className="flex items-center gap-3">
                    <Brain className="h-8 w-8 text-blue-600" />
                    <div>
                      <p className="text-2xl font-bold text-blue-700 dark:text-blue-400">
                        {searchStats.backendSimilarHits}
                      </p>
                      <p className="text-sm text-blue-600 dark:text-blue-500">IA Similaire</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-4 bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-950 dark:to-violet-950 border-purple-200 dark:border-purple-800">
                  <div className="flex items-center gap-3">
                    <Database className="h-8 w-8 text-purple-600" />
                    <div>
                      <p className="text-2xl font-bold text-purple-700 dark:text-purple-400">
                        {searchStats.ragPipelineUses}
                      </p>
                      <p className="text-sm text-purple-600 dark:text-purple-500">RAG Pipeline</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-4 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950 border-orange-200 dark:border-orange-800">
                  <div className="flex items-center gap-3">
                    <Clock className="h-8 w-8 text-orange-600" />
                    <div>
                      <p className="text-2xl font-bold text-orange-700 dark:text-orange-400">
                        {Math.round(searchStats.averageSearchTime)}ms
                      </p>
                      <p className="text-sm text-orange-600 dark:text-orange-500">Temps Moyen</p>
                    </div>
                  </div>
                </Card>
              </div>

              {/* Efficacité */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Efficacité du Système
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-primary mb-2">
                      {stats.efficiency?.cacheHitRate?.toFixed(1) || 0}%
                    </div>
                    <p className="text-sm text-muted-foreground">Taux de Cache Hit</p>
                    <Badge variant="secondary" className="mt-2">
                      {stats.efficiency?.cacheHitRate > 50 ? 'Excellent' : 
                       stats.efficiency?.cacheHitRate > 30 ? 'Bon' : 'À améliorer'}
                    </Badge>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-primary mb-2">
                      {stats.efficiency?.ragUsageRate?.toFixed(1) || 0}%
                    </div>
                    <p className="text-sm text-muted-foreground">Utilisation RAG</p>
                    <Badge variant="secondary" className="mt-2">
                      {stats.efficiency?.ragUsageRate < 50 ? 'Optimisé' : 
                       stats.efficiency?.ragUsageRate < 70 ? 'Normal' : 'Élevé'}
                    </Badge>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-primary mb-2">
                      {searchStats.totalSearches}
                    </div>
                    <p className="text-sm text-muted-foreground">Total Recherches</p>
                    <Badge variant="secondary" className="mt-2">
                      Session Actuelle
                    </Badge>
                  </div>
                </div>
              </Card>

              {/* Cache Local */}
              {stats.local && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Cache Local</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold">{stats.local.totalEntries}</p>
                      <p className="text-sm text-muted-foreground">Entrées</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{stats.local.totalHits}</p>
                      <p className="text-sm text-muted-foreground">Hits Total</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{Math.round(stats.local.averageConfidence * 100)}%</p>
                      <p className="text-sm text-muted-foreground">Confiance Moy.</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{stats.local.cacheHitRate?.toFixed(1) || 0}%</p>
                      <p className="text-sm text-muted-foreground">Taux de Hit</p>
                    </div>
                  </div>
                </Card>
              )}

              {/* Backend Stats */}
              {stats.backend && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Historique Backend</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold">{stats.backend.total_questions}</p>
                      <p className="text-sm text-muted-foreground">Questions Totales</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{Math.round(stats.backend.average_confidence * 100)}%</p>
                      <p className="text-sm text-muted-foreground">Confiance Moy.</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{stats.backend.cache_hit_rate?.toFixed(1) || 0}%</p>
                      <p className="text-sm text-muted-foreground">Taux Global</p>
                    </div>
                  </div>
                </Card>
              )}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-muted-foreground">Aucune donnée disponible</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};
