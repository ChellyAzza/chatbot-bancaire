/**
 * Configuration API pour connexion avec votre RAG
 */

export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',  // Backend RAG
  FRONTEND_URL: 'http://localhost:8080',  // Votre frontend
  ENDPOINTS: {
    CHAT: '/chat',
    HEALTH: '/health'
  },
  TIMEOUT: 30000, // 30 secondes pour le RAG
};

export interface ChatRequest {
  message: string;
  conversation_id?: string;
}

export interface ChatResponse {
  response: string;
  response_time: number;
  contexts_found: number;
  similarity_score: number;
  conversation_id: string;
}

export class RAGApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_CONFIG.BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}${API_CONFIG.ENDPOINTS.CHAT}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async checkHealth(): Promise<{ status: string; model_loaded: boolean; gpu_available: boolean }> {
    const response = await fetch(`${this.baseUrl}${API_CONFIG.ENDPOINTS.HEALTH}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }
}

export const ragApiClient = new RAGApiClient();
