# 🏦 Banking Chatbot - AI-Powered Customer Service Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated banking chatbot leveraging cutting-edge AI technologies including **RAG (Retrieval-Augmented Generation)**, **LoRA fine-tuning** on **Llama 3.1 8B**, and **multi-modal voice integration** to provide intelligent, context-aware banking assistance.

## 🌟 Key Features

### 🤖 Hybrid AI Architecture
- **RAG System**: Retrieval-Augmented Generation using ChromaDB vector database and TF-IDF similarity search
- **Fine-tuned LLM**: Llama 3.1 8B model fine-tuned with LoRA (Low-Rank Adaptation) on banking-specific dataset
- **Intelligent Context Retrieval**: Multi-level history system with semantic search and similarity scoring
- **GPU Acceleration**: CUDA-optimized inference with 4-bit quantization for efficient memory usage

### 🎙️ Voice Integration
- **Speech-to-Text**: OpenAI Whisper model for accurate voice recognition
- **Text-to-Speech**: Google TTS (gTTS) and Coqui TTS for natural voice responses
- **Multi-language Support**: Optimized for French and English banking queries

### 💻 Modern Web Interface
- **React Frontend**: Built with React 18.3, TypeScript, and Vite
- **UI Components**: Shadcn/ui with Radix UI primitives for accessible, beautiful interfaces
- **Responsive Design**: TailwindCSS with dark/light theme support
- **Real-time Chat**: WebSocket-ready architecture for instant responses

### 📊 Advanced Features
- **Conversation History**: Persistent chat history with semantic search
- **Context-Aware Responses**: Maintains conversation context across sessions
- **Performance Monitoring**: Response time tracking and similarity scoring
- **Health Checks**: Comprehensive API health monitoring endpoints

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  Port 8080 - Modern UI with Voice Controls                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend Services                            │
├─────────────────────────────────────────────────────────────┤
│  RAG API (FastAPI)          │  Speech Service (FastAPI)     │
│  Port 8000                  │  Port 8004                    │
│  - Llama 3.1 8B + LoRA     │  - Whisper STT                │
│  - ChromaDB Vector Store    │  - gTTS/Coqui TTS             │
│  - TF-IDF Retrieval        │  - Audio Processing           │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Knowledge Base & Models                         │
│  - wasifis/bank-assistant-qa Dataset                        │
│  - ChromaDB Vector Database                                 │
│  - Fine-tuned LoRA Adapters                                 │
└─────────────────────────────────────────────────────────────┘
```

## 📸 Screenshots

### Main Interface
<img src="/IMG_1603.png" width="400" alt="Welcome screen with quick action buttons">
*Welcome screen with quick action buttons*

### Account Information Query
<img src="/IMG_1607.png" width="400" alt="Answering about minimum balance requirements">
*Answering about minimum balance requirements for NUST accounts*

### Interactive Conversation
<img src="/IMG_1615.png" width="400" alt="Real-time conversation">
*Real-time conversation with confidence scoring*

### Contact Information
<img src="/IMG_1616.png" width="400" alt="Contact information">
*Providing bank department contact information with local cache*

### Transfer-Related Queries
<img src="images/IMG_1618.png" width="400" alt="Transfer questions">
*Suggested questions for transfer operations*

### Chat History
<img src="/IMG_1605.png" width="400" alt="Chat history">
*Complete chat history and session management*

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm/bun
- CUDA-compatible GPU (optional, for faster inference)
- 16GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/banking-chatbot.git
cd banking-chatbot
```

2. **Install Python dependencies**
```bash
# For full development environment
pip install -r requirements.txt

# Or minimal production setup
pip install -r requirements-backend-minimal.txt
```

3. **Install Frontend dependencies**
```bash
cd "chat-bank-nexus-main(frontend v0)"
npm install
# or
bun install
```

4. **Download and prepare models**
```bash
# Download Llama 3.1 8B model
python download_llama_final.py

# Prepare the knowledge base
python rag_system_wasifis.py
```

### Running the Application

#### Option 1: Professional Launcher (Recommended)
```bash
python start_professional_chatbot.py
```
This will automatically start all services in the correct order.

#### Option 2: Manual Start

**Terminal 1 - RAG API:**
```bash
uvicorn backend_rag_api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Speech Service:**
```bash
uvicorn compatible_speech_service:app --host 0.0.0.0 --port 8004
```

**Terminal 3 - Frontend:**
```bash
cd "chat-bank-nexus-main(frontend v0)"
npm run dev
```

Access the application at: `http://localhost:8080`

## 📚 Technology Stack

### Backend
- **Framework**: FastAPI 0.104+ with async/await support
- **AI/ML**: 
  - PyTorch 2.0+ with CUDA support
  - Transformers 4.35+ (Hugging Face)
  - PEFT 0.6+ (Parameter-Efficient Fine-Tuning)
  - Accelerate 0.24+ for distributed training
  - BitsAndBytes for 4-bit quantization
- **Vector Database**: ChromaDB 0.4+ with persistent storage
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Retrieval**: Scikit-learn TF-IDF + Cosine Similarity
- **Speech**: 
  - OpenAI Whisper (base model)
  - gTTS (Google Text-to-Speech)
  - Coqui TTS (optional)

### Frontend
- **Framework**: React 18.3 with TypeScript 5.5
- **Build Tool**: Vite 5.4
- **UI Library**: 
  - Shadcn/ui components
  - Radix UI primitives
  - Lucide React icons
- **Styling**: TailwindCSS 3.4 with custom animations
- **State Management**: TanStack Query 5.56
- **Routing**: React Router DOM 6.26
- **Forms**: React Hook Form 7.53 + Zod validation
- **Theme**: next-themes for dark/light mode

### Data & Training
- **Dataset**: wasifis/bank-assistant-qa (Hugging Face)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training Framework**: TRL (Transformer Reinforcement Learning)
- **Monitoring**: Weights & Biases, TensorBoard

## 📁 Project Structure

```
chatbot/
├── backend_rag_api.py              # Main RAG API service
├── backend_rag_fast.py             # Optimized fast version
├── compatible_speech_service.py    # Speech service API
├── professional_speech_service.py  # Advanced speech features
├── start_professional_chatbot.py   # Coordinated launcher
├── banking_knowledge_base.py       # Knowledge base management
├── backend_chat_history.py         # Conversation history system
│
├── models/                         # AI models directory
│   └── Llama-3.1-8B-Instruct/     # Base model
│
├── llama_banking_final_fidelity/  # Fine-tuned LoRA adapters
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
├── chroma_db_wasifis/             # Vector database
│
├── banking_documents/              # Knowledge base documents
│   ├── comptes_bancaires.txt
│   ├── cartes_bancaires.txt
│   ├── prets_credits.txt
│   └── faq.json
│
├── chat-bank-nexus-main(frontend v0)/  # React frontend
│   ├── src/
│   │   ├── components/            # React components
│   │   ├── hooks/                 # Custom hooks
│   │   └── lib/                   # Utilities
│   ├── package.json
│   └── vite.config.ts
│
├── requirements.txt                # Python dependencies
├── requirements-backend-minimal.txt
├── requirements-training.txt
└── README.md
```

## 🎯 API Endpoints

### RAG API (Port 8000)

- `GET /` - API status
- `GET /health` - Health check with model status
- `POST /chat` - Main chat endpoint
  ```json
  {
    "message": "Comment ouvrir un compte bancaire?",
    "conversation_id": "optional-id"
  }
  ```
- `POST /history/search` - Search conversation history
- `GET /history/all` - Get all conversations
- `DELETE /history/clear` - Clear history

### Speech Service (Port 8004)

- `POST /transcribe` - Convert speech to text (Whisper)
- `POST /synthesize` - Convert text to speech (TTS)
- `GET /health` - Service health check

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Model Configuration
MODEL_PATH=./models/Llama-3.1-8B-Instruct
ADAPTER_PATH=./llama_banking_final_fidelity
DEVICE=cuda  # or cpu

# API Configuration
RAG_API_PORT=8000
SPEECH_API_PORT=8004
FRONTEND_PORT=8080

# Hugging Face (for model downloads)
HF_TOKEN=your_huggingface_token_here
```

## 🧪 Testing

```bash
# Test the RAG system
python test_banking_chatbot.py

# Test fine-tuned model
python test_finetuned_llama.py

# Test speech integration
python test_whisper_ffmpeg.py

# Run complete integration tests
python test_integration_complete.py
```

## 📈 Performance

- **Response Time**: 1-3 seconds (with GPU)
- **Similarity Accuracy**: 85%+ on banking queries
- **Model Size**: ~4.5GB (quantized)
- **Memory Usage**: ~8GB VRAM (GPU) / ~16GB RAM (CPU)

## 🛠️ Development

### Training Custom Model

```bash
# Prepare dataset
python prepare_banking_dataset.py

# Start fine-tuning
python final_high_fidelity.py

# Resume training if interrupted
python resume_final_training.py
```

### Adding New Banking Documents

1. Add documents to `banking_documents/`
2. Rebuild vector database:
```bash
python rag_system_wasifis.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Llama 3.1** by Meta AI
- **Hugging Face** for transformers and datasets
- **wasifis/bank-assistant-qa** dataset
- **OpenAI Whisper** for speech recognition
- **Shadcn/ui** for beautiful UI components






