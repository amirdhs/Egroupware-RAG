# EGroupware RAG System

A complete Retrieval-Augmented Generation (RAG) system for EGroupware with support for multiple database backends for semantic search and question answering.

## 🌟 Features

- **Multi-Application Support**: Index and search across Addressbook, Calendar, and InfoLog
- **Semantic Search**: Natural language search using state-of-the-art embeddings
- **User Isolation**: Multi-user support with data isolation
- **Flexible Embeddings**: Support for HuggingFace, OpenAI, and IONOS embedding models
- **LLM Integration**: Natural language responses using OpenAI or IONOS LLMs
- **Dual Database Support**: Choose between Qdrant (vector database) or MariaDB (relational database)
- **Web Interface**: Beautiful and intuitive web UI
- **RESTful API**: Complete API for integration

## 🏗️ Architecture

```
┌─────────────────┐
│  Web Interface  │
│   (Flask + JS)  │
└────────┬────────┘
         │
┌────────▼────────┐
│   RAG Service   │
│  - Indexing     │
│  - Search       │
│  - LLM Gen      │
└─┬─────┬────┬───┘
  │     │    │
  │     │    └──────────┐
  │     │               │
┌─▼─────▼────┐    ┌────▼────────┐
│ EGroupware │    │   Qdrant    │
│   Client   │    │  Database   │
└────────────┘    └─────────────┘
```

## 📋 Requirements

- Python 3.8+
- EGroupware instance with REST API access
- 2GB+ RAM (for embedding models)
- Optional: Qdrant server (or use embedded mode)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the System

Edit `config.yaml` to set your preferences:

```yaml
# EGroupware Connection
egroupware:
  base_url: "https://your-domain.com/egroupware"
  username: "your_username"
  password: "your_password"

# Embedding Configuration
embeddings:
  provider: "ionos"  # or "huggingface", "openai"
  api_key: "your_api_key"
  model: "BAAI/bge-m3"

# LLM Configuration
llm:
  provider: "ionos"  # or "openai"
  api_key: "your_api_key"
  model: "meta-llama/Llama-3.3-70B-Instruct"

# Qdrant Configuration
qdrant:
  mode: "disk"  # or "memory", "server"
  path: "./qdrant_storage"
```

### 3. Run the Application

```bash
python run.py
```

The application will start on `http://localhost:5002`

### 4. Access the Web Interface

1. Open `http://localhost:5002` in your browser
2. Login with your EGroupware credentials
3. Index your data (click "Index All Data")
4. Start searching!

## 📊 Qdrant Configuration Modes

### Memory Mode (Development)
Fast but data is lost on restart:
```yaml
qdrant:
  mode: "memory"
```

### Disk Mode (Recommended)
Persistent storage on local disk:
```yaml
qdrant:
  mode: "disk"
  path: "./qdrant_storage"
```

### Server Mode (Production)
Connect to a Qdrant server:
```yaml
qdrant:
  mode: "server"
  host: "localhost"
  port: 6333
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
docker-compose up -d
```

This will start:
- Flask application (port 5002)
- Qdrant server (port 6333)

## 📡 API Endpoints

### Authentication
- `POST /login` - Login with EGroupware credentials
- `POST /logout` - Logout

### Data Management
- `POST /api/index/all` - Index all applications
- `POST /api/index/addressbook` - Index addressbook
- `POST /api/index/calendar` - Index calendar
- `POST /api/index/infolog` - Index infolog

### Search
- `POST /api/search` - Semantic search
  ```json
  {
    "query": "Find contacts from Berlin",
    "app_filter": "addressbook",
    "top_k": 5
  }
  ```

### System
- `GET /api/stats` - Get system statistics
- `POST /api/reset` - Reset database
- `GET /health` - Health check

## 🔧 Configuration Options

### Embedding Providers

#### HuggingFace (Free, Local)
```yaml
embeddings:
  provider: "huggingface"
  hf_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

#### OpenAI (Paid, Cloud)
```yaml
embeddings:
  provider: "openai"
  api_key: "sk-..."
  openai_model: "text-embedding-3-small"
```

#### IONOS (Paid, Cloud)
```yaml
embeddings:
  provider: "ionos"
  api_key: "your_token"
  api_url: "https://openai.inference.de-txl.ionos.com/v1"
  model: "BAAI/bge-m3"
```

### LLM Providers

Similar configuration for `llm` section with OpenAI or IONOS providers.


## 📚 Directory Structure

```
RAG/
├── app/
│   ├── __init__.py
│   ├── server.py        # Flask web server
│   ├── egroupware.py    # EGroupware API client
│   ├── embeddings.py    # Embedding service
│   ├── database.py      # Qdrant database
│   ├── llm.py          # LLM service
│   └── rag.py          # Core RAG logic
├── templates/
│   ├── index.html      # Main UI
│   └── login.html      # Login page
├── config.yaml         # Configuration
├── requirements.txt    # Python dependencies
├── run.py             # Entry point
└── README.md          # This file
```
