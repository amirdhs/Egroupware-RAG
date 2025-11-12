# EGroupware RAG System

A complete Retrieval-Augmented Generation (RAG) system for EGroupware with MariaDB backend for semantic search and question answering.

## Features

- **Multi-Application Support**: Index and search across Addressbook, Calendar, and InfoLog
- **Semantic Search**: Natural language search using state-of-the-art embeddings with hybrid scoring
- **User Isolation**: Multi-user support with complete data isolation
- **Flexible Embeddings**: Support for HuggingFace, OpenAI, and IONOS embedding models
- **LLM Integration**: Natural language responses using OpenAI or IONOS LLMs
- **MariaDB Backend**: Reliable relational database with vector search capabilities

##  Architecture

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
│ EGroupware │    │   MariaDB   │
│   Client   │    │  Database   │
│(REST API)  │    │(Vector DB)  │
└────────────┘    └─────────────┘
```


### 2. Configure the System

Copy and edit `config.yaml`:

```yaml
# EGroupware Connection
egroupware:
  base_url: "https://your-egroupware.com/egw/groupdav.php/username"
  username: "your_username"
  password: "your_password"
  timeout: 30

# Embedding Configuration
embeddings:
  provider: "ionos"  # or "huggingface", "openai"
  api_key: "your_ionos_api_key"
  api_url: "https://openai.inference.de-txl.ionos.com/v1"
  model: "BAAI/bge-m3"

# LLM Configuration
llm:
  provider: "ionos"  # or "openai"
  api_key: "your_ionos_api_key"
  api_url: "https://openai.inference.de-txl.ionos.com/v1"
  model: "meta-llama/Llama-3.3-70B-Instruct"
  temperature: 0.3
  max_tokens: 600

# Database Configuration (MariaDB only)
mariadb:
  host: "mariadb"  # Docker service name
  port: 3306
  database: "rag_vectors"
  user: "rag_user"
  password: "rag_password"

# Chunking Configuration
chunking:
  api_fetch_size: 1000  # Items per API request
  text_chunk_size: 1000
  text_chunk_overlap: 200
  embedding_batch_size: 32
```

### 3. Run with Docker 

```bash
docker compose up -d
```

This will start:
- MariaDB database (port 3307)
- RAG Application (port 5002)


The application will start on `http://localhost:5002`


##  Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Services included:**
- **MariaDB 11.2**: Vector database (port 3307)
- **RAG Application**: Flask web server (port 5002)



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


## 📚 Directory Structure

```
Egroupware-RAG/
├── app/
│   ├── __init__.py
│   ├── server.py            # Flask web server
│   ├── egroupware.py        # EGroupware API client with pagination
│   ├── embeddings.py        # Embedding service (IONOS/OpenAI/HF)
│   ├── mariadb_database.py  # MariaDB vector database
│   ├── llm.py              # LLM service
│   └── rag.py              # Core RAG logic
├── templates/
│   ├── index.html          # Main UI
│   └── login.html          # Login page
├── docker/
│   └── init-db.sql         # MariaDB initialization
├── config.yaml             # Main configuration
├── docker-compose.yml      # Docker services
├── requirements.txt        # Python dependencies
├── run.py                 # Entry point
├── Dockerfile             # Docker image definition
└── README.md             # This file
```


### Hybrid Search
- Combines semantic similarity with text matching
- Boosts exact matches and keyword relevance
- Configurable weighting between semantic and lexical search

### User Isolation
- Each user has completely isolated data
- Session-based authentication
- No cross-user data leakage


## Performance

- **Indexing Speed**: ~100 documents/minute (depends on API response time)
- **Search Speed**: <1 second for typical queries
- **Memory Usage**: ~2GB for embedding models + MariaDB
- **Storage**: ~1MB per 1000 indexed documents
