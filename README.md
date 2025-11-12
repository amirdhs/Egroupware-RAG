# EGroupware RAG System

A complete Retrieval-Augmented Generation (RAG) system for EGroupware with MariaDB backend for semantic search and question answering.

## 🌟 Features

- **Multi-Application Support**: Index and search across Addressbook, Calendar, and InfoLog
- **Semantic Search**: Natural language search using state-of-the-art embeddings with hybrid scoring
- **User Isolation**: Multi-user support with complete data isolation
- **Flexible Embeddings**: Support for HuggingFace, OpenAI, and IONOS embedding models
- **LLM Integration**: Natural language responses using OpenAI or IONOS LLMs
- **MariaDB Backend**: Reliable relational database with vector search capabilities
- **Enhanced Pagination**: Improved API pagination with detailed logging for fetching large datasets
- **Web Interface**: Beautiful and intuitive web UI
- **RESTful API**: Complete API for integration
- **Docker Support**: Easy deployment with Docker Compose

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
│ EGroupware │    │   MariaDB   │
│   Client   │    │  Database   │
│(REST API)  │    │(Vector DB)  │
└────────────┘    └─────────────┘
```

## 📋 Requirements

- Python 3.11+
- Docker & Docker Compose (recommended)
- EGroupware instance with REST API access
- MariaDB 11.2+ (included in Docker setup)
- 2GB+ RAM (for embedding models)
- IONOS AI API key (or OpenAI/HuggingFace alternative)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/amirdhs/Egroupware-RAG.git
cd Egroupware-RAG
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

### 3. Run with Docker (Recommended)

```bash
docker compose up -d
```

This will start:
- MariaDB database (port 3307)
- RAG Application (port 5002)

### 4. Run Locally (Alternative)

```bash
pip install -r requirements.txt
python run.py
```

The application will start on `http://localhost:5002`

### 4. Access the Web Interface

1. Open `http://localhost:5002` in your browser
2. Login with your EGroupware credentials
3. Index your data (click "Index All Data")
4. Start searching!

## � Docker Deployment

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

### Docker Configuration

The `docker-compose.yml` includes:
- Automatic database initialization
- Health checks for MariaDB
- Volume persistence for data
- Network isolation

### Accessing the Application

- **Web Interface**: http://localhost:5002
- **MariaDB**: localhost:3307 (from host)

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

#### IONOS (Recommended, Paid, Cloud)
```yaml
embeddings:
  provider: "ionos"
  api_key: "your_ionos_token"
  api_url: "https://openai.inference.de-txl.ionos.com/v1"
  model: "BAAI/bge-m3"
```

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

### LLM Providers

#### IONOS (Recommended)
```yaml
llm:
  provider: "ionos"
  api_key: "your_ionos_token"
  api_url: "https://openai.inference.de-txl.ionos.com/v1"
  model: "meta-llama/Llama-3.3-70B-Instruct"
  temperature: 0.3
  max_tokens: 600
```

#### OpenAI
```yaml
llm:
  provider: "openai"
  api_key: "sk-..."
  model: "gpt-4o-mini"
  temperature: 0.7
```

### Database Configuration

**MariaDB Backend** (Current implementation):
- Hybrid search with cosine similarity + text matching
- User isolation with proper indexing
- Persistent storage with Docker volumes
- Full SQL query capabilities


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

## 🔍 Key Features Explained

### Enhanced API Pagination
- Fetches large datasets (2500+ entries) from EGroupware
- Detailed logging for debugging
- Safety limits to prevent infinite loops
- Support for sync-token based pagination

### Hybrid Search
- Combines semantic similarity with text matching
- Boosts exact matches and keyword relevance
- Configurable weighting between semantic and lexical search

### User Isolation
- Each user has completely isolated data
- Session-based authentication
- No cross-user data leakage

### Multi-Collection Support
- Automatically discovers group-specific collections
- Fetches from default, shared, global, and public collections
- Deduplicates entries across collections

## 📖 Documentation

- **[DOCKER.md](DOCKER.md)**: Docker deployment guide
- **[MARIADB_MIGRATION.md](MARIADB_MIGRATION.md)**: Database migration guide
- **[MARIADB_ONLY_SETUP.md](MARIADB_ONLY_SETUP.md)**: MariaDB-only configuration
- **[INFOLOG_ISSUE_ANALYSIS.md](INFOLOG_ISSUE_ANALYSIS.md)**: InfoLog pagination analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🐛 Known Issues

- **InfoLog Pagination**: EGroupware API has a hard limit of 500 items per collection. The system discovers multiple collections to fetch all entries.
- **API Response Time**: First-time indexing can take several minutes for large datasets.

## 💡 Tips

1. **Increase API fetch size** in `config.yaml` for faster indexing (default: 1000)
2. **Use Docker** for easier deployment and database management
3. **Check logs** with `docker compose logs -f` for troubleshooting
4. **Test the API** directly with curl or Postman before indexing large datasets

## 🆘 Support

For issues and questions:
1. Check the documentation files in the repository
2. Review the [INFOLOG_ISSUE_ANALYSIS.md](INFOLOG_ISSUE_ANALYSIS.md) for common problems
3. Open an issue on GitHub

## ⚡ Performance

- **Indexing Speed**: ~100 documents/minute (depends on API response time)
- **Search Speed**: <1 second for typical queries
- **Memory Usage**: ~2GB for embedding models + MariaDB
- **Storage**: ~1MB per 1000 indexed documents
