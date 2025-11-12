"""
Flask Web Server for EGroupware RAG System
"""

import logging
import yaml
import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from typing import Dict, Any

from app.egroupware import EGroupwareClient
from app.embeddings import EmbeddingService
from app.mariadb_database import MariaDBDatabase
from app.llm import LLMService
from app.rag import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the correct paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(base_dir, 'templates')

# Initialize Flask app with correct template folder
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'your-secret-key-change-this-in-production'
CORS(app)

# Global services
config = None
embedding_service = None
llm_service = None


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import time
    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            # Use unbuffered read to avoid Docker/macOS file locking issues
            config_path = 'config.yaml'
            # Open with os.open to bypass Python's buffering
            fd = os.open(config_path, os.O_RDONLY)
            try:
                file_content = os.read(fd, os.fstat(fd).st_size)
                content = file_content.decode('utf-8')
                return yaml.safe_load(content)
            finally:
                os.close(fd)
        except OSError as e:
            if e.errno == 35 and attempt < max_retries - 1:
                # Resource deadlock - retry after a short delay
                logger.warning(f"Config read attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)
                continue
            logger.error(f"Failed to load config: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise


def initialize_services():
    """Initialize embedding and LLM services"""
    global config, embedding_service, llm_service

    try:
        # Always load configuration so server settings are available
        config = load_config()

        # Allow skipping heavy/remote service initialization in development
        # Set environment variable DEV_SKIP_SERVICES=1 to skip loading embeddings/LLM
        if os.environ.get('DEV_SKIP_SERVICES', '0') == '1':
            logger.info('DEV_SKIP_SERVICES=1 detected — skipping embedding and LLM initialization')
            return

        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(config['embeddings'])

        # Initialize LLM service
        logger.info("Initializing LLM service...")
        llm_service = LLMService(config['llm'])

        logger.info("✅ All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


def get_user_services():
    """Get user-specific service instances"""
    if 'user_id' not in session:
        raise ValueError("User not authenticated")

    user_id = session['user_id']
    egw_config = session.get('egroupware_config')

    if not egw_config:
        raise ValueError("EGroupware configuration not found in session")

    # Add chunking config to egroupware config
    chunking_config = config.get('chunking', {})
    egw_config['api_fetch_size'] = chunking_config.get('api_fetch_size', 500)

    # Create user-specific instances
    egroupware_client = EGroupwareClient(egw_config)

    # Create database with user context - using MariaDB only
    logger.info("Using MariaDB database backend")
    database = MariaDBDatabase(config, embedding_service.get_dimension())
    database.set_user_id(user_id)

    # Create RAG service with config
    rag_service = RAGService(
        egroupware_client=egroupware_client,
        embedding_service=embedding_service,
        database=database,
        llm_service=llm_service,
        config=config
    )

    return rag_service, database


@app.route('/')
def index():
    """Main page"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')


@app.route('/login')
def login_page():
    """Login page"""
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    """Handle login"""
    try:
        data = request.json
        base_url = data.get('base_url', '').rstrip('/')
        username = data.get('username', '')
        password = data.get('password', '')

        if not all([base_url, username, password]):
            return jsonify({'error': 'All fields are required'}), 400

        # Test connection
        egw_config = {
            'base_url': base_url,
            'username': username,
            'password': password,
            'timeout': 30
        }

        try:
            test_client = EGroupwareClient(egw_config)
            # Try a simple request to validate credentials
            test_client._request_single('/')
        except Exception as e:
            logger.error(f"EGroupware connection failed: {e}")
            return jsonify({'error': f'Failed to connect to EGroupware: {str(e)}'}), 401

        # Store in session
        session['user_id'] = f"{base_url}:{username}"
        session['egroupware_config'] = egw_config
        session['username'] = username

        logger.info(f"User logged in: {username}")

        return jsonify({
            'success': True,
            'message': 'Successfully connected to EGroupware'
        })

    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/logout', methods=['POST'])
def logout():
    """Handle logout"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/user-info', methods=['GET'])
def user_info():
    """Get current user information"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    # Extract username and base_url from the session
    username = session.get('username', 'Unknown')
    user_id = session.get('user_id', '')
    
    # Get email from EGroupware config if available
    email = ''
    egw_config = session.get('egroupware_config', {})
    if egw_config:
        email = egw_config.get('username', '')
        
    return jsonify({
        'username': username,
        'user_id': user_id,
        'name': username,  # Add name property for frontend compatibility
        'email': email     # Add email property for frontend compatibility
    })


# Add alias for /api/user (used by frontend)
@app.route('/api/user', methods=['GET'])
def user():
    """Get current user information (alias)"""
    return user_info()


@app.route('/api/index/<app_name>', methods=['POST'])
def index_app(app_name):
    """Index data from specific app"""
    try:
        rag_service, _ = get_user_services()

        if app_name == 'all':
            result = rag_service.index_all()
        else:
            result = rag_service.index_app(app_name)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint with optional fast mode"""
    try:
        rag_service, database = get_user_services()

        data = request.json
        query = data.get('query', '')
        app_filter = data.get('app_filter')
        top_k = data.get('top_k', 5)
        # Fast mode: skip LLM for instant results
        fast_mode = data.get('fast_mode', False)

        # Get user_id for debugging
        user_id = session.get('user_id', 'unknown')
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        # Debug the search parameters
        mode_str = "FAST" if fast_mode else "NORMAL"
        logger.info(f"Search request ({mode_str}): query='{query}', app_filter={app_filter}, top_k={top_k}, user_id={user_id}")

        # Get database stats before search for debugging
        db_stats = database.get_stats()
        logger.info(f"Database stats before search: {db_stats}")

        result = rag_service.search(query, app_filter, top_k, fast_mode=fast_mode)
        
        # Log search results
        result_count = len(result.get('results', []))
        logger.info(f"Search complete: found {result_count} results")
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get statistics"""
    try:
        rag_service, _ = get_user_services()
        result = rag_service.get_stats()
        return jsonify(result)

    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset database"""
    try:
        rag_service, _ = get_user_services()

        # Handle both JSON and non-JSON requests
        data = {}
        if request.is_json:
            data = request.json or {}
        elif request.form:
            data = request.form.to_dict()

        app_name = data.get('app_name')

        result = rag_service.reset_data(app_name)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'embeddings': embedding_service is not None,
            'llm': llm_service is not None
        }
    })


def main():
    """Main entry point"""
    try:
        # Initialize global services
        initialize_services()

        # Get server config
        server_config = config.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 5002)
        debug = server_config.get('debug', False)

        logger.info(f"Starting server on {host}:{port}")
        app.run(host=host, port=port, debug=debug)

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise


if __name__ == '__main__':
    main()
